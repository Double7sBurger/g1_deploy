"""Export actors and verify the encoder-space deployment interface offline."""

import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys

import gymnasium as gym
import numpy as np
import torch
import warp as wp
import yaml

wp.config.enable_backward = False
torch.set_float32_matmul_precision('highest')
from isaaclab.app import launch_simulation
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from rsl_rl.runners import OnPolicyRunner
from motion_tracking.tasks.motion_tracking.agents.rsl_rl_ppo_cfg import G1MotionTrackingPPORunnerCfg

DEPLOY = Path('/home/henry/workspace/g1_deploy')
PACKAGES = DEPLOY / 'policies/walk_sysid_friction_20261029'
sys.path.insert(0, str(DEPLOY))
from g1_deploy.tracking import G1TrackingPolicyRunner


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor(value):
    return value.torch if hasattr(value, 'torch') else value


def array(value):
    return tensor(value).detach().cpu().numpy()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def configured_gain(groups, name, field):
    matches = [g for g in groups.values() if any(re.fullmatch(p, name) for p in g['joint_names_expr'])]
    assert len(matches) == 1, name
    value = matches[0][field]
    if isinstance(value, dict):
        matches = [v for p, v in value.items() if re.fullmatch(p, name)]
        assert len(matches) == 1, name
        value = matches[0]
    return float(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['old_implicit200', 'old_explicit400'], required=True)
    args = parser.parse_args()
    entries = [e for e in json.loads((PACKAGES / 'selection.json').read_text())['entries']
               if e['candidate']['mode'] == args.mode]
    if args.mode == 'old_implicit200':
        from motion_tracking.tasks.motion_tracking.motion_tracking_allfive_env_cfg import MotionTrackingAllFiveEnvCfg
        cfg = MotionTrackingAllFiveEnvCfg()
    else:
        from motion_tracking.tasks.motion_tracking.motion_tracking_sysid_env_cfg import MotionTrackingSysIdEnvCfg
        cfg = MotionTrackingSysIdEnvCfg()
    cfg.scene.num_envs = 4
    cfg.seed = 20261029
    cfg.commands.motion.motion_file = str(PACKAGES / entries[0]['directory'] / 'motion.npz')
    cfg.observations.policy.enable_corruption = False
    cfg.events.add_base_mass = None
    cfg.events.push_robot = None
    cfg.events.physics_material.params['static_friction_range'] = (1., 1.)
    cfg.events.physics_material.params['dynamic_friction_range'] = (1., 1.)
    agent_cfg = handle_deprecated_rsl_rl_cfg(G1MotionTrackingPPORunnerCfg(), importlib.metadata.version('rsl-rl-lib'))
    assert agent_cfg.clip_actions is None and not agent_cfg.actor.obs_normalization
    reports = {}
    with launch_simulation(cfg, {'visualizer': None, 'visualizer_explicit': True}):
        env = gym.make('Motion-Tracking-G1-v0', cfg=cfg).unwrapped
        vec = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(vec, agent_cfg.to_dict(), log_dir=None, device=env.device)
        robot = env.scene['robot']
        action = env.action_manager.get_term('joint_pos')
        command = env.command_manager.get_term('motion')
        assert 'Newton' in env.sim.physics_manager.__name__
        env.reset()
        stiffness = torch.zeros((env.num_envs, robot.num_joints), device=env.device)
        damping = torch.zeros_like(stiffness)
        for actuator in robot.actuators.values():
            stiffness[:, actuator.joint_indices] = actuator.stiffness
            damping[:, actuator.joint_indices] = actuator.damping
        bias = action.sysid_bias if args.mode == 'old_implicit200' else robot.actuators['sysid'].encoder_bias
        assert bias.shape[-1] == robot.num_joints == 29
        for name in ('waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint'):
            assert (bias[:, robot.joint_names.index(name)] == 0).all()
        contract = {
            'schema_version': 1, 'policy_type': 'motion_tracking', 'task': 'Motion-Tracking-G1-v0',
            'joint_names': robot.joint_names, 'action_dim': robot.num_joints,
            'obs_1d_dim': 154, 'history_length': 1, 'obs_normalization': False,
            'obs_1d_terms': {'command': 58, 'motion_anchor_ori_b': 6, 'base_ang_vel': 3,
                             'joint_pos': 29, 'joint_vel': 29, 'actions': 29},
            'control_dt': env.step_dt, 'physics_dt': env.physics_dt,
            'default_joint_pos': array(robot.data.default_joint_pos)[0].tolist(),
            'default_joint_vel': array(robot.data.default_joint_vel)[0].tolist(),
            'stiffness': array(stiffness)[0].tolist(), 'damping': array(damping)[0].tolist(),
            'action_scale': array(action._scale)[0].tolist(), 'use_default_offset': True, 'clip_actions': None,
            'joint_pos_limits': array(robot.data.joint_pos_limits)[0].tolist(),
            'joint_vel_limits': array(robot.data.joint_vel_limits)[0].tolist(),
            'anchor_body_name': 'torso_link', 'root_body_name': 'pelvis',
            'quaternion_order': 'wxyz', 'orientation_encoding': 'matrix_first_two_columns_row_major',
            'motion_file': 'motion.npz', 'motion_metadata_file': 'motion.json',
            'asset_url': 'https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.2/Isaac/IsaacLab/Robots/Unitree/G1/g1_29dof.usd',
            'asset_variants': {'Physics': 'SimplifiedPhysX', 'left_hand': 'None', 'right_hand': 'None'},
            'training_engine': env.sim.physics_manager.__name__, 'hardware_validated': False,
            'joint_coordinate_convention': 'Encoder observations and position commands; no extra SysID bias or delay on hardware.',
            'training_sysid_encoder_bias': array(bias)[0].tolist(),
            'training_sysid_file': 'training/sysid_parameters.json',
        }
        np.testing.assert_allclose(array(action._offset)[0], contract['default_joint_pos'])
        for entry in entries:
            c = entry['candidate']
            folder = PACKAGES / entry['directory']
            checkpoint = folder / Path(c['checkpoint']).name
            experiment = json.loads((folder / 'training/experiment.json').read_text())
            saved_env = yaml.load((folder / 'training/params/env.yaml').read_text(), Loader=yaml.BaseLoader)
            saved_agent = yaml.safe_load((folder / 'training/params/agent.yaml').read_text())
            assert saved_agent['actor'] == agent_cfg.to_dict()['actor']
            assert float(saved_env['sim']['dt']) == env.physics_dt == experiment['physics_dt']
            assert int(saved_env['decimation']) * env.physics_dt == env.step_dt
            for field in ('stiffness', 'damping'):
                gains = [configured_gain(saved_env['scene']['robot']['actuators'], name, field)
                         for name in robot.joint_names]
                np.testing.assert_allclose(gains, contract[field], atol=1e-6, rtol=1e-6)
            assert sha(checkpoint) == c['checkpoint_sha256']
            assert sha(folder / 'motion.npz') == c['motion_sha256']
            assert sha(folder / 'training/sysid_parameters.json') == c['sysid_sha256'] == cfg.sysid_sha256
            runner.load(str(checkpoint))
            policy = runner.get_inference_policy(device=env.device)
            eager_cpu = copy.deepcopy(policy).cpu().eval()
            scripted = torch.jit.script(eager_cpu.as_jit().cpu().eval())
            generator = torch.Generator().manual_seed(20261029)
            samples = torch.randn(256, 154, generator=generator)
            with torch.inference_mode():
                expected = eager_cpu({'policy': samples})
                actual = scripted(samples)
            torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
            scripted.save(str(folder / 'policy.pt'))
            current = dict(contract, checkpoint=checkpoint.name, checkpoint_sha256=sha(checkpoint),
                           motion_sha256=c['motion_sha256'], policy_sha256=sha(folder / 'policy.pt'),
                           sysid_sha256=c['sysid_sha256'], training_seed=c['training_seed'],
                           training_iteration=c['iteration'], training_arm=c['arm'])
            write_json(folder / 'contract.json', current)
            deployment = G1TrackingPolicyRunner(folder)
            n = command.motion.time_step_total
            frames = [0, n // 4, n // 2, 3 * n // 4, n - 2]
            maxima = dict(observation=0., action=0., target=0.)
            previous = torch.zeros((env.num_envs, 29), device=env.device)
            for frame in frames:
                command.time_steps[:] = frame
                q, dq = command.joint_pos.clone(), command.joint_vel.clone()
                if frame:
                    q += torch.linspace(-.03, .03, 29, device=env.device)
                    dq += torch.linspace(-.1, .1, 29, device=env.device)
                root = torch.cat((command.body_pos_w[:, 0], command.body_quat_w[:, 0],
                                  command.body_lin_vel_w[:, 0], command.body_ang_vel_w[:, 0]), dim=-1)
                robot.write_joint_state_to_sim(q, dq)
                robot.write_root_state_to_sim(root)
                env.sim.forward()
                env.scene.update(env.physics_dt)
                command.time_steps[:] = frame - 1
                command._update_command()
                env.action_manager.process_action(previous)
                env.obs_buf = env.observation_manager.compute()
                obs = vec.get_observations()
                assert obs['policy'].shape[-1] == 154
                # The measured bias maps physical q to the encoder readings available on G1.
                q_encoder = q - bias
                q_motor = array(q_encoder)[0][np.argsort(deployment.policy_to_robot)]
                dq_motor = array(dq)[0][np.argsort(deployment.policy_to_robot)]
                quat = array(robot.data.root_quat_w)[0][[3, 0, 1, 2]]
                gyro = array(robot.data.root_ang_vel_b)[0]
                actual_obs = deployment.observe(q_motor, dq_motor, quat, gyro, frame_index=frame)
                expected_obs = array(obs['policy'])[0]
                np.testing.assert_allclose(actual_obs, expected_obs, rtol=1e-5, atol=2e-6)
                with torch.inference_mode():
                    expected_action = policy(obs)
                target, actual_action = deployment.step(q_motor, dq_motor, quat, gyro, elapsed_s=frame * env.step_dt)
                env.action_manager.process_action(expected_action)
                expected_target = array(action.processed_actions)[0]
                np.testing.assert_allclose(actual_action, array(expected_action)[0], rtol=1e-4, atol=2e-5)
                np.testing.assert_allclose(target[deployment.policy_to_robot], expected_target, rtol=1e-4, atol=2e-5)
                for key, diff in [('observation', actual_obs - expected_obs),
                                  ('action', actual_action - array(expected_action)[0]),
                                  ('target', target[deployment.policy_to_robot] - expected_target)]:
                    maxima[key] = max(maxima[key], float(abs(diff).max()))
                previous = torch.from_numpy(actual_action.copy()).to(env.device).repeat(env.num_envs, 1)
            report = {
                'protocol': 'Frozen Newton training interface vs current G1TrackingPolicyRunner, DR/noise disabled',
                'reference_frames': frames, 'max_abs_error': maxima, 'random_observation_samples': 256,
                'eager_cpu_vs_jit_cpu_max_abs_error': float((actual - expected).abs().max()),
                'encoder_bias_applied_only_to_simulated_sensor_readings': True,
                'saved_training_pd_gains_verified': True, 'hardware_validated': False,
                'policy_sha256': current['policy_sha256'], 'checkpoint_sha256': c['checkpoint_sha256'],
                'deployment_tracking_sha256': sha(DEPLOY / 'g1_deploy/tracking.py'),
            }
            write_json(folder / 'validation.json', report)
            reports[entry['directory']] = report
            print('VERIFIED', entry['directory'], json.dumps(report), flush=True)
        env.close()
    write_json(Path(__file__).with_name(f'validation_{args.mode}.json'), reports)


if __name__ == '__main__':
    main()
