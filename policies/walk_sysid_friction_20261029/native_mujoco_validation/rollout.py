"""Standalone MuJoCo closed-loop tests using the unmodified deployment runner."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import torch

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parent
DEPLOY = Path('/home/henry/workspace/g1_deploy')
PACKAGES = DEPLOY / 'policies/walk_sysid_friction_20261029'
sys.path.insert(0, str(DEPLOY))
from g1_deploy.tracking import G1TrackingPolicyRunner, rotation_matrix


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class NativeTrial:
    def __init__(self, directory):
        self.directory = directory
        self.folder = PACKAGES / directory
        self.runner = G1TrackingPolicyRunner(self.folder)
        self.contract = self.runner.contract
        self.mode = 'implicit200' if directory.startswith('fixed/') else 'explicit400'
        self.model_file = ROOT / 'models' / self.mode / 'model.mjb'
        self.info = json.loads(self.model_file.with_suffix('.json').read_text())
        assert self.info['model_sha256'] == sha(self.model_file)
        self.model = mujoco.MjModel.from_binary_path(str(self.model_file))
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep
        assert self.dt == self.contract['physics_dt']
        self.decimation = round(self.runner.control_dt / self.dt)
        assert abs(self.decimation * self.dt - .02) < 1e-10
        assert self.info['joint_order'] == self.runner.joint_names
        self.joint_ids = np.array([j['id'] for j in self.info['joints']])
        self.qids = self.model.jnt_qposadr[self.joint_ids]
        self.vids = self.model.jnt_dofadr[self.joint_ids]
        self.rootq = int(self.model.jnt_qposadr[self.info['free_joint']])
        self.rootv = int(self.model.jnt_dofadr[self.info['free_joint']])
        self.body_names = list(self.info['bodies'])
        self.body_ids = list(self.info['bodies'].values())
        self.pelvis = self.info['bodies']['pelvis']
        self.torso = self.info['bodies']['torso_link']
        self.floor = self.info['floor_geom']
        self.feet = [self.info['bodies'][n + '_ankle_roll_link'] for n in ('left', 'right')]
        self.tracked = [self.info['bodies'][n] for n in self.info['tracked_bodies']]
        self.ref = dict(np.load(self.folder / 'motion.npz'))
        names = json.loads((self.folder / 'motion.json').read_text())
        self.ref_bodies = [names['body_names'].index(n) for n in self.body_names]
        self.ref_tracked = [names['body_names'].index(n) for n in self.info['tracked_bodies']]
        self.ref_pelvis = names['body_names'].index('pelvis')
        self.ref_torso = names['body_names'].index('torso_link')
        self.ref_joints = [names['joint_names'].index(n) for n in self.runner.joint_names]
        self.bias = np.asarray(self.contract['training_sysid_encoder_bias'])
        params = json.loads((self.folder / 'training/sysid_parameters.json').read_text())
        self.lags = np.array([math.ceil(params['joints'].get(n.removesuffix('_joint'), {}).get(
            'parameters', {}).get('delay_steps', 0) * params['command_dt_s'] / self.dt - 1e-10)
                             for n in self.runner.joint_names], dtype=int)
        self.kp = np.asarray(self.contract['stiffness'])
        self.kd = np.asarray(self.contract['damping'])
        self.efforts = np.asarray(self.info['effort_limits'])
        self.vel_limits = np.asarray(self.contract['joint_vel_limits'])
        self.limits = np.asarray(self.contract['joint_pos_limits'])
        self.soft_limits = np.asarray(self.info['soft_joint_limits'])
        self.position_actuators = []
        if self.mode == 'implicit200':
            for i, j in enumerate(self.joint_ids):
                ids = np.flatnonzero(self.model.actuator_trnid[:, 0] == j)
                p = [a for a in ids if self.model.actuator_biasprm[a, 1] != 0]
                v = [a for a in ids if self.model.actuator_biasprm[a, 2] != 0]
                assert len(p) == len(v) == 1
                np.testing.assert_allclose(self.model.actuator_gainprm[p[0], 0], self.kp[i], rtol=1e-6)
                np.testing.assert_allclose(self.model.actuator_gainprm[v[0], 0], self.kd[i], rtol=1e-6)
                self.position_actuators.append(p[0])
        else:
            assert self.model.nu == 0
        self.velocity = np.zeros(6)
        self.contact_force = np.zeros(6)

    def reset(self, mu, seed, jitter):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        self.runner.reset()
        m.geom_friction[:, 0] = 1.
        m.geom_friction[self.floor, 0] = mu
        m.geom_priority[self.floor] = 1
        rng = np.random.default_rng(seed)
        pos = self.ref['body_pos_w'][0, self.ref_pelvis].astype(float).copy()
        rot = Rotation.from_quat(self.ref['body_quat_w'][0, self.ref_pelvis][[1, 2, 3, 0]])
        lin = self.ref['body_lin_vel_w'][0, self.ref_pelvis].astype(float).copy()
        ang = self.ref['body_ang_vel_w'][0, self.ref_pelvis].astype(float).copy()
        q = self.ref['joint_pos'][0, self.ref_joints].astype(float).copy()
        if jitter:
            pose_delta = rng.uniform(-1., 1., 6) * [.05, .05, .01, .1, .1, .2]
            pos += pose_delta[:3]
            rot = Rotation.from_euler('xyz', pose_delta[3:]) * rot
            vel_delta = rng.uniform(-1., 1., 6) * [.5, .5, .2, .52, .52, .78]
            lin += vel_delta[:3]
            ang += vel_delta[3:]
            q += rng.uniform(-.1, .1, 29)
            q = np.clip(q, self.soft_limits[:, 0], self.soft_limits[:, 1])
        d.qpos[self.rootq:self.rootq+3] = pos
        d.qpos[self.rootq+3:self.rootq+7] = rot.as_quat()[[3, 0, 1, 2]]
        d.qpos[self.qids] = q
        d.qvel[self.rootv:self.rootv+3] = lin
        d.qvel[self.rootv+3:self.rootv+6] = rot.inv().apply(ang)
        d.qvel[self.vids] = self.ref['joint_vel'][0, self.ref_joints]
        mujoco.mj_forward(m, d)
        self.history = None
        self.cursor = 0
        return {'qpos': d.qpos.copy().tolist(), 'qvel': d.qvel.copy().tolist()}

    def advance(self, target):
        if self.history is None:
            self.history = np.tile(target, (int(self.lags.max()) + 1, 1))
        self.history[self.cursor] = target
        delayed = self.history[(self.cursor - self.lags) % len(self.history), np.arange(29)]
        self.cursor = (self.cursor + 1) % len(self.history)
        if self.mode == 'implicit200':
            self.data.ctrl[self.position_actuators] = delayed + self.bias
        else:
            velocity = self.data.qvel[self.vids]
            effort = self.kp * (delayed - (self.data.qpos[self.qids] - self.bias)) - self.kd * velocity
            speed = np.clip(velocity, -2 * self.vel_limits, 2 * self.vel_limits)
            upper = np.minimum(self.efforts * (1. - speed / self.vel_limits), self.efforts)
            lower = np.maximum(self.efforts * (-1. - speed / self.vel_limits), -self.efforts)
            self.data.qfrc_applied[self.vids] = np.clip(effort, lower, upper)
        before = self.data.time
        mujoco.mj_step(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError('Nonfinite native MuJoCo state')
        if abs(self.data.time - before - self.dt) > 1e-7:
            raise FloatingPointError('MuJoCo reset its simulation clock')
        bad = [mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
               mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADCTRL]
        if any(self.data.warning[int(i)].number for i in bad):
            raise FloatingPointError('MuJoCo reported an invalid position/velocity/acceleration/control')

    def error(self, frame):
        frame = min(frame, len(self.ref['joint_pos']) - 1)
        ref_torso = self.ref['body_pos_w'][frame, self.ref_torso]
        rr = Rotation.from_quat(self.ref['body_quat_w'][frame, self.ref_torso][[1, 2, 3, 0]])
        ar = Rotation.from_quat(self.data.xquat[self.torso][[1, 2, 3, 0]])
        relative = (ar * rr.inv()).as_quat()
        yaw = 2 * np.arctan2(relative[2], relative[3])
        base = self.data.xpos[self.torso].copy()
        base[2] = ref_torso[2]
        expected = Rotation.from_euler('z', yaw).apply(
            self.ref['body_pos_w'][frame, self.ref_tracked] - ref_torso) + base
        return float(np.linalg.norm(self.data.xpos[self.tracked] - expected, axis=-1).mean())

    def run(self, *, mu=1., seed=2031, jitter=False, capture=False):
        started = time.monotonic()
        initial = self.reset(mu, seed, jitter)
        m, d = self.model, self.data
        clip_steps = self.runner.num_steps
        total_steps = clip_steps + 100
        errors, heights, tilts, actions, targets = [], [], [], [], []
        poses, quats, qs, physics_q, physics_targets = [], [], [], [], []
        max_excess = np.zeros(29)
        max_speed = np.zeros(29)
        friction_samples = []
        slips = []
        bad_steps, fall_time, failure = 0, None, None
        with torch.inference_mode():
            for step in range(total_steps + 1):
                # mj_step leaves some derived fields at the preceding state.
                mujoco.mj_forward(m, d)
                height = float(d.xpos[self.torso, 2])
                tilt = float(np.degrees(np.arccos(np.clip(d.xmat[self.torso].reshape(3, 3)[2, 2], -1., 1.))))
                heights.append(height)
                tilts.append(tilt)
                errors.append(self.error(step))
                if step:
                    bad_steps = bad_steps + 1 if height < .35 or tilt > 75. else 0
                    if bad_steps >= 10 and fall_time is None:
                        fall_time = step * .02
                if capture:
                    poses.append(d.xpos[self.body_ids].copy())
                    quats.append(d.xquat[self.body_ids].copy())
                    qs.append(d.qpos[self.qids].copy())
                for contact in d.contact:
                    if self.floor in contact.geom:
                        friction_samples.append(float(contact.friction[0]))
                foot_velocities = []
                for foot in self.feet:
                    force = 0.
                    for ci, contact in enumerate(d.contact):
                        if self.floor not in contact.geom:
                            continue
                        other = contact.geom[1] if contact.geom[0] == self.floor else contact.geom[0]
                        if m.geom_bodyid[other] == foot:
                            mujoco.mj_contactForce(m, d, ci, self.contact_force)
                            force += max(0., self.contact_force[0])
                    if force > 10.:
                        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, foot, self.velocity, 0)
                        foot_velocities.append(float(np.linalg.norm(self.velocity[3:5])))
                if foot_velocities:
                    slips.append(float(np.mean(foot_velocities)))
                if step == total_steps:
                    break
                q_motor = self.runner.to_motors(d.qpos[self.qids] - self.bias)
                dq_motor = self.runner.to_motors(d.qvel[self.vids])
                mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, self.pelvis, self.velocity, 1)
                attempted_time = step * .02
                try:
                    target_motor, action = self.runner.step(q_motor, dq_motor, d.xquat[self.pelvis],
                                                          self.velocity[:3].copy(), elapsed_s=step * .02)
                    target = target_motor[self.runner.policy_to_robot]
                    if capture:
                        actions.append(action.copy())
                        targets.append(target.copy())
                    for substep in range(self.decimation):
                        attempted_time = step * .02 + (substep + 1) * self.dt
                        self.advance(target)
                        q, v = d.qpos[self.qids], d.qvel[self.vids]
                        max_excess = np.maximum(max_excess, np.maximum(self.limits[:, 0] - q, q - self.limits[:, 1]))
                        max_speed = np.maximum(max_speed, np.abs(v))
                        if capture:
                            physics_q.append(q.copy())
                            physics_targets.append((target + self.bias).copy())
                except (FloatingPointError, ValueError) as exc:
                    failure = {'time_s': attempted_time, 'raw_mujoco_clock_s': float(d.time), 'error': str(exc)}
                    break
        assert friction_samples, 'No physical floor contacts recorded'
        np.testing.assert_allclose(friction_samples, mu, atol=1e-7, rtol=1e-7)
        record = {
            'directory': self.directory, 'physics': 'Standalone native MuJoCo mj_step; no Newton/Lab stepping',
            'mujoco_version': mujoco.__version__, 'mode': self.mode, 'physics_dt': self.dt, 'policy_dt': .02,
            'model_sha256': self.info['model_sha256'], 'policy_sha256': self.contract['policy_sha256'],
            'checkpoint_sha256': self.contract['checkpoint_sha256'], 'motion_sha256': self.contract['motion_sha256'],
            'floor_friction': mu, 'robot_friction': 1., 'floor_priority': 1,
            'observed_floor_contact_friction_min': min(friction_samples),
            'observed_floor_contact_friction_max': max(friction_samples),
            'seed': seed, 'initial_jitter': jitter, 'runtime_dr': False, 'initial_state': initial,
            'clip_duration_s': clip_steps * .02, 'post_clip_hold_s': 2.,
            'simulated_s': failure['time_s'] if failure else float(d.time),
            'fall_time_s': fall_time, 'numerical_failure': failure,
            'clip_numerically_valid': failure is None or failure['time_s'] > clip_steps * .02 + 1e-8,
            'clip_success': (failure is None or failure['time_s'] > clip_steps * .02 + 1e-8)
                            and (fall_time is None or fall_time > clip_steps * .02 + 1e-8),
            'clip_and_hold_success': failure is None and fall_time is None,
            'warning_counts': {str(mujoco.mjtWarning(i)): int(d.warning[i].number)
                               for i in range(int(mujoco.mjtWarning.mjNWARNING))},
            'clip_mean_body_error_m': float(np.mean(errors[:clip_steps+1])),
            'clip_max_body_error_m': float(np.max(errors[:clip_steps+1])),
            'mean_contact_foot_origin_speed_m_s': float(np.mean(slips)) if slips else None,
            'max_actual_joint_limit_excess_deg': float(np.rad2deg(max_excess.max())),
            'worst_limit_joint': self.runner.joint_names[int(max_excess.argmax())],
            'joint_peak_excess_deg': np.rad2deg(max_excess).tolist(),
            'joint_peak_speed_rad_s': max_speed.tolist(),
            'height_min_m': min(heights), 'tilt_max_deg': max(tilts),
            'wall_seconds': time.monotonic() - started, 'hardware_validated': False,
        }
        if capture:
            trajectory = dict(time=np.arange(len(poses)) * .02, body_pos_w=poses, body_quat_w=quats,
                              joint_pos=qs, error_body_pos=errors, torso_height=heights, torso_tilt_deg=tilts,
                              actions=actions, encoder_targets=targets, physics_joint_pos=physics_q,
                              physical_targets_without_delay=physics_targets,
                              body_names=np.asarray(self.body_names), joint_names=np.asarray(self.runner.joint_names))
        else:
            trajectory = None
        # A standalone test must not import a hidden training physics stepper.
        assert not any(n == 'newton' or n.startswith('isaaclab') for n in sys.modules)
        return record, trajectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', required=True)
    parser.add_argument('--mu', type=float, default=1.)
    parser.add_argument('--seed', type=int, default=2031)
    parser.add_argument('--jitter', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    trial = NativeTrial(args.directory)
    result, poses = trial.run(mu=args.mu, seed=args.seed, jitter=args.jitter, capture=True)
    np.savez_compressed(args.output / 'rollout.npz', **poses)
    result['rollout_sha256'] = sha(args.output / 'rollout.npz')
    (args.output / 'rollout.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('initial_state', 'joint_peak_excess_deg', 'joint_peak_speed_rad_s')}, indent=2))


if __name__ == '__main__':
    main()
