# Copyright (c) 2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Export the G1 motion actor using the Isaac Lab environment that trained it."""

import argparse
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from tensordict import TensorDict

from isaaclab_rl.model_based.models import MultiEncoderAE
from isaaclab_tasks.contrib.motion_tracking.kinematics import G1_JOINT_NAMES

from g1_deploy.hardware import MOTOR_NAMES
from scripts.export_standing import digest, resolve


class MotionActor(torch.nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.obs_normalizer = actor.obs_normalizer
        self.goal_normalizer = actor.goal_normalizers['motion']
        self.encoder = actor.encoders['motion']
        self.decoder = actor.decoder

    def forward(self, obs):
        history = self.obs_normalizer(obs[:, :930].reshape(-1, 10, 93)).flatten(1)
        latent = torch.tanh(self.encoder(self.goal_normalizer(obs[:, 930:])))
        return self.decoder(torch.cat((latent, history), dim=-1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--joint-metadata', type=Path, required=True, help='Native evaluation rollouts.npz')
    parser.add_argument('--motions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new export directory')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    cfg, env, state = checkpoint['agent_cfg'], checkpoint['env_cfg'], checkpoint['actor']
    if cfg['actor_obs'] != 'proprio' or cfg['history_length'] != 10 or cfg['slot_mix']['motion'] != 1.0:
        raise ValueError('Only the G1 proprioception-history motion actor is supported')
    names = np.load(args.joint_metadata, allow_pickle=False)['joint_names'].tolist()
    if len(names) != 29 or set(names) != {name + '_joint' for name in MOTOR_NAMES}:
        raise ValueError('Expected native evaluation metadata for 29 G1 joints')
    specs = [SimpleNamespace(name=name, goal_dim=state[f'goal_normalizers.{name}._mean'].numel())
             for name in cfg['slot_mix']]
    actor = MultiEncoderAE(93, 29, specs, cfg['slot_mix'], cfg['latent_dim'], cfg['actor_hidden_dims'],
                           cfg['latent_bc_weight'], 10).eval()
    actor.load_state_dict(state)
    wrapper = MotionActor(actor).eval()
    script = torch.jit.script(wrapper)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    max_error = 0.0
    with torch.inference_mode():
        for batch in (1, 17, 128):
            probe = torch.randn(batch, 1570)
            expected = actor(TensorDict(dict(history=probe[:, :930].reshape(batch, 10, 93),
                                             goal_motion=probe[:, 930:]), batch_size=[batch]))
            actual = script(probe)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
            max_error = max(max_error, (actual-expected).abs().max().item())
    robot, action = env['scene']['robot'], env['actions']['joint_pos']
    if not action['use_default_offset'] or action['clip'] is not None:
        raise ValueError('Unsupported action processing')
    c = dict(schema_version=1, policy_type='model_based_tracking', obs_1d_dim=1570, sensor_dim=93,
             history_length=10, goal_frames=10, goal_stride=5, action_dim=29, obs_normalization='embedded',
             input_layout='history_oldest_first_then_goal_joints_then_orientation', joint_names=names,
             control_dt=env['sim']['dt']*env['decimation'], use_default_offset=True, clip_actions=None,
             default_joint_pos=resolve(robot['init_state']['joint_pos'], names, 0.),
             default_joint_vel=resolve(robot['init_state']['joint_vel'], names, 0.),
             action_scale=resolve(action['scale'], names), anchor_body_name='pelvis', root_body_name='pelvis',
             quaternion_order='wxyz', orientation_encoding='matrix_first_two_columns_row_major',
             motion_file='motion.npz', hardware_validated=False, iteration=checkpoint['iteration'],
             checkpoint_sha256=digest(args.checkpoint), joint_metadata_sha256=digest(args.joint_metadata))
    for key, target in [('stiffness','stiffness'), ('damping','damping'), ('armature','armature'),
                        ('effort_limit_sim','joint_effort_limits'), ('velocity_limit_sim','joint_vel_limits')]:
        values = {}
        for actuator in robot['actuators'].values():
            selected = [n for n in names if any(re.fullmatch(p, n) for p in actuator['joint_names_expr'])]
            if values.keys() & set(selected):
                raise ValueError('Overlapping actuator assignments')
            values.update(zip(selected, resolve(actuator[key], selected)))
        c[target] = [values[n] for n in names]
    args.output.mkdir(parents=True)
    script.save(str(args.output/'policy.pt'))
    c['policy_sha256'] = digest(args.output/'policy.pt')
    source_names = env['commands']['motion']['joint_names'] or G1_JOINT_NAMES
    for path in sorted(args.motions.rglob('*.npz')):
        destination = args.output / path.relative_to(args.motions).with_suffix('')
        destination.mkdir(parents=True)
        data = dict(np.load(path, allow_pickle=False))
        data.update(joint_names=np.asarray(source_names), body_names=np.asarray(['pelvis']),
                    body_quat_w=data['root_quat_w'][:, None])
        np.savez_compressed(destination/'motion.npz', **data)
        # Relative links keep the actor shared when the entire export directory is moved.
        (destination/'policy.pt').symlink_to(os.path.relpath(args.output/'policy.pt', destination))
        contract = dict(c, motion_sha256=digest(destination/'motion.npz'), source_motion_sha256=digest(path),
                        source_motion=str(path.relative_to(args.motions)))
        (destination/'contract.json').write_text(json.dumps(contract, indent=2)+'\n')
    (args.output/'export_validation.json').write_text(json.dumps(dict(
        max_actor_error=max_error, batches=[1,17,128], checkpoint_sha256=c['checkpoint_sha256']), indent=2)+'\n')
    print(args.output, 'actor max error:', max_error)


if __name__ == '__main__':
    main()
