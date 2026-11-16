# Copyright (c) 2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Export a single-frame RSL-RL standing actor with its saved training contract."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np
import torch

from g1_deploy.hardware import MOTOR_NAMES


def resolve(value, names, default=None):
    if not isinstance(value, dict):
        return [value] * len(names)
    result = []
    for name in names:
        matches = [v for pattern, v in value.items() if re.fullmatch(pattern, name)]
        if len(matches) != 1 and not (len(matches) == 0 and default is not None):
            raise ValueError(f"Ambiguous or missing configuration for {name}")
        result.append(matches[0] if matches else default)
    return result


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--joint-metadata', type=Path, required=True,
                        help='Recorded training/evaluation articulation joint_names; not motion-file order.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import yaml

    class Loader(yaml.SafeLoader):
        pass
    Loader.add_constructor('tag:yaml.org,2002:python/tuple', lambda loader, node: loader.construct_sequence(node))
    Loader.add_constructor('tag:yaml.org,2002:python/object/apply:builtins.slice',
                           lambda loader, node: slice(*loader.construct_sequence(node)))
    source = args.checkpoint.parent
    env = yaml.load((source / 'params/env.yaml').read_text(), Loader=Loader)
    agent = yaml.load((source / 'params/agent.yaml').read_text(), Loader=Loader)
    names = json.loads(args.joint_metadata.read_text())['joint_names']
    motors = [name + '_joint' for name in MOTOR_NAMES]
    if len(names) != 29 or set(names) != set(motors):
        raise ValueError('Expected 29 body joints from the evaluated articulation')
    terms = dict(command=58, motion_anchor_ori_b=6, base_ang_vel=3, joint_pos=29, joint_vel=29, actions=29)
    obs = env['observations']['policy']
    if (not obs['concatenate_terms'] or obs['concatenate_dim'] != -1 or obs['history_length']
            or [key for key, value in obs.items() if isinstance(value, dict)] != list(terms)):
        raise ValueError('Unsupported standing observation group')
    if any(
        obs[key].get('scale') is not None or obs[key].get('clip') is not None
        or obs[key].get('modifiers') or obs[key].get('history_length', 0)
        for key in terms
    ):
        raise ValueError('Unsupported standing observation layout')
    if agent['actor']['obs_normalization'] or agent['clip_actions'] is not None or agent['actor']['activation'] != 'elu':
        raise ValueError('Expected unnormalized ELU actor with unclipped actions')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    widths = [154, *agent['actor']['hidden_dims'], 29]
    layers = []
    for i, (a, b) in enumerate(zip(widths, widths[1:])):
        layers.append(torch.nn.Linear(a, b))
        if i < len(widths) - 2:
            layers.append(torch.nn.ELU())
    actor = torch.nn.Sequential(*layers).eval()
    actor.load_state_dict({k.removeprefix('mlp.'): v for k, v in checkpoint['actor_state_dict'].items() if k.startswith('mlp.')})
    script = torch.jit.script(actor)
    torch.manual_seed(0)
    with torch.inference_mode():
        probes = torch.randn(128, 154)
        torch.testing.assert_close(script(probes), actor(probes), rtol=0, atol=0)
    robot = env['scene']['robot']
    vectors = {key: {} for key in ('stiffness', 'damping', 'armature', 'effort_limit_sim', 'velocity_limit_sim')}
    for actuator in robot['actuators'].values():
        selected = [n for n in names if any(re.fullmatch(p, n) for p in actuator['joint_names_expr'])]
        for key in vectors:
            for name, value in zip(selected, resolve(actuator[key], selected)):
                if name in vectors[key]:
                    raise ValueError(f'Multiple actuators for {name}')
                vectors[key][name] = value
    if env['commands']['motion']['fixed_frame_index'] != 0:
        raise ValueError('Expected a standing policy trained on fixed frame zero')
    action = env['actions']['joint_pos']
    if not action['use_default_offset'] or action['clip'] is not None:
        raise ValueError('Unsupported standing action processing')
    c = dict(schema_version=1, policy_type='motion_tracking', task='Motion-Standing-G1-v0',
             joint_names=names, action_dim=29, obs_1d_dim=154, history_length=1, obs_normalization=False,
             obs_1d_terms=terms, control_dt=env['sim']['dt'] * env['decimation'],
             default_joint_pos=resolve(robot['init_state']['joint_pos'], names, 0.),
             default_joint_vel=resolve(robot['init_state']['joint_vel'], names, 0.),
             action_scale=resolve(action['scale'], names), use_default_offset=True, clip_actions=None,
             anchor_body_name='torso_link', root_body_name='pelvis', quaternion_order='wxyz',
             orientation_encoding='matrix_first_two_columns_row_major', motion_file='motion.npz',
             motion_metadata_file='motion.json', fixed_frame_index=0, hardware_validated=False,
             checkpoint_sha256=digest(args.checkpoint), joint_metadata_sha256=digest(args.joint_metadata))
    for key, target in [('stiffness','stiffness'), ('damping','damping'), ('armature','armature'),
                        ('effort_limit_sim','joint_effort_limits'), ('velocity_limit_sim','joint_vel_limits')]:
        c[target] = [vectors[key][n] for n in names]
    reference = dict(np.load(source / 'reference.npz'))
    for key in ('joint_pos', 'body_pos_w', 'body_quat_w'):
        np.testing.assert_array_equal(reference[key][0], reference[key][1])
    for key in ('joint_vel', 'body_lin_vel_w', 'body_ang_vel_w'):
        assert np.count_nonzero(reference[key]) == 0, key
    metadata = json.loads((source / 'reference.json').read_text())
    if metadata['sha256'] != digest(source / 'reference.npz'):
        raise ValueError('Reference metadata checksum mismatch')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'policy.pt').exists():
        raise FileExistsError('Export already exists; choose a new output directory')
    script.save(str(args.output / 'policy.pt'))
    shutil.copy2(source / 'reference.npz', args.output / 'motion.npz')
    shutil.copy2(source / 'reference.json', args.output / 'motion.json')
    c.update(policy_sha256=digest(args.output/'policy.pt'), motion_sha256=metadata['sha256'])
    (args.output/'contract.json').write_text(json.dumps(c, indent=2)+'\n')
    print(args.output)


if __name__ == '__main__':
    main()
