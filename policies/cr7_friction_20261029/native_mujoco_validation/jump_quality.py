"""Measure actual foot-box clearance in the recorded nominal MuJoCo trials."""

import json
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
PACKAGES = Path('/home/henry/workspace/g1_deploy/policies/cr7_friction_20261029')
model = mujoco.MjModel.from_binary_path(str(ROOT / 'models/implicit200/model.mjb'))
info = json.loads((ROOT / 'models/implicit200/model.json').read_text())
data = mujoco.MjData(model)
qids = [j['qpos'] for j in info['joints']]
feet = [info['bodies'][s + '_ankle_roll_link'] for s in ('left', 'right')]
geoms = [np.flatnonzero(model.geom_bodyid == b).item() for b in feet]
assert all(model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX for g in geoms)
assert model.jnt_qposadr[info['free_joint']] == 0


def clearance(positions, quats, joints, pelvis):
    out = []
    for pos, quat, q in zip(positions, quats, joints):
        data.qpos[:3] = pos[pelvis]
        data.qpos[3:7] = quat[pelvis]
        data.qpos[qids] = q
        mujoco.mj_forward(model, data)
        heights = [data.geom_xpos[g, 2] - np.sum(np.abs(data.geom_xmat[g].reshape(3, 3)[2]) * model.geom_size[g])
                   for g in geoms]
        out.append(heights)
    return np.asarray(out)


selection = json.loads((PACKAGES / 'selection.json').read_text())['entries']
package = PACKAGES / selection[0]['directory']
ref = np.load(package / 'motion.npz')
names = json.loads((package / 'motion.json').read_text())
order = [names['joint_names'].index(n) for n in info['joint_order']]
ref_clear = clearance(ref['body_pos_w'], ref['body_quat_w'], ref['joint_pos'][:, order], names['body_names'].index('pelvis'))
both_ref = ref_clear.min(axis=1)
airborne = np.flatnonzero(both_ref > .02)
assert len(airborne)
window = (max(0, airborne[0] - 15), min(len(both_ref), airborne[-1] + 16))


def stats(values):
    both = values[window[0]:window[1]].min(axis=1)
    return {'peak_both_feet_clearance_m': float(both.max()),
            'both_feet_above_2cm_duration_s': float((both > .02).sum() * .02)}


rows = []
for entry in selection:
    p = ROOT / 'results' / entry['directory'] / 'nominal/0'
    poses = np.load(p / 'rollout.npz')
    c = clearance(poses['body_pos_w'], poses['body_quat_w'], poses['joint_pos'], poses['body_names'].tolist().index('pelvis'))
    evaluation = json.loads((p / 'evaluation.json').read_text())
    rows.append({'directory': entry['directory'], **stats(c),
                 'clip_mean_body_error_cm': evaluation['clip_mean_body_error_m'] * 100,
                 'clip_peak_body_error_cm': evaluation['clip_max_body_error_m'] * 100,
                 'clip_and_hold_success': evaluation['clip_and_hold_success']})
report = {'measurement': 'Lowest point of each physical foot collision box relative to flat z=0 floor; minimum of the two feet',
          'scope': 'Nominal exact-reference initial state, floor friction 1.0; not an additional success gate',
          'window_s': [window[0] * .02, (window[1] - 1) * .02], 'reference': stats(ref_clear), 'candidates': rows}
(ROOT / 'jump_quality.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
