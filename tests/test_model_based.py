import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from g1_deploy import core
from g1_deploy.hardware import MOTOR_NAMES
from g1_deploy.model_based import G1ModelBasedPolicyRunner
from g1_deploy.tracking import load_tracking_runner


class ModelBasedTrackingTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.directory = Path(tmp.name)
        self.addCleanup(setattr, core, 'IMU_TILT_WXYZ', core.IMU_TILT_WXYZ.copy())
        core.set_imu_tilt(0, 0)
        # Use LayerNorm/GELU as in the real export, including repeated inference after probing.
        model = torch.nn.Sequential(torch.nn.Linear(1570, 32), torch.nn.LayerNorm(32),
                                    torch.nn.GELU(), torch.nn.Linear(32, 29)).eval()
        torch.jit.script(model).save(str(self.directory/'policy.pt'))
        names = [name+'_joint' for name in MOTOR_NAMES]
        self.positions = np.arange(60, dtype=np.float32)[:, None] + np.arange(29, dtype=np.float32)[None]/100
        np.savez(self.directory/'motion.npz', joint_pos=self.positions, joint_vel=-self.positions,
                 body_quat_w=np.tile([1., 0, 0, 0], (60, 1, 1)), fps=np.array([50]),
                 joint_names=names, body_names=['pelvis'])
        self.contract = dict(schema_version=1, policy_type='model_based_tracking', obs_1d_dim=1570,
                             sensor_dim=93, history_length=10, goal_frames=10, goal_stride=5, action_dim=29,
                             obs_normalization='embedded', root_body_name='pelvis', anchor_body_name='pelvis',
                             quaternion_order='wxyz', orientation_encoding='matrix_first_two_columns_row_major',
                             use_default_offset=True, clip_actions=None,
                             input_layout='history_oldest_first_then_goal_joints_then_orientation',
                             joint_names=names, control_dt=.02, default_joint_pos=[.1]*29, default_joint_vel=[0.]*29,
                             action_scale=[.5]*29, stiffness=[10.]*29, damping=[1.]*29, motion_file='motion.npz',
                             motion_sha256=hashlib.sha256((self.directory/'motion.npz').read_bytes()).hexdigest(),
                             policy_sha256=hashlib.sha256((self.directory/'policy.pt').read_bytes()).hexdigest())
        self.save_contract()

    def save_contract(self):
        (self.directory/'contract.json').write_text(json.dumps(self.contract))

    def sensors(self, value=0.):
        return np.full(29, value), np.full(29, value*2), np.array([1., 0, 0, 0]), np.array([.1, .2, .3])

    def test_history_goal_window_and_full_clip_length(self):
        r = load_tracking_runner(self.directory)
        self.assertIsInstance(r, G1ModelBasedPolicyRunner)
        self.assertEqual(r.num_steps, 60)
        for i in range(12):
            obs = r.observe(*self.sensors(i), frame_index=0)
        history = obs[:930].reshape(10, 93)
        np.testing.assert_allclose(history[:, 6:35], np.repeat((np.arange(2,12)-.1)[:,None],29,axis=1))
        np.testing.assert_allclose(history[-1,:6], [0,0,-1,.1,.2,.3])
        goal = obs[930:1510].reshape(10,58)
        np.testing.assert_allclose(goal[:,:29], self.positions[np.arange(10)*5])
        np.testing.assert_allclose(goal[:,29:], -self.positions[np.arange(10)*5])
        np.testing.assert_allclose(obs[1510:].reshape(10,6), np.tile([1,0,0,1,0,0],(10,1)))
        final = r.observe(*self.sensors(), frame_index=59)[930:1510].reshape(10,58)
        np.testing.assert_allclose(final[:,:29], np.repeat(self.positions[-1:,:],10,axis=0))
        self.assertEqual(r.index_at(100),59)
        with self.assertRaisesRegex(ValueError,'ended'):
            r.observe(*self.sensors(),frame_index=60)

    def test_warmup_retains_measured_history_and_applied_action_on_handover(self):
        r = load_tracking_runner(self.directory)
        for i in range(10):
            r.set_previous_target(np.full(29, .1+.5*i))
            r.warmup(*self.sensors(i))
        r.begin_motion()
        obs = r.observe(*self.sensors(10),frame_index=0)
        history = obs[:930].reshape(10,93)
        np.testing.assert_allclose(history[:,6], np.arange(1,11)-.1)
        np.testing.assert_allclose(history[:,-1], [1,2,3,4,5,6,7,8,9,9])
        r.reset()
        obs = r.observe(*self.sensors(20))
        np.testing.assert_allclose(obs[:930].reshape(10,93)[:,6], 19.9)
        np.testing.assert_array_equal(obs[:930].reshape(10,93)[:,-29:], 0)

    def test_joint_mapping_and_pelvis_heading_are_independent_of_waist(self):
        self.contract['joint_names'] = self.contract['joint_names'][::-1]
        self.save_contract()
        r = load_tracking_runner(self.directory)
        q, dq, _, gyro = self.sensors()
        q[:] = np.arange(29)/100
        yaw = np.array([2**-.5,0,0,2**-.5])
        obs = r.observe(q,dq,yaw,gyro)
        np.testing.assert_allclose(obs[:930].reshape(10,93)[-1,6:35],q[::-1]-.1,atol=1e-7)
        np.testing.assert_allclose(obs[1510:].reshape(10,6),np.tile([1,0,0,1,0,0],(10,1)),atol=1e-7)
        # A subsequent pelvis turn changes all future orientations; waist joints do not enter FK.
        obs = r.observe(q,dq,[1,0,0,0],gyro,frame_index=1)
        np.testing.assert_allclose(obs[1510:].reshape(10,6),np.tile([0,-1,1,0,0,0],(10,1)),atol=1e-7)

    def test_loaded_layernorm_actor_supports_repeated_inference(self):
        r = load_tracking_runner(self.directory)
        for _ in range(5):
            target, action = r.step(*self.sensors(),elapsed_s=0)
            self.assertTrue(np.isfinite(target).all())
            np.testing.assert_allclose(target,.1+.5*action)

    def test_rejects_incompatible_goal_contract(self):
        self.contract['goal_stride'] = 1
        self.save_contract()
        with self.assertRaisesRegex(ValueError,'Unsupported'):
            load_tracking_runner(self.directory)


if __name__ == '__main__':
    unittest.main()
