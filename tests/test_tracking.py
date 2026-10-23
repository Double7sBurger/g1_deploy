import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from g1_deploy import bootstrap, core
from g1_deploy.hardware import MOTOR_NAMES
from g1_deploy.tracking import G1TrackingPolicyRunner


class TrackingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.tilt = core.IMU_TILT_WXYZ.copy()
        self.addCleanup(setattr, core, "IMU_TILT_WXYZ", self.tilt)
        core.set_imu_tilt(0, 0)
        self.names = [name + "_joint" for name in MOTOR_NAMES]
        self.action = np.arange(29, dtype=np.float32) / 10
        model = torch.nn.Linear(154, 29)
        with torch.no_grad():
            model.weight.zero_()
            model.bias.copy_(torch.from_numpy(self.action))
        torch.jit.script(model).save(str(self.directory / "policy.pt"))
        self.positions = np.zeros((4, 29), np.float32)
        self.positions[1] = np.arange(29) / 100
        self.quats = np.array([[1, 0, 0, 0], [2**-.5, 0, 0, 2**-.5], [0, 1, 0, 0], [1, 0, 0, 0]])
        self.contract = dict(
            schema_version=1, policy_type="motion_tracking", obs_1d_dim=154, action_dim=29,
            history_length=1, obs_normalization=False, use_default_offset=True, clip_actions=None,
            obs_1d_terms=dict(command=58, motion_anchor_ori_b=6, base_ang_vel=3, joint_pos=29, joint_vel=29, actions=29),
            joint_names=self.names, default_joint_pos=(np.arange(29)/100).tolist(), default_joint_vel=[0]*29,
            action_scale=[.5]*29, stiffness=list(range(1, 30)), damping=[1]*29, control_dt=.02,
            root_body_name="pelvis", anchor_body_name="torso_link", quaternion_order="wxyz",
            orientation_encoding="matrix_first_two_columns_row_major", motion_file="motion.npz",
            motion_metadata_file="motion.json",
        )
        self.save_bundle()

    def save_bundle(self):
        np.savez(self.directory / "motion.npz", fps=50, joint_pos=self.positions,
                 joint_vel=np.zeros_like(self.positions), body_quat_w=self.quats[:, None],
                 joint_names=self.names, body_names=["torso_link"])
        for key, name in (("motion_sha256", "motion.npz"), ("policy_sha256", "policy.pt")):
            self.contract[key] = hashlib.sha256((self.directory/name).read_bytes()).hexdigest()
        (self.directory / "contract.json").write_text(json.dumps(self.contract))

    def sensors(self):
        return [np.zeros(29), np.zeros(29), np.array([1, 0, 0, 0]), np.zeros(3)]

    def test_observation_slots_and_previous_raw_action(self):
        runner = G1TrackingPolicyRunner(self.directory)
        sensors = self.sensors()
        sensors[0] = np.arange(29)/50
        sensors[1] = np.arange(29)/100
        sensors[3] = np.array([.1, .2, .3])
        # Keep waist neutral so the independent expected relative rotation is identity.
        sensors[0][12:15] = 0
        obs = runner.observe(*sensors)
        self.assertEqual(obs.shape, (154,))
        np.testing.assert_array_equal(obs[:58], np.zeros(58))
        np.testing.assert_allclose(obs[58:64], [1, 0, 0, 1, 0, 0], atol=1e-7)
        np.testing.assert_allclose(obs[64:67], sensors[3])
        np.testing.assert_allclose(obs[67:96], sensors[0]-np.arange(29)/100, atol=1e-7)
        np.testing.assert_allclose(obs[96:125], sensors[1])
        np.testing.assert_array_equal(obs[125:], np.zeros(29))
        target, action = runner.step(*sensors)
        np.testing.assert_allclose(target, np.arange(29)/100 + .5*self.action, atol=1e-7)
        np.testing.assert_array_equal(action, self.action)
        np.testing.assert_array_equal(runner.observe(*sensors)[125:], self.action)
        runner.reset()
        np.testing.assert_array_equal(runner.observe(*sensors)[125:], np.zeros(29))

    def test_heading_alignment_is_fixed_and_preserves_reference_turn(self):
        runner = G1TrackingPolicyRunner(self.directory)
        sensors = self.sensors()
        sensors[2] = np.array([2**-.5, 0, 0, 2**-.5])
        np.testing.assert_allclose(runner.observe(*sensors)[58:64], [1, 0, 0, 1, 0, 0], atol=1e-7)
        obs = runner.observe(*sensors, frame_index=1)
        np.testing.assert_allclose(obs[58:64], [0, -1, 1, 0, 0, 0], atol=1e-7)

    def test_torso_uses_waist_and_imu_mounting_rotation(self):
        runner = G1TrackingPolicyRunner(self.directory)
        sensors = self.sensors()
        core.set_imu_tilt(0, 90)
        sensors[2] = np.array([2**-.5, 2**-.5, 0, 0])
        sensors[3] = np.array([0, 1, 0])
        obs = runner.observe(*sensors)
        np.testing.assert_allclose(obs[64:67], [0, 0, 1], atol=1e-7)
        sensors[0][13] = np.pi/2
        obs = runner.observe(*sensors)
        np.testing.assert_allclose(obs[58:64], [1, 0, 0, 0, 0, -1], atol=1e-7)
        self.assertAlmostEqual(runner.tilt_error, np.pi/2)

    def test_intended_inversion_is_not_a_fall_but_reference_deviation_is(self):
        runner = G1TrackingPolicyRunner(self.directory)
        sensors = self.sensors()
        runner.observe(*sensors)
        sensors[2] = np.array([0, 1, 0, 0])
        runner.observe(*sensors, frame_index=2)
        self.assertAlmostEqual(runner.tilt_error, 0)
        sensors[2] = np.array([1, 0, 0, 0])
        runner.observe(*sensors, frame_index=2)
        self.assertAlmostEqual(runner.tilt_error, np.pi)

    def test_named_joint_mapping_is_not_an_assumed_motor_order(self):
        # The motion stays in motor order; only the policy's contract is reversed.
        self.contract['joint_names'] = list(reversed(self.names))
        self.save_bundle()
        runner = G1TrackingPolicyRunner(self.directory)
        q = np.arange(29)/10
        obs = runner.observe(q, np.zeros(29), [1, 0, 0, 0], np.zeros(3), frame_index=1)
        np.testing.assert_array_equal(obs[:29], self.positions[1, ::-1])
        np.testing.assert_allclose(obs[67:96], q[::-1]-np.arange(29)/100, atol=1e-6)
        np.testing.assert_array_equal(runner.kp, np.arange(29, 0, -1))
        target, _ = runner.step(*self.sensors())
        np.testing.assert_allclose(target, (np.arange(29)/100 + .5*self.action)[::-1], atol=1e-7)

    def test_rejects_mismatched_bundle_and_invalid_sensors(self):
        runner = G1TrackingPolicyRunner(self.directory)
        sensors = self.sensors()
        sensors[0][4] = np.nan
        with self.assertRaisesRegex(ValueError, 'sensor'):
            runner.step(*sensors)
        with self.assertRaisesRegex(ValueError, 'quaternion'):
            runner.step(np.zeros(29), np.zeros(29), np.zeros(4), np.zeros(3))
        (self.directory / 'motion.npz').write_bytes(b'bad data')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            G1TrackingPolicyRunner(self.directory)

    def test_clip_ends_without_wrapping_or_resetting(self):
        runner = G1TrackingPolicyRunner(self.directory)
        self.assertAlmostEqual(runner.duration, .06)
        for t in [0, .02, .04]:
            runner.step(*self.sensors(), elapsed_s=t)
        with self.assertRaisesRegex(ValueError, 'ended'):
            runner.step(*self.sensors(), elapsed_s=.06)

    def loop(self, mode=5, extra=(), quaternion=(1, 0, 0, 0)):
        with patch.object(bootstrap, 'ensure_cyclonedds'):
            from scripts import run_policy_loop
        state = SimpleNamespace(mode_machine=mode, tick=0,
                                imu_state=SimpleNamespace(quaternion=quaternion, gyroscope=[0, 0, 0]),
                                motor_state=[SimpleNamespace(q=0., dq=0.) for _ in range(29)])
        link = Mock()
        link.wait_for_state.return_value = state
        link.snapshot = (state, 0)
        argv = ['loop', '--sim', 'none', '--tracking', str(self.directory), '--duration', 'inf', '--no_log', *extra]
        with (patch.object(sys, 'argv', argv), patch.object(run_policy_loop, 'ChannelFactoryInitialize'),
              patch.object(run_policy_loop, 'G1ControlLink', return_value=link),
              patch.object(run_policy_loop.PeriodicDeadline, 'wait'),
              contextlib.redirect_stdout(io.StringIO())):
            result = run_policy_loop.main()
        return result, link

    def test_loop_uses_tracking_gains_all_29_outputs_and_stops_at_clip_end(self):
        result, link = self.loop()
        self.assertEqual(result, 0)
        self.assertEqual(link.send.call_count, 3)
        for call in link.send.call_args_list:
            np.testing.assert_allclose(call.args[0], np.arange(29)/100 + .5*self.action, atol=1e-7)
            np.testing.assert_array_equal(call.args[1], np.arange(1, 30))
            np.testing.assert_array_equal(call.args[2], np.ones(29))

    def test_locked_waist_variant_cannot_engage_tracking(self):
        with self.assertRaisesRegex(ValueError, '29 DoF'):
            self.loop(mode=6)

    def test_tracking_guard_stops_before_sending_another_motor_command(self):
        result, link = self.loop(quaternion=(2**-.5, 2**-.5, 0, 0))
        self.assertEqual(result, 1)
        link.send.assert_not_called()

    def test_motion_recorder_keeps_its_contract_and_reference_identity(self):
        from g1_deploy.analysis import analyze
        from g1_deploy.recording import create_recorder
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
        runner = G1TrackingPolicyRunner(self.directory)
        args = SimpleNamespace(tracking=str(self.directory), policy=None, depth=None,
                               log_dir=str(self.directory/'logs'), duration=runner.duration)
        recorder = create_recorder(args, runner.contract, core, tracking=runner)
        start = recorder.metadata['monotonic_ns']
        for i in range(2):
            state = unitree_hg_msg_dds__LowState_()
            state.tick = i*20
            state.imu_state.quaternion = [1., 0., 0., 0.]
            stamp = start + i*20_000_000
            recorder.message('lowstate', state, stamp)
            recorder.arrays('step', stamp, step=i, command=np.zeros(58),
                            state_received_ns=stamp, captured_ns=stamp)
            recorder.arrays('tracking', stamp, reference_time=i*.02, tilt_error=i*.1)
        recorder.close()
        metadata = json.loads((recorder.directory/'metadata.json').read_text())
        self.assertEqual(metadata['policy_joint_names'], self.names)
        self.assertEqual(metadata['policy_to_robot'], list(range(29)))
        self.assertEqual(metadata['tracking']['motion_sha256'], runner.contract['motion_sha256'])
        _, summary, arrays = analyze(recorder.directory)
        self.assertNotIn('heading_error_deg', summary)
        self.assertNotIn('command', arrays)
        self.assertEqual(arrays['reference_command'].shape, (2, 58))
        np.testing.assert_allclose(arrays['tracking_tilt_error'], [0., .1])


if __name__ == '__main__':
    unittest.main()
