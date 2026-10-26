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
from g1_deploy.tracking import G1TrackingPolicyRunner, StandingTrackingRunner


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

    def test_elapsed_clock_holds_last_frame_without_wrapping_or_resetting(self):
        runner = G1TrackingPolicyRunner(self.directory)
        self.assertAlmostEqual(runner.duration, .06)
        for t in [0, .02, .04]:
            runner.step(*self.sensors(), elapsed_s=t)
        for t in [.06, 60.0]:
            runner.step(*self.sensors(), elapsed_s=t)
            self.assertEqual(runner.frame_index, runner.num_steps - 1)
            np.testing.assert_array_equal(runner.command_at(t)[:29], self.positions[2])
        with self.assertRaisesRegex(ValueError, 'ended'):
            runner.step(*self.sensors())

    def test_standing_wait_does_not_consume_motion_and_handover_feeds_back_blended_target(self):
        standing = G1TrackingPolicyRunner(self.directory)
        motion = G1TrackingPolicyRunner(self.directory)
        motion.policy = Mock(return_value=np.full((1, 29), -.5, np.float32))
        sequence = StandingTrackingRunner(standing, motion, transition_s=1)
        sensors = self.sensors()
        initial, _ = sequence.step(*sensors, now=0)
        np.testing.assert_array_equal(initial, sensors[0])
        held, _ = sequence.step(*sensors, now=30)
        self.assertEqual(standing.frame_index, 0)
        self.assertEqual(motion._next_frame, 0)
        motion.policy.assert_not_called()
        sequence.start_motion(30)
        target, _ = sequence.step(*sensors, now=30)
        np.testing.assert_allclose(target, held, atol=1e-7)
        # First motion inference sees the target actually sent by the standing controller.
        np.testing.assert_allclose(motion.policy.call_args.args[0][0, 125:],
                                   (held - motion.default_joint_pos) / motion.action_scale)
        target, _ = sequence.step(*sensors, now=30.5)
        expected = .5*held + .5*(motion.default_joint_pos - .5*motion.action_scale)
        np.testing.assert_allclose(target, expected, atol=1e-7)
        np.testing.assert_allclose(motion._last_action, (target-motion.default_joint_pos)/motion.action_scale)
        self.assertEqual(motion.frame_index, 0)
        applied = np.clip(target, -.05, .05)
        sequence.set_previous_target(applied)
        np.testing.assert_allclose(motion._last_action, (applied-motion.default_joint_pos)/motion.action_scale)
        np.testing.assert_allclose(standing._last_action, (applied-standing.default_joint_pos)/standing.action_scale)
        sequence.step(*sensors, now=np.nextafter(31., 0.))
        self.assertEqual(sequence.phase, 'motion')
        self.assertEqual(motion.frame_index, 0)
        sequence.step(*sensors, now=31.02)
        self.assertEqual(motion.frame_index, 1)
        # A repeated confirmation must never restart the motion clock.
        sequence.start_motion(31.02)
        self.assertEqual(sequence.motion_start, 31)

    def test_standing_rejects_gain_change_and_confirmation_before_engagement(self):
        standing = G1TrackingPolicyRunner(self.directory)
        motion = G1TrackingPolicyRunner(self.directory)
        sequence = StandingTrackingRunner(standing, motion)
        with self.assertRaisesRegex(RuntimeError, 'engaged'):
            sequence.start_motion(0)
        motion.kp *= 2
        with self.assertRaisesRegex(ValueError, 'Kp/Kd'):
            StandingTrackingRunner(standing, motion)

    def test_confirmation_reader_waits_for_complete_exact_line_and_detects_eof(self):
        import os
        from g1_deploy.hardware import ConfirmationReader, OperatorAbort

        read_fd, write_fd = os.pipe()
        with os.fdopen(read_fd, 'rb', buffering=0) as stream:
            reader = ConfirmationReader(stream=stream)
            self.assertFalse(reader.poll())
            os.write(write_fd, b'wrong\ngo')
            self.assertFalse(reader.poll())
            os.write(write_fd, b'2\n')
            self.assertTrue(reader.poll())
            self.assertFalse(reader.poll())
            os.close(write_fd)
            self.assertFalse(reader.poll())
        read_fd, write_fd = os.pipe()
        with os.fdopen(read_fd, 'rb', buffering=0) as stream:
            reader = ConfirmationReader(stream=stream)
            os.close(write_fd)
            with self.assertRaisesRegex(OperatorAbort, 'closed'):
                reader.poll()

    def standing_loop(self, *, abort=False, timeout=300, extra=(), stop_after=None, recorder=None):
        from g1_deploy import hardware
        with patch.object(bootstrap, 'ensure_cyclonedds'):
            from scripts import run_policy_loop
        self.quats[:] = [1, 0, 0, 0]
        self.save_bundle()
        clock = [0.0]
        state = SimpleNamespace(mode_machine=5, tick=0,
                                imu_state=SimpleNamespace(quaternion=[1, 0, 0, 0], gyroscope=[0, 0, 0]),
                                motor_state=[SimpleNamespace(q=0., dq=0.) for _ in range(29)])
        link = Mock()
        link.wait_for_state.return_value = state
        link.snapshot = state, 0
        def sent(*_):
            if stop_after is not None and clock[0] >= stop_after:
                raise KeyboardInterrupt
            clock[0] += .02
            link.snapshot = state, int(clock[0]*1e9)
        link.send.side_effect = sent
        reader = Mock()
        reader.poll.side_effect = hardware.OperatorAbort('closed') if abort else [False, True]
        argv = ['loop', '--real', '--skip_release_mode', '--tracking', str(self.directory),
                '--stand_policy', str(self.directory), '--duration', 'inf', '--no_log',
                '--align_timeout', str(timeout), *extra]
        if recorder is not None:
            argv.remove('--no_log')
            recorder.wrap_policy.side_effect = lambda policy, **_: policy
        with (patch.object(sys, 'argv', argv), patch.object(run_policy_loop, 'ChannelFactoryInitialize'),
              patch.object(run_policy_loop, 'G1ControlLink', return_value=link),
              patch('g1_deploy.recording.create_recorder', return_value=recorder),
              patch('g1_deploy.analysis.save_report') as export,
              patch.object(run_policy_loop.time, 'monotonic', side_effect=lambda: clock[0]),
              patch.object(run_policy_loop.time, 'monotonic_ns', side_effect=lambda: int(clock[0]*1e9)),
              patch.object(run_policy_loop.PeriodicDeadline, 'wait'),
              patch.object(hardware, 'confirm') as confirm, patch.object(hardware, 'check_upright'),
              patch.object(hardware, 'check_abort'), patch.object(hardware, 'ConfirmationReader', return_value=reader),
              patch.object(hardware, 'ramp_to_pose') as ramp, patch.object(hardware, 'hold_until_confirmed') as hold,
              patch.object(hardware, 'damp_down') as damp, contextlib.redirect_stdout(io.StringIO())):
            try:
                result = run_policy_loop.main()
            finally:
                confirm.assert_called_once()
                ramp.assert_not_called()
                hold.assert_not_called()
                damp.assert_called_once()
            if recorder is not None:
                export.assert_called_once_with(recorder.directory, None, plots=False)
        return result, link, clock[0]

    def test_go_go2_loop_keeps_controlling_and_starts_full_clip_after_wait_and_blend(self):
        result, link, elapsed = self.standing_loop()
        self.assertEqual(result, 0)
        self.assertAlmostEqual(elapsed, 1.36)
        self.assertEqual(link.send.call_count, 68)
        np.testing.assert_array_equal(link.send.call_args_list[0].args[0], np.zeros(29))
        for call in link.send.call_args_list:
            np.testing.assert_array_equal(call.args[1], np.arange(1, 30))

    def test_standing_hold_at_end_honors_motion_duration_and_unbounded_abort(self):
        for extra, stop_after in [(['--hold_at_end', '--duration', '.2'], None),
                                  (['--hold_at_end'], 1.5)]:
            with self.subTest(extra=extra):
                result, link, elapsed = self.standing_loop(extra=extra, stop_after=stop_after)
                self.assertEqual(result, 0)
                self.assertAlmostEqual(elapsed, 1.5)
                self.assertGreater(link.send.call_count, 68)
                for call in link.send.call_args_list:
                    np.testing.assert_array_equal(call.args[1], np.arange(1, 30))
                    np.testing.assert_array_equal(call.args[2], np.ones(29))

    def test_standing_recording_starts_policy_window_at_first_motion_step(self):
        recorder = Mock()
        recorder.directory = self.directory / 'recorded'
        result, _, _ = self.standing_loop(recorder=recorder)
        self.assertEqual(result, 0)
        events = [c for c in recorder.event.call_args_list if c.args[0] == 'policy_start']
        self.assertEqual(len(events), 1)
        first = next(c for c in recorder.arrays.call_args_list if c.args[0] == 'tracking')
        self.assertEqual(first.kwargs['frame'], 0)
        self.assertEqual(events[0].kwargs['step'], first.kwargs['step'])
        control = next(c for c in recorder.arrays.call_args_list
                       if c.args[0] == 'step' and c.kwargs['step'] == first.kwargs['step'])
        self.assertEqual(events[0].kwargs['timestamp_ns'], control.args[1])
        self.assertEqual(recorder.wrap_policy.call_count, 2)
        self.assertEqual(recorder.wrap_policy.call_args_list[1].kwargs['kind'], 'standing_policy')

    def test_active_and_passive_standing_options_are_mutually_exclusive(self):
        with patch.object(bootstrap, 'ensure_cyclonedds'):
            from scripts import run_policy_loop
        argv = ['loop', '--tracking', str(self.directory), '--stand_policy', str(self.directory),
                '--stand_kp', '300', '--no_log']
        with (patch.object(sys, 'argv', argv), contextlib.redirect_stderr(io.StringIO()) as error,
              patch.object(run_policy_loop, 'ChannelFactoryInitialize') as initialize):
            with self.assertRaises(SystemExit) as exit:
                run_policy_loop.main()
            self.assertEqual(exit.exception.code, 2)
            self.assertIn('alternative standing modes', error.getvalue())
            initialize.assert_not_called()

    def test_standing_input_abort_and_timeout_damp_without_starting_motion(self):
        result, link, elapsed = self.standing_loop(abort=True)
        self.assertEqual(result, 0)
        self.assertEqual(link.send.call_count, 1)
        with self.assertRaisesRegex(TimeoutError, 'not received'):
            self.standing_loop(timeout=.01)

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

    def test_passive_stand_kp_returns_to_trained_gains_on_policy_takeover(self):
        from g1_deploy import hardware

        with (patch.object(hardware, 'confirm'), patch.object(hardware, 'check_upright'),
              patch.object(hardware, 'check_abort'), patch.object(hardware, 'damp_down'),
              patch.object(hardware, 'ramp_to_pose') as ramp,
              patch.object(hardware, 'hold_until_confirmed') as hold,
              patch.object(hardware, 'report_tracking')):
            result, link = self.loop(extra=['--real', '--skip_release_mode', '--align', '--stand_kp', '300'])
        self.assertEqual(result, 0)
        legs = [i for i, name in enumerate(MOTOR_NAMES)
                if any(group in name for group in ('hip', 'knee', 'ankle', 'waist'))]
        for call in (ramp.call_args, hold.call_args):
            np.testing.assert_array_equal(call.args[2][legs], np.full(len(legs), 300))
            np.testing.assert_allclose(call.args[3][legs], np.sqrt(300 / np.arange(1, 30)[legs]))
        self.assertEqual(link.send.call_count, 3)
        for call in link.send.call_args_list:
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
