import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from g1_deploy.analysis import quaternion_rpy, records, tick_intervals, vibration_spectrum
from g1_deploy.recording import DeploymentRecorder


class RecordingTests(unittest.TestCase):
    def test_in_place_model_does_not_overwrite_recorded_inputs(self):
        with tempfile.TemporaryDirectory() as root:
            recorder = DeploymentRecorder(Path(root) / "run", {})
            def policy(observation):
                observation[:] = 0
                return observation
            recorder.wrap_policy(policy)(np.ones(3))
            recorder.close()
            _, saved = next(records(recorder.directory, "policy"))
            np.testing.assert_array_equal(saved["input_0"], np.ones(3))
            np.testing.assert_array_equal(saved["action_raw"], np.zeros(3))

    def test_committed_samples_survive_process_exit_without_cleanup(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "interrupted"
            code = """
import os, sys, time
from g1_deploy.recording import DeploymentRecorder
recorder = DeploymentRecorder(sys.argv[1], {})
recorder.event('before_crash')
deadline = time.monotonic() + 3
while not recorder.written and time.monotonic() < deadline:
    time.sleep(0.01)
os._exit(7)
"""
            result = subprocess.run([sys.executable, "-c", code, str(directory)], timeout=5)
            self.assertEqual(result.returncode, 7)
            self.assertEqual(next(records(directory, "event"))[1]["name"], "before_crash")
            self.assertFalse(json.loads((directory / "metadata.json").read_text())["clean_shutdown"])

    def test_policy_inputs_and_unclipped_output_are_snapshots(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "run"
            recorder = DeploymentRecorder(directory, {})
            observation = np.arange(10, dtype=np.float32)
            depth = np.ones((1, 3, 4, 5), dtype=np.float32)
            output = np.array([12.0, -13.0], dtype=np.float32)
            wrapped = recorder.wrap_policy(lambda obs, image: output)
            self.assertIs(wrapped(observation, depth), output)
            observation[:] = 0
            output[:] = 0
            recorder.close()
            stamp, saved = next(records(directory, "policy", include_depth=True))
            np.testing.assert_array_equal(saved["input_0"], np.arange(10))
            np.testing.assert_array_equal(saved["action_raw"], [12, -13])
            np.testing.assert_array_equal(saved["input_1"], depth)
            self.assertGreater(stamp, 0)
            self.assertTrue(json.loads((directory / "metadata.json").read_text())["clean_shutdown"])

    def test_slow_writer_drops_instead_of_waiting(self):
        entered, release = threading.Event(), threading.Event()

        class SlowMessage:
            def serialize(self):
                entered.set()
                release.wait(3)
                return b"state"

        with tempfile.TemporaryDirectory() as root:
            recorder = DeploymentRecorder(Path(root) / "run", {}, queue_size=1)
            try:
                recorder.message("lowstate", SlowMessage(), 1)
                self.assertTrue(entered.wait(1))
                recorder.event("queued")
                started = time.monotonic()
                recorder.event("dropped")
                self.assertLess(time.monotonic() - started, 0.1)
                self.assertEqual(recorder.dropped["event"], 1)
            finally:
                release.set()
                recorder.close()
            self.assertEqual(len(list(records(recorder.directory, "event"))), 1)

    def test_writer_failure_is_reported_and_does_not_escape_to_control(self):
        class BrokenMessage:
            def serialize(self):
                raise OSError("disk or serializer failed")

        with tempfile.TemporaryDirectory() as root:
            recorder = DeploymentRecorder(Path(root) / "run", {})
            recorder.message("lowstate", BrokenMessage(), 1)
            recorder._thread.join(2)
            recorder.event("after_failure")
            recorder.close()
            status = json.loads((recorder.directory / "metadata.json").read_text())
            self.assertFalse(status["clean_shutdown"])
            self.assertIn("failed", status["error"])
            self.assertEqual(status["dropped"]["event"], 1)

    def test_full_dds_message_roundtrip_including_unused_motor_slots(self):
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

        with tempfile.TemporaryDirectory() as root:
            message = unitree_hg_msg_dds__LowState_()
            message.tick = 2**32 - 2
            message.motor_state[34].sensor = [42, 17]
            message.motor_state[34].temperature = [31, 32]
            recorder = DeploymentRecorder(Path(root) / "run", {})
            recorder.message("lowstate", message, 456)
            recorder.close()
            stamp, payload = next(records(recorder.directory, "lowstate"))
            restored = LowState_.deserialize(payload)
            self.assertEqual(stamp, 456)
            self.assertEqual(restored.serialize(), message.serialize())
            self.assertEqual(restored.motor_state[34].sensor, [42, 17])
            self.assertEqual(restored.motor_state[34].temperature, [31, 32])

    def test_partial_recording_is_readable_before_shutdown(self):
        with tempfile.TemporaryDirectory() as root:
            recorder = DeploymentRecorder(Path(root) / "run", {})
            try:
                recorder.event("started")
                deadline = time.monotonic() + 3
                while not list(records(recorder.directory, "event")) and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertEqual(next(records(recorder.directory, "event"))[1]["name"], "started")
                status = json.loads((recorder.directory / "metadata.json").read_text())
                self.assertFalse(status["clean_shutdown"])
            finally:
                recorder.close()


class SignalTests(unittest.TestCase):
    def test_orientation_uses_quaternion_and_preserves_invalid_samples(self):
        half = np.sqrt(0.5)
        angles = quaternion_rpy([[half, 0, 0, half], [0, 0, 0, 0]])
        np.testing.assert_allclose(angles[0], [0, 0, np.pi / 2])
        self.assertTrue(np.isnan(angles[1]).all())

    def test_tick_wrap_and_reset_are_distinct(self):
        np.testing.assert_allclose(tick_intervals([2**32 - 2, 0, 2, 0]), [0.002, 0.002, -0.002])

    def test_known_oscillation_frequency_and_rms(self):
        tick = np.arange(2048) * 2
        velocity = np.sin(2 * np.pi * 25 * tick * 0.001)[:, None]
        result = vibration_spectrum(tick, velocity)
        self.assertAlmostEqual(result["peak_hz"][0], 25, delta=2)
        self.assertAlmostEqual(result["band_rms"][0], 1 / np.sqrt(2), delta=0.02)

    def test_spectrum_does_not_interpolate_across_missing_data(self):
        tick = np.arange(256) * 2
        tick[128:] += 100
        self.assertIsNone(vibration_spectrum(tick, np.ones((256, 2))))


if __name__ == "__main__":
    unittest.main()
