import copy
import contextlib
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from g1_deploy import core
from g1_deploy.leapp import G1LeappPolicyRunner


@unittest.skipUnless(importlib.util.find_spec("leapp"), "requires the leapp extra")
class LeappTests(unittest.TestCase):
    def setUp(self):
        import onnx
        import torch
        import yaml
        from onnx import TensorProto, helper, numpy_helper
        from safetensors.torch import save_file

        previous_backend = next(name for name, joints in core.JOINT_ORDER_BY_BACKEND.items()
                                if list(joints) == list(core.POLICY_JOINT_NAMES))
        previous_scale, previous_tilt = copy.deepcopy(core.ACTION_SCALE), core.IMU_TILT_WXYZ.copy()
        def restore():
            core.set_policy_backend(previous_backend)
            core.set_action_scale(previous_scale)
            core.IMU_TILT_WXYZ = previous_tilt
        self.addCleanup(restore)
        core.set_policy_backend("g1_29dof")
        core.set_imu_tilt(0, 0)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        names = list(core.POLICY_JOINT_NAMES)
        n = len(names)
        self.contract = dict(joint_names=names, default_joint_pos=core.DEFAULT_JOINT_POS.tolist(),
                             depth_shape=[3, 2, 3], depth_max_range_m=3.0, action_scale=0.5)
        self.contract["default_joint_pos"][22] = 0.25
        (self.directory / "contract.json").write_text(json.dumps(self.contract))

        def port(name, shape, connection=None, elements=None):
            result = dict(name=name, shape=shape, dtype="float32", type="tensor")
            if connection:
                result["isaaclab_connection"] = connection
            if elements:
                result["element_names"] = [elements]
            return result

        inputs = [
            port("q", [1, n], "state:robot:joint_pos", names),
            port("dq", [1, n], "state:robot:joint_vel", names),
            port("gyro", [1, 3], "state:robot:root_ang_vel_b"),
            port("quat", [1, 4], "state:robot:root_quat_w", ["qx", "qy", "qz", "qw"]),
            port("cmd", [1, 3], "command:base_velocity"),
            port("depth", [1, 2, 3, 1], "state:depth_camera:output:distance_to_image_plane"),
            port("carry", [1, n]),
        ]
        outputs = [
            port("target", [1, n], "write:robot:set_joint_position_target_index", names),
            port("kp", [1, n], "write:robot:write_joint_stiffness_to_sim_index", names),
            port("kd", [1, n], "write:robot:write_joint_damping_to_sim_index", names),
        ]
        self.gains = np.arange(1, n + 1, dtype=np.float32)[None]
        nodes = [helper.make_node("Add", ["q", "carry"], ["target"]),
                 helper.make_node("Identity", ["kp_value"], ["kp"]),
                 helper.make_node("Identity", ["kd_value"], ["kd"])]
        graph = helper.make_graph(
            nodes, "stateful_target",
            [helper.make_tensor_value_info(p["name"], TensorProto.FLOAT, p["shape"]) for p in inputs],
            [helper.make_tensor_value_info(p["name"], TensorProto.FLOAT, p["shape"]) for p in outputs],
            [numpy_helper.from_array(self.gains, "kp_value"),
             numpy_helper.from_array(self.gains / 10, "kd_value")],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
        onnx.save(model, self.directory / "model.onnx")
        data = (self.directory / "model.onnx").read_bytes()
        save_file({"node/carry": torch.zeros(1, n)}, str(self.directory / "initial.safetensors"))
        self.description = dict(
            models={"node": dict(inputs=inputs, outputs=outputs, parameters=dict(
                backend="onnx", model_path="model.onnx", md5sum=hashlib.md5(data).hexdigest(),
                sha256sum=hashlib.sha256(data).hexdigest()))},
            pipeline=dict(inputs={"node": [p["name"] for p in inputs[:-1]]},
                          outputs={"node": [p["name"] for p in outputs]}, data_flow={},
                          feedback_flow={"node/target": ["node/carry"]},
                          initial_values="initial.safetensors"),
        )
        self.description["system information"] = {"leapp version": "0.6.0"}
        self.yaml = self.directory / "model.yaml"
        self.yaml.write_text(yaml.safe_dump(self.description))

    def inputs(self):
        return (np.linspace(0, 0.28, 29, dtype=np.float32), np.zeros(29, np.float32),
                np.array([0.5, 0.5, 0.5, 0.5], np.float32), np.array([1, 2, 3], np.float32),
                np.array([0.2, 0, 0.1], np.float32),
                np.array([[0, np.nan, np.inf], [1, 2, 3]], np.float32))

    def test_official_feedback_reset_and_targets_not_scaled_twice(self):
        runner = G1LeappPolicyRunner(self.yaml, self.contract)
        first, _ = runner.step(*self.inputs())
        second, _ = runner.step(*self.inputs())
        np.testing.assert_allclose(first, self.inputs()[0], atol=1e-7)
        np.testing.assert_allclose(second, 2 * first, atol=1e-7)
        runner.reset()
        np.testing.assert_array_equal(runner.step(*self.inputs())[0], first)
        mapped = core.POLICY_TO_ROBOT >= 0
        np.testing.assert_array_equal(runner.kp[core.POLICY_TO_ROBOT[mapped]], self.gains[0, mapped])

    def test_sensor_units_fingers_and_orientation_at_graph_boundary(self):
        runner = G1LeappPolicyRunner(self.yaml, self.contract)
        runner.step(*self.inputs())
        values = runner.inference.value_dict["node"]
        np.testing.assert_array_equal(values["depth"].numpy().reshape(2, 3), [[3, 2, 1], [3, 3, 3]])
        self.assertEqual(float(values["q"][0, 22]), 0.25)
        inputs = list(self.inputs())
        inputs[2] = np.array([1, 0, 0, 0], np.float32)
        runner.step(*inputs)
        np.testing.assert_array_equal(values["quat"].numpy(), [[0, 0, 0, 1]])
        np.testing.assert_array_equal(values["gyro"].numpy(), [[1, 2, 3]])

    def test_rejects_bad_inputs_and_nonfinite_outputs(self):
        import torch

        runner = G1LeappPolicyRunner(self.yaml, self.contract)
        with self.assertRaisesRegex(ValueError, "action_limit"):
            runner.step(*self.inputs(), action_limit=8)
        inputs = list(self.inputs())
        inputs[3] = np.array([np.nan, 0, 0], np.float32)
        with self.assertRaisesRegex(ValueError, "Invalid sensor input"):
            runner.step(*inputs)
        with patch.object(runner.inference, "run_policy", return_value={
            "node/target": torch.full((1, 43), float("nan")),
        }), self.assertRaisesRegex(ValueError, "Invalid LEAPP output"):
            runner.step(*self.inputs())

    def test_joint_order_mismatch_is_rejected(self):
        import yaml

        self.description["models"]["node"]["outputs"][0]["element_names"][0] = list(reversed(core.POLICY_JOINT_NAMES))
        self.yaml.write_text(yaml.safe_dump(self.description))
        with self.assertRaisesRegex(ValueError, "joint shape/order"):
            G1LeappPolicyRunner(self.yaml, self.contract)

    def test_recording_saves_targets_and_bundle_identity(self):
        from types import SimpleNamespace
        from g1_deploy.analysis import records
        from g1_deploy.recording import create_recorder

        args = SimpleNamespace(leapp=str(self.yaml), policy=None, depth=str(self.directory),
                               log_dir=str(self.directory / "logs"), duration=0.1)
        recorder = create_recorder(args, self.contract, core)
        try:
            runner = G1LeappPolicyRunner(self.yaml, self.contract)
            runner.recorder = recorder
            runner.step(*self.inputs())
            runner.reset()
        finally:
            recorder.close()
        _, saved = next(records(recorder.directory, "leapp_policy", include_depth=True))
        self.assertNotIn("action_raw", saved)
        self.assertEqual(saved["joint_target"].shape, (43,))
        self.assertGreater(float(saved["node/depth"].max()), 0)
        metadata = json.loads((recorder.directory / "metadata.json").read_text())
        self.assertEqual(len(metadata["leapp"]["artifact_sha256"]), 3)

    def test_control_loop_selects_leapp_without_loading_torchscript(self):
        from types import SimpleNamespace
        from g1_deploy import bootstrap

        # Keep this in-process test from re-execing to configure the DDS loader.
        with patch.object(bootstrap, "ensure_cyclonedds"):
            from scripts import run_policy_loop

        state = SimpleNamespace(mode_machine=5, tick=0,
                                imu_state=SimpleNamespace(quaternion=[1, 0, 0, 0], gyroscope=[0, 0, 0]),
                                motor_state=[SimpleNamespace(q=float(q), dq=0.0) for q in self.inputs()[0]])
        link = unittest.mock.Mock()
        link.wait_for_state.return_value = state
        link.snapshot = (state, 0)
        argv = ["run_policy_loop.py", "--sim", "none", "--depth", str(self.directory),
                "--leapp", str(self.yaml), "--policy_physics", "g1_29dof", "--blind", "offline",
                "--duration", "0.04", "--no_log"]
        with (patch.object(sys, "argv", argv),
              patch.object(run_policy_loop, "ChannelFactoryInitialize"),
              patch.object(run_policy_loop, "G1ControlLink", return_value=link),
              patch.object(run_policy_loop.PeriodicDeadline, "wait"),
              patch("g1_deploy.depth.load_depth_policy", side_effect=AssertionError("Must not load policy.pt")),
              contextlib.redirect_stdout(io.StringIO())):
            self.assertEqual(run_policy_loop.main(), 0)
        self.assertEqual(link.send.call_count, 2)
        first, second = link.send.call_args_list
        np.testing.assert_allclose(first.args[0], self.inputs()[0], atol=1e-7)
        np.testing.assert_allclose(second.args[0], 2 * self.inputs()[0], atol=1e-7)


if __name__ == "__main__":
    unittest.main()
