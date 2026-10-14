"""G1 sensor/motor adapter for the official LEAPP InferenceManager.

The exported graph owns observation processing, histories and action scaling.
This adapter only converts the existing G1 transport data to/from its YAML ports.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from g1_deploy import core


class G1LeappPolicyRunner:
    """Run a LEAPP depth student using the same sensor inputs as G1DepthPolicyRunner.

    Args:
        path: Official exported YAML, with its ONNX and initial-state files beside it.
        contract: Matching depth export contract, used for initial pose and absent fingers.
        depth_rotation: Rotate transport images by 0 or 180 degrees before the graph.
            The current Newton export flips both axes internally, so the existing upright
            G1 camera stream needs 180 here to retain its established image orientation.
    """

    def __init__(self, path: str | Path, contract: dict, *, depth_rotation: int = 180):
        import torch
        from leapp import InferenceManager

        if depth_rotation not in (0, 180):
            raise ValueError("depth_rotation must be 0 or 180")
        if list(contract["joint_names"]) != list(core.POLICY_JOINT_NAMES):
            raise ValueError("LEAPP contract joint names disagree with the selected policy physics")
        self._torch = torch
        self._default = np.asarray(contract["default_joint_pos"], dtype=np.float32)
        self._map = core.POLICY_TO_ROBOT.copy()
        self._mapped = self._map >= 0
        self.depth_shape = tuple(int(v) for v in contract["depth_shape"][1:])
        self._far = float(contract["depth_max_range_m"])
        self._depth_rotation = depth_rotation
        self.recorder = None
        torch.set_num_threads(1)
        self.inference = InferenceManager(str(Path(path).resolve()))
        self._inputs = self._ports(self.inference.inputs, "inputs")
        self._outputs = self._ports(self.inference.outputs, "outputs")
        expected = {
            "state:robot:root_ang_vel_b": (1, 3),
            "state:robot:root_quat_w": (1, 4),
            "command:base_velocity": (1, 3),
            "state:robot:joint_pos": (1, len(self._map)),
            "state:robot:joint_vel": (1, len(self._map)),
            "state:depth_camera:output:distance_to_image_plane": (1, *self.depth_shape, 1),
        }
        if set(self._inputs) != set(expected):
            raise ValueError(f"Unsupported LEAPP inputs: {list(self._inputs)}")
        for connection, shape in expected.items():
            _, spec = self._inputs[connection]
            if tuple(spec["shape"]) != shape or spec["dtype"] != "float32":
                raise ValueError(f"Unexpected shape/dtype for {connection}: {spec}")
        quat_spec = self._inputs["state:robot:root_quat_w"][1]
        if quat_spec.get("element_names") != [["qx", "qy", "qz", "qw"]]:
            raise ValueError("LEAPP orientation must declare XYZW quaternion elements")
        self._target = "write:robot:set_joint_position_target_index"
        self._kp = "write:robot:write_joint_stiffness_to_sim_index"
        self._kd = "write:robot:write_joint_damping_to_sim_index"
        if set(self._outputs) != {self._target, self._kp, self._kd}:
            raise ValueError(f"Unsupported LEAPP outputs: {list(self._outputs)}")
        for connection in ("state:robot:joint_pos", "state:robot:joint_vel"):
            self._check_joints(self._inputs[connection][1], contract)
        for _, spec in self._outputs.values():
            self._check_joints(spec, contract)
        if any(torch.device(node.device).type != "cpu" for node in self.inference.nodes.values()):
            raise ValueError("This G1 LEAPP adapter currently requires CPU inference")

        self.default_pose = self._to_motors(self._default)
        # Load and warm the full graph before DDS ownership or any ramp can start.
        self.kp = self.kd = None
        for _ in range(3):
            self.step(self.default_pose, np.zeros(core.NUM_ROBOT_MOTORS, np.float32),
                      np.array([1, 0, 0, 0], np.float32), np.zeros(3, np.float32),
                      np.zeros(3, np.float32), np.full(self.depth_shape, self._far, np.float32))
        self.reset()

    def _ports(self, keys, direction):
        result = {}
        for key in keys:
            node, name = key.split("/", 1)
            spec = next(item for item in self.inference.models[node][direction] if item["name"] == name)
            connection = spec.get("isaaclab_connection")
            if not connection or connection in result:
                raise ValueError(f"Missing or duplicate isaaclab_connection for {key}")
            result[connection] = (key, spec)
        return result

    @staticmethod
    def _check_joints(spec, contract):
        if (spec.get("element_names") != [contract["joint_names"]]
                or tuple(spec["shape"]) != (1, len(contract["joint_names"]))
                or spec["dtype"] != "float32"):
            raise ValueError(f"LEAPP joint shape/order differs from contract: {spec['name']}")

    def _to_motors(self, values):
        result = np.zeros(core.NUM_ROBOT_MOTORS, np.float32)
        result[self._map[self._mapped]] = values[self._mapped]
        return result

    def reset(self) -> None:
        """Let the official runtime restore the exported feedback initial state."""
        with self._torch.inference_mode():
            self.inference.reset()

    def step(self, joint_pos, joint_vel, quat_wxyz, gyro, command, depth_m, *, action_limit=None):
        """Return motor targets [rad] and graph targets [rad], without rescaling either.

        Sensor arguments follow G1DepthPolicyRunner.step. Invalid camera measurements
        become far range as on the existing G1 link; clipping, scaling and history stay
        inside the graph. The second return value is a target, not a raw action.
        """
        if action_limit is not None:
            raise ValueError("--action_limit cannot modify actions inside an exported LEAPP graph")
        q, dq = self._default.copy(), np.zeros_like(self._default)
        q[self._mapped] = np.asarray(joint_pos)[self._map[self._mapped]]
        dq[self._mapped] = np.asarray(joint_vel)[self._map[self._mapped]]
        quat = np.asarray(quat_wxyz, dtype=np.float32)
        gyro = np.asarray(gyro, dtype=np.float32)
        if core.IMU_TILT_WXYZ[0] != 1.0:
            # R_pelvis_world = R_imu_world * inverse(R_imu_to_pelvis).
            a, av = quat[0], quat[1:]
            b, bv = core.IMU_TILT_WXYZ[0], -core.IMU_TILT_WXYZ[1:]
            quat = np.concatenate(([a * b - np.dot(av, bv)], a * bv + b * av + np.cross(av, bv)))
            gyro = core.quat_apply_wxyz(core.IMU_TILT_WXYZ, gyro)
        frame = np.asarray(depth_m, dtype=np.float32)
        if frame.shape != self.depth_shape:
            raise ValueError(f"Depth shape {frame.shape}, expected {self.depth_shape}")
        frame = np.where(np.isfinite(frame) & (frame > 0), frame, self._far)
        if self._depth_rotation == 180:
            frame = frame[::-1, ::-1]
        values = {
            "state:robot:root_ang_vel_b": gyro[None],
            "state:robot:root_quat_w": quat[[1, 2, 3, 0]][None],
            "command:base_velocity": np.asarray(command, dtype=np.float32)[None],
            "state:robot:joint_pos": q[None],
            "state:robot:joint_vel": dq[None],
            "state:depth_camera:output:distance_to_image_plane": frame[None, :, :, None],
        }
        inputs = {}
        for connection, value in values.items():
            key, spec = self._inputs[connection]
            array = np.ascontiguousarray(value, dtype=np.float32)
            if array.shape != tuple(spec["shape"]) or not np.isfinite(array).all():
                raise ValueError(f"Invalid sensor input: {connection}")
            inputs[key] = self._torch.from_numpy(array)
        started = time.monotonic_ns()
        with self._torch.inference_mode():
            outputs = self.inference.run_policy(inputs)
        elapsed = time.monotonic_ns() - started
        converted = {}
        for connection, (key, spec) in self._outputs.items():
            value = outputs[key].detach().cpu().numpy()
            if value.shape != tuple(spec["shape"]) or not np.isfinite(value).all():
                raise ValueError(f"Invalid LEAPP output: {connection}")
            converted[connection] = value.reshape(-1).copy()
        kp, kd = self._to_motors(converted[self._kp]), self._to_motors(converted[self._kd])
        if (kp < 0).any() or (kd < 0).any():
            raise ValueError("LEAPP returned negative PD gains")
        if self.kp is None:
            self.kp, self.kd = kp, kd
        elif not (np.array_equal(kp, self.kp) and np.array_equal(kd, self.kd)):
            raise ValueError("This control loop requires constant exported PD gains")
        target = converted[self._target]
        if self.recorder is not None:
            self.recorder.arrays(
                "leapp_policy", started, step=self.recorder.step_index, inference_ns=elapsed,
                **{key: tensor.numpy() for key, tensor in inputs.items()},
                joint_target=target, kp=converted[self._kp], kd=converted[self._kd],
            )
        return self._to_motors(target), target
