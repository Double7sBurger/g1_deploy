# Copyright (c) 2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""G1 proprioception-history actor with a future motion-reference window."""

import numpy as np

from g1_deploy import core
from g1_deploy.tracking import G1TrackingPolicyRunner, rotation_matrix


class G1ModelBasedPolicyRunner(G1TrackingPolicyRunner):
    """Run the exported motion encoder and decoder; no world model is needed."""

    def _validate_contract(self, c):
        expected = dict(schema_version=1, policy_type="model_based_tracking", obs_1d_dim=1570,
                        action_dim=29, sensor_dim=93, history_length=10, goal_frames=10, goal_stride=5,
                        obs_normalization="embedded", root_body_name="pelvis", anchor_body_name="pelvis",
                        quaternion_order="wxyz", orientation_encoding="matrix_first_two_columns_row_major",
                        use_default_offset=True, clip_actions=None,
                        input_layout="history_oldest_first_then_goal_joints_then_orientation")
        if any(c.get(key) != value for key, value in expected.items()):
            raise ValueError("Unsupported model-based tracking contract")

    def reset(self):
        super().reset()
        self._history = None

    def begin_motion(self):
        # Keep the measured standing history while establishing the new heading origin.
        self.frame_index = self._next_frame = 0
        self._yaw_alignment = None
        self.tilt_error = 0.0

    def warmup(self, joint_pos, joint_vel, quat_wxyz, gyro):
        q, dq, quat, gyro = (np.asarray(v) for v in (joint_pos, joint_vel, quat_wxyz, gyro))
        if (q.shape != (29,) or dq.shape != (29,) or quat.shape != (4,) or gyro.shape != (3,)
                or not all(np.isfinite(v).all() for v in (q, dq, quat, gyro))):
            raise ValueError("Invalid model-based tracking sensor input")
        mounting = rotation_matrix(core.IMU_TILT_WXYZ)
        pelvis = rotation_matrix(quat) @ mounting.T
        sensor = np.concatenate((-pelvis[2], mounting @ gyro,
                                 q[self.policy_to_robot] - self.default_joint_pos,
                                 dq[self.policy_to_robot], self._last_action)).astype(np.float32)
        if self._history is None:
            self._history = np.repeat(sensor[None], 10, axis=0)
        else:
            self._history[:-1] = self._history[1:]
            self._history[-1] = sensor
        return pelvis

    def observe(self, joint_pos, joint_vel, quat_wxyz, gyro, *, frame_index=None):
        index = self._next_frame if frame_index is None else frame_index
        if index < 0 or index >= self.num_steps:
            raise ValueError("Reference motion has ended")
        self.frame_index = index
        pelvis = self.warmup(joint_pos, joint_vel, quat_wxyz, gyro)
        if self._yaw_alignment is None:
            initial = self.reference_rotation[0]
            yaw = np.arctan2(pelvis[1, 0], pelvis[0, 0]) - np.arctan2(initial[1, 0], initial[0, 0])
            c, s = np.cos(yaw), np.sin(yaw)
            self._yaw_alignment = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        frames = np.minimum(index + np.arange(10) * 5, self.num_steps - 1)
        references = self._yaw_alignment @ self.reference_rotation[frames]
        relative = pelvis.T @ references
        self.tilt_error = float(np.arccos(np.clip(np.dot(pelvis[2], references[0, 2]), -1, 1)))
        joints = np.concatenate((self.reference_pos[frames], self.reference_vel[frames]), axis=-1)
        return np.concatenate((self._history.ravel(), joints.ravel(), relative[..., :2].ravel())).astype(np.float32)
