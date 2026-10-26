# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single-clip G1 motion tracking from encoders and the pelvis IMU."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from g1_deploy import core
from g1_deploy.hardware import MOTOR_NAMES


def rotation_matrix(quaternion):
    """Convert unit WXYZ quaternions to body-to-world rotation matrices."""
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape[-1] != 4 or not np.isfinite(q).all():
        raise ValueError("Invalid WXYZ quaternion")
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1e-6):
        raise ValueError("Zero quaternion")
    w, x, y, z = np.moveaxis(q / norm, -1, 0)
    return np.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                     2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                     2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), axis=-1).reshape(q.shape[:-1] + (3, 3))


def torso_rotation(pelvis_rotation, motor_positions):
    """Official G1 pelvis-to-torso FK using waist yaw, roll and pitch [rad]."""
    yaw, roll, pitch = np.asarray(motor_positions)[12:15]
    cy, sy, cr, sr, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    return pelvis_rotation @ rz @ rx @ ry


class G1TrackingPolicyRunner:
    """Run the 154-input, 29-output actor with its own training contract.

    The clip plays once. There is no simulator reset or automatic jump to another
    reference phase. The caller owns the 50 Hz clock and the shutdown sequence.
    """

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.contract = json.loads((self.directory / "contract.json").read_text())
        c = self.contract
        if (c.get("policy_type") != "motion_tracking" or c.get("schema_version") != 1
                or c.get("obs_1d_dim") != 154 or c.get("action_dim") != 29
                or c.get("history_length") != 1 or c.get("obs_normalization") is not False
                or c.get("anchor_body_name") != "torso_link" or c.get("root_body_name") != "pelvis"
                or c.get("quaternion_order") != "wxyz"
                or c.get("orientation_encoding") != "matrix_first_two_columns_row_major"
                or c.get("use_default_offset") is not True or c.get("clip_actions") is not None):
            raise ValueError("Unsupported motion-tracking contract")
        terms = {"command": 58, "motion_anchor_ori_b": 6, "base_ang_vel": 3,
                 "joint_pos": 29, "joint_vel": 29, "actions": 29}
        if list(c.get("obs_1d_terms", {}).items()) != list(terms.items()):
            raise ValueError("Tracking observation order differs from the trained contract")
        self.joint_names = list(c["joint_names"])
        motors = [name + "_joint" for name in MOTOR_NAMES]
        if len(self.joint_names) != 29 or set(self.joint_names) != set(motors):
            raise ValueError("Tracking requires exactly the 29 G1 body joints")
        self.policy_to_robot = np.array([motors.index(name) for name in self.joint_names])
        self.control_dt = float(c["control_dt"])
        if not np.isclose(self.control_dt, core.CONTROL_DT, rtol=0, atol=1e-8):
            raise ValueError("Tracking policy must run at the trained 50 Hz")
        self.default_joint_pos = self._vector("default_joint_pos")
        self.default_joint_vel = self._vector("default_joint_vel")
        self.action_scale = self._vector("action_scale")
        stiffness, damping = self._vector("stiffness"), self._vector("damping")
        if (self.action_scale <= 0).any() or (stiffness <= 0).any() or (damping < 0).any():
            raise ValueError("Invalid tracking action scale or PD gains")
        self.kp, self.kd = self.to_motors(stiffness), self.to_motors(damping)
        motion_path = self.directory / c["motion_file"]
        self._check_hash(motion_path, c["motion_sha256"])
        self._check_hash(self.directory / "policy.pt", c["policy_sha256"])
        with np.load(motion_path, allow_pickle=False) as motion:
            if "joint_names" in motion and "body_names" in motion:
                names = {key: motion[key].tolist() for key in ("joint_names", "body_names")}
            else:
                names = json.loads((self.directory / c["motion_metadata_file"]).read_text())
                if names.get("sha256") != c["motion_sha256"]:
                    raise ValueError("Motion names do not describe the packaged reference")
            order = [names["joint_names"].index(name) for name in self.joint_names]
            self.reference_pos = np.asarray(motion["joint_pos"][:, order], dtype=np.float32)
            self.reference_vel = np.asarray(motion["joint_vel"][:, order], dtype=np.float32)
            quat = np.asarray(motion["body_quat_w"][:, names["body_names"].index("torso_link")])
            if not np.isclose(float(motion["fps"]) * self.control_dt, 1, rtol=0, atol=1e-6):
                raise ValueError("Reference FPS differs from the policy control rate")
        n = len(self.reference_pos)
        if (n < 2 or self.reference_pos.shape != (n, 29) or self.reference_vel.shape != (n, 29)
                or quat.shape != (n, 4) or not np.isfinite(self.reference_pos).all()
                or not np.isfinite(self.reference_vel).all()
                or not np.allclose(np.linalg.norm(quat, axis=-1), 1, rtol=0, atol=1e-3)):
            raise ValueError("Invalid motion arrays or reference quaternions")
        self.reference_rotation = rotation_matrix(quat)
        self.num_steps = n - 1
        self.duration = self.num_steps * self.control_dt
        self.default_pose = self.to_motors(self.reference_pos[0])
        self.policy = core.load_policy(str(self.directory / "policy.pt"), obs_dim=154, action_dim=29)
        self.reset()

    @staticmethod
    def _check_hash(path, expected):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Artifact checksum mismatch: {path}")

    def _vector(self, name):
        value = np.asarray(self.contract[name], dtype=np.float32)
        if value.shape != (29,) or not np.isfinite(value).all():
            raise ValueError(f"Invalid tracking contract field: {name}")
        return value

    def to_motors(self, values):
        result = np.empty(29, dtype=np.float32)
        result[self.policy_to_robot] = values
        return result

    def set_previous_target(self, target):
        """Feed back the applied target when another controller blends the output."""
        target = np.asarray(target, dtype=np.float32)
        if target.shape != (29,) or not np.isfinite(target).all():
            raise ValueError("Invalid previous joint target")
        self._last_action = (target[self.policy_to_robot] - self.default_joint_pos) / self.action_scale

    def reset(self):
        self.frame_index = 0
        self._next_frame = 0
        self._yaw_alignment = None
        self._last_action = np.zeros(29, dtype=np.float32)
        self.tilt_error = 0.0

    def index_at(self, elapsed_s):
        """Reference frame for ``elapsed_s``, held at the last one once the clip has played.

        Every packaged clip ends standing still, so a frozen final frame is a reference the policy
        can keep tracking rather than a cliff.
        """
        if not np.isfinite(elapsed_s) or elapsed_s < 0:
            raise ValueError("Invalid reference time")
        return min(int(elapsed_s / self.control_dt + 1e-6), self.num_steps - 1)

    def command_at(self, elapsed_s):
        index = self.index_at(elapsed_s)
        return np.concatenate((self.reference_pos[index], self.reference_vel[index]))

    def observe(self, joint_pos, joint_vel, quat_wxyz, gyro, *, frame_index=None):
        """Assemble the exact training observation without history or sensor noise."""
        q, dq = np.asarray(joint_pos), np.asarray(joint_vel)
        gyro = np.asarray(gyro, dtype=np.float32)
        quat = np.asarray(quat_wxyz)
        if (q.shape != (29,) or dq.shape != (29,) or gyro.shape != (3,) or quat.shape != (4,)
                or not all(np.isfinite(v).all() for v in (q, dq, gyro, quat))):
            raise ValueError("Invalid tracking sensor input")
        index = self._next_frame if frame_index is None else frame_index
        if index < 0 or index >= self.num_steps:
            raise ValueError("Reference motion has ended")
        self.frame_index = index
        # IMU mounting rotation maps sensor coordinates into the trained pelvis frame.
        mounting = rotation_matrix(core.IMU_TILT_WXYZ)
        pelvis = rotation_matrix(quat) @ mounting.T
        torso = torso_rotation(pelvis, q)
        reference = self.reference_rotation[index]
        if self._yaw_alignment is None:
            initial = self.reference_rotation[0]
            yaw = np.arctan2(torso[1, 0], torso[0, 0]) - np.arctan2(initial[1, 0], initial[0, 0])
            c, s = np.cos(yaw), np.sin(yaw)
            self._yaw_alignment = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        reference = self._yaw_alignment @ reference
        relative = torso.T @ reference
        # Compare gravity directions in the two torso frames, allowing intentional inversion.
        self.tilt_error = float(np.arccos(np.clip(np.dot(torso[2], reference[2]), -1, 1)))
        return np.concatenate((self.reference_pos[index], self.reference_vel[index],
                               relative[:, :2].reshape(-1), mounting @ gyro,
                               q[self.policy_to_robot] - self.default_joint_pos,
                               dq[self.policy_to_robot] - self.default_joint_vel,
                               self._last_action)).astype(np.float32)

    def step(self, joint_pos, joint_vel, quat_wxyz, gyro, *, elapsed_s=None, action_limit=None):
        index = self._next_frame if elapsed_s is None else self.index_at(elapsed_s)
        obs = self.observe(joint_pos, joint_vel, quat_wxyz, gyro, frame_index=index)
        action = np.asarray(self.policy(obs[None]), dtype=np.float32)
        if action.shape != (1, 29) or not np.isfinite(action).all():
            raise ValueError("Invalid tracking policy output")
        action = action[0]
        if action_limit is not None:
            if not np.isfinite(action_limit) or action_limit <= 0:
                raise ValueError("action_limit must be positive and finite")
            action = np.clip(action, -action_limit, action_limit)
        self._last_action = action.copy()
        target = self.to_motors(self.default_joint_pos + self.action_scale * action)
        if not np.isfinite(target).all():
            raise ValueError("Invalid tracking joint target")
        self._next_frame = index + 1
        return target, action


class StandingTrackingRunner:
    """Keep a stationary reference active, then blend into a separate motion actor.

    Both actors keep receiving fresh sensors. The motion clock starts only after
    confirmation and the transition; waiting never consumes any reference frames.
    """

    def __init__(self, standing, motion, transition_s=1.0, engage_s=0.3):
        if standing is motion:
            raise ValueError("Standing and motion need separate runner state")
        if any(not np.allclose(getattr(standing, key), getattr(motion, key), rtol=1e-6, atol=1e-6)
               for key in ("kp", "kd")):
            raise ValueError("Standing and tracking policies must use the same motor Kp/Kd")
        if not np.isfinite(transition_s) or transition_s <= 0 or not np.isfinite(engage_s) or engage_s <= 0:
            raise ValueError("Standing transition and engagement durations must be positive and finite")
        self.standing, self.motion = standing, motion
        self.transition_s, self.engage_s = float(transition_s), float(engage_s)
        self.reset()

    def reset(self):
        self.standing.reset()
        self.motion.reset()
        self.phase = "standing"
        self.engaged_at = self.transition_at = self.motion_start = None
        self.initial_target = self.last_target = None
        self.tilt_error = 0.0
        self._motion_seeded = False

    @staticmethod
    def _blend(fraction):
        u = float(np.clip(fraction, 0, 1))
        return u**3 * (10 + u * (-15 + 6*u))

    def start_motion(self, now):
        if not np.isfinite(now):
            raise ValueError("Invalid standing clock")
        if self.engaged_at is None:
            raise RuntimeError("Standing must be engaged before go2")
        if self.transition_at is not None:
            return
        self.transition_at = max(float(now), self.engaged_at + self.engage_s)
        self.motion_start = self.transition_at + self.transition_s

    def elapsed_s(self, now):
        return 0.0 if self.motion_start is None else max(0.0, float(now) - self.motion_start)

    def command_at(self, now):
        if self.transition_at is None or now < self.transition_at:
            return self.standing.command_at(0.0)
        return self.motion.command_at(self.elapsed_s(now))

    def set_previous_target(self, target):
        """Record the applied target, including any caller-requested joint clamp."""
        target = np.asarray(target, dtype=np.float32)
        if target.shape != (29,) or not np.isfinite(target).all():
            raise ValueError("Invalid previous joint target")
        self.last_target = target.copy()
        if self.phase != "motion":
            self.standing.set_previous_target(target)
        if self._motion_seeded:
            self.motion.set_previous_target(target)

    def step(self, joint_pos, joint_vel, quat_wxyz, gyro, *, now, action_limit=None):
        if not np.isfinite(now):
            raise ValueError("Invalid standing clock")
        if self.engaged_at is None:
            self.engaged_at = float(now)
            self.initial_target = np.asarray(joint_pos, dtype=np.float32).copy()
        sensors = joint_pos, joint_vel, quat_wxyz, gyro
        if self.motion_start is not None and now >= self.motion_start - 1e-8:
            self.phase = "motion"
        elif self.transition_at is not None and now >= self.transition_at - 1e-8:
            self.phase = "transition"
        else:
            self.phase = "standing"

        if self.phase != "motion":
            target, action = self.standing.step(*sensors, elapsed_s=0.0, action_limit=action_limit)
            self.tilt_error = self.standing.tilt_error
        if self.phase != "standing":
            if not self._motion_seeded:
                self.motion.reset()
                self.motion.set_previous_target(self.last_target)
                self._motion_seeded = True
            moving, action = self.motion.step(*sensors, elapsed_s=self.elapsed_s(now), action_limit=action_limit)
            if self.phase == "transition":
                alpha = self._blend((now - self.transition_at) / self.transition_s)
                target = (1-alpha) * target + alpha * moving
                self.tilt_error = max(self.tilt_error, self.motion.tilt_error)
            else:
                target = moving
                self.tilt_error = self.motion.tilt_error
        if now < self.engaged_at + self.engage_s:
            alpha = self._blend((now - self.engaged_at) / self.engage_s)
            target = (1-alpha) * self.initial_target + alpha * target
        self.set_previous_target(target)
        return target, action
