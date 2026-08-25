# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot-side half of ``Isaac-Velocity-Flat-G1-DR``, with no Isaac Lab dependency.

This is the piece that runs on the G1 -- or against ``unitree_mujoco``, which speaks the same DDS
interface. It owns the contract that training defined: which joints exist and in what order, what
the observation vector looks like, and how a policy output becomes a joint target. Nothing here
imports Isaac Lab, so it can be dropped into a deployment process that has only numpy and a
TorchScript or ONNX runtime.

The contract is asserted, not assumed: ``verify_against_isaaclab.py`` drives the real environment
and this module from the same simulated state and requires the two observation vectors to agree to
1e-6. Run it after any change here.

**Index spaces.** Two coexist and confusing them is the classic deployment bug:

* *policy space* -- the 37 joints of ``g1_minimal.usd``, in Isaac Lab's articulation order. The
  network's input and output live here.
* *robot space* -- the 29 motors of :class:`G1JointIndex` in ``unitree_sdk2``, which is what
  ``rt/lowstate`` reports and ``rt/lowcmd`` drives.

:data:`POLICY_TO_ROBOT` maps one to the other by joint name.  The articulation order is backend
dependent: PhysX does not expose the joints in the depth-first order used by Newton/MuJoCo.  The
network follows the order of the backend it was trained with, so treating its indices as an
anatomical left-leg/right-leg sequence silently drives the wrong motors.
"""

from __future__ import annotations

import numpy as np

##
# Joint layout
##

_PHYSX_JOINT_NAMES = [
    # PhysX articulation/action order for g1_minimal.usd.  Verified at runtime by
    # verify_against_isaaclab.py.  This order is deliberately not grouped anatomically.
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "torso_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_elbow_pitch_joint",
    "right_elbow_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_elbow_roll_joint",
    "right_elbow_roll_joint",
    "left_five_joint",
    "left_three_joint",
    "left_zero_joint",
    "right_five_joint",
    "right_three_joint",
    "right_zero_joint",
    "left_six_joint",
    "left_four_joint",
    "left_one_joint",
    "right_six_joint",
    "right_four_joint",
    "right_one_joint",
    "left_two_joint",
    "right_two_joint",
]
_NEWTON_JOINT_NAMES = [
    # Newton/MJWarp articulation order: depth first, whole left leg then whole right leg.
    # Dumped from the running articulation, not assumed.
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "torso_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_elbow_roll_joint",
    "left_five_joint",
    "left_six_joint",
    "left_three_joint",
    "left_four_joint",
    "left_zero_joint",
    "left_one_joint",
    "left_two_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_elbow_roll_joint",
    "right_five_joint",
    "right_six_joint",
    "right_three_joint",
    "right_four_joint",
    "right_zero_joint",
    "right_one_joint",
    "right_two_joint",
]

JOINT_ORDER_BY_BACKEND = {"physx": _PHYSX_JOINT_NAMES, "newton": _NEWTON_JOINT_NAMES}
"""Articulation order per training backend -- they genuinely differ for the same USD.

PhysX enumerates breadth first (both hip pitches, then the waist, then both hip rolls...);
Newton enumerates depth first (the whole left leg, then the whole right leg). A checkpoint
follows whichever backend trained it, and using the other order drives the wrong motors
without raising anything -- a Newton policy replayed under the PhysX table dropped from a
3.5 s median survival to 0.9 s, which reads exactly like a bad policy.
"""

POLICY_JOINT_NAMES = list(_PHYSX_JOINT_NAMES)
"""Active order. Call :func:`set_policy_backend` before anything else to switch."""

NUM_POLICY_JOINTS = len(POLICY_JOINT_NAMES)
NUM_ROBOT_MOTORS = 29
"""``G1_NUM_MOTOR`` in ``unitree_sdk2/example/g1/low_level/g1_ankle_swing_example.cpp``."""

_ROBOT_INDEX_BY_POLICY_NAME = {
    "left_hip_pitch_joint": 0,
    "left_hip_roll_joint": 1,
    "left_hip_yaw_joint": 2,
    "left_knee_joint": 3,
    "left_ankle_pitch_joint": 4,
    "left_ankle_roll_joint": 5,
    "right_hip_pitch_joint": 6,
    "right_hip_roll_joint": 7,
    "right_hip_yaw_joint": 8,
    "right_knee_joint": 9,
    "right_ankle_pitch_joint": 10,
    "right_ankle_roll_joint": 11,
    "torso_joint": 12,
    "left_shoulder_pitch_joint": 15,
    "left_shoulder_roll_joint": 16,
    "left_shoulder_yaw_joint": 17,
    "left_elbow_pitch_joint": 18,
    "left_elbow_roll_joint": 19,
    "right_shoulder_pitch_joint": 22,
    "right_shoulder_roll_joint": 23,
    "right_shoulder_yaw_joint": 24,
    "right_elbow_pitch_joint": 25,
    "right_elbow_roll_joint": 26,
}
POLICY_TO_ROBOT = np.array(
    [_ROBOT_INDEX_BY_POLICY_NAME.get(name, -1) for name in POLICY_JOINT_NAMES],
    dtype=np.int32,
)
""":data:`POLICY_JOINT_NAMES` index -> ``G1JointIndex``; ``-1`` for joints the robot does not have.

The seven-per-hand entries are the three-finger hand in ``g1_minimal.usd``. A standard G1 does not
drive them over ``rt/lowcmd`` -- a Dex3 hand is a separate device on its own topic -- so the policy's
commands for them are dropped and their state reads back as "at default" (see
:meth:`G1PolicyRunner.observe`).

Robot motors with no policy counterpart -- 13/14 (WaistRoll, WaistPitch) and 20/21/27/28 (wrist pitch
and yaw) -- exist only on the 29-DoF variant. Hold them at zero; :data:`UNMAPPED_ROBOT_MOTORS` lists
them.
"""

UNMAPPED_ROBOT_MOTORS = np.array(
    sorted(set(range(NUM_ROBOT_MOTORS)) - set(POLICY_TO_ROBOT[POLICY_TO_ROBOT >= 0].tolist())),
    dtype=np.int32,
)
"""Motors present on a 29-DoF G1 that the policy never commands. Absent on the 23-DoF variant."""

LEG_POLICY_INDICES = np.array(
    [i for i, name in enumerate(POLICY_JOINT_NAMES) if any(part in name for part in ("hip_", "knee_", "ankle_"))],
    dtype=np.int32,
)
"""Indices of the twelve leg actions in backend policy space."""

_DEFAULT_JOINT_POS_BY_NAME = {
    "left_hip_pitch_joint": -0.20,
    "right_hip_pitch_joint": -0.20,
    "left_knee_joint": 0.42,
    "right_knee_joint": 0.42,
    "left_ankle_pitch_joint": -0.23,
    "right_ankle_pitch_joint": -0.23,
    "left_elbow_pitch_joint": 0.87,
    "right_elbow_pitch_joint": 0.87,
    "left_shoulder_pitch_joint": 0.35,
    "right_shoulder_pitch_joint": 0.35,
    "left_shoulder_roll_joint": 0.16,
    "right_shoulder_roll_joint": -0.16,
    "left_one_joint": 1.0,
    "right_one_joint": -1.0,
    "left_two_joint": 0.52,
    "right_two_joint": -0.52,
}
"""Isaac Lab's ``default_joint_pos`` [rad] keyed by joint name, so it survives an order switch."""

DEFAULT_JOINT_POS = np.zeros(NUM_POLICY_JOINTS, dtype=np.float32)
"""Default pose in the active order; the offset the action term adds to."""
for _name, _value in _DEFAULT_JOINT_POS_BY_NAME.items():
    DEFAULT_JOINT_POS[POLICY_JOINT_NAMES.index(_name)] = _value

##
# Control constants -- these are the trained-for values, not suggestions
##

CONTROL_DT = 0.02
"""Policy period [s]: Isaac Lab's ``physics_dt`` 0.005 times ``decimation`` 4, i.e. 50 Hz."""

ACTION_SCALE = 0.5
"""``JointPositionActionCfg.scale``; the target is ``default + scale * action``."""

HISTORY_LENGTH = 5
"""Frames stacked per observation term, oldest first."""

OBS_DIM = HISTORY_LENGTH * (3 + 3 + 3 + NUM_POLICY_JOINTS * 3)
"""600 for the deployable policy. A 615 here means the checkpoint still expects ``base_lin_vel``."""

NOMINAL_KP = np.zeros(NUM_ROBOT_MOTORS, dtype=np.float32)
NOMINAL_KD = np.zeros(NUM_ROBOT_MOTORS, dtype=np.float32)
"""Gains the policy was trained against, in robot index space [N·m/rad], [N·m·s/rad].

Taken from ``G1_CFG`` in ``isaaclab_assets``. The randomization spanned 0.5x-2x around these, so
setting anything else moves the robot off the centre of the trained band -- and the out-of-range
sweep showed half the environments fall over at 4x. These are not the SDK example's gains.
"""
for _idx, _kp, _kd in [
    *[(i, 150.0, 5.0) for i in (1, 2, 7, 8)],  # hip roll, hip yaw
    *[(i, 200.0, 5.0) for i in (0, 3, 6, 9, 12)],  # hip pitch, knee, waist yaw
    *[(i, 20.0, 2.0) for i in (4, 5, 10, 11)],  # ankles
    *[(i, 40.0, 10.0) for i in (15, 16, 17, 18, 19, 22, 23, 24, 25, 26)],  # shoulders, elbows
]:
    NOMINAL_KP[_idx], NOMINAL_KD[_idx] = _kp, _kd


def set_policy_backend(backend: str) -> None:
    """Switch the joint order to the backend a checkpoint was trained with.

    Everything downstream of the order -- the robot mapping, the default pose, the leg indices --
    is rebuilt, so this has to run before any of those module constants are read and before a
    :class:`G1PolicyRunner` is built. Importing modules that captured the old values by name will
    not see the change.

    ``verify_against_isaaclab.py`` asserts the active order against the live articulation, so a
    wrong choice fails loudly there rather than silently driving the wrong motors.

    Args:
        backend: ``"physx"`` or ``"newton"``.

    Raises:
        ValueError: If the backend is unknown.
    """
    global POLICY_JOINT_NAMES, POLICY_TO_ROBOT, UNMAPPED_ROBOT_MOTORS, LEG_POLICY_INDICES, DEFAULT_JOINT_POS

    if backend not in JOINT_ORDER_BY_BACKEND:
        raise ValueError(f"unknown backend {backend!r}; expected one of {sorted(JOINT_ORDER_BY_BACKEND)}")

    POLICY_JOINT_NAMES = list(JOINT_ORDER_BY_BACKEND[backend])
    POLICY_TO_ROBOT = np.array(
        [_ROBOT_INDEX_BY_POLICY_NAME.get(name, -1) for name in POLICY_JOINT_NAMES], dtype=np.int32
    )
    UNMAPPED_ROBOT_MOTORS = np.array(
        sorted(set(range(NUM_ROBOT_MOTORS)) - set(POLICY_TO_ROBOT[POLICY_TO_ROBOT >= 0].tolist())), dtype=np.int32
    )
    LEG_POLICY_INDICES = np.array(
        [i for i, name in enumerate(POLICY_JOINT_NAMES) if any(p in name for p in ("hip_", "knee_", "ankle_"))],
        dtype=np.int32,
    )
    DEFAULT_JOINT_POS = np.zeros(NUM_POLICY_JOINTS, dtype=np.float32)
    for name, value in _DEFAULT_JOINT_POS_BY_NAME.items():
        DEFAULT_JOINT_POS[POLICY_JOINT_NAMES.index(name)] = value


_UNMAPPED_HOLD_GAINS = {
    # Waist roll and pitch. These carry the whole upper body and are the reason this table exists as
    # something other than an afterthought: ``g1_minimal.usd`` has a single ``torso_joint`` (yaw), so
    # during training the torso is *welded* to the pelvis in roll and pitch and the policy never once
    # felt it move. A 29-DoF G1 -- and the 29-DoF MJCF -- really does have those two joints, so
    # whatever holds them has to be stiff enough to look welded, or the policy is stabilizing a robot
    # with a floppy upper body it has no model of.
    #
    # Measured, holding at 40/1: the torso pitched 24 deg away from the pelvis and survival at
    # vx=0.5 was 8.4 s (2/5 runs reaching 12 s). At 200/5 the same deviation is 1.5 deg and survival
    # is 12 s in 5/5. 200/5 is also exactly what training uses for waist yaw, and the peak torque it
    # asks for is ~21 N.m against the motor's 88 N.m.
    13: (200.0, 5.0),
    14: (200.0, 5.0),
    # Wrist pitch and yaw. Light, distal, and not load bearing -- a soft hold is fine and keeps them
    # from fighting the arm swing.
    20: (40.0, 1.0),
    21: (40.0, 1.0),
    27: (40.0, 1.0),
    28: (40.0, 1.0),
}
"""``G1JointIndex`` -> ``(kp, kd)`` for motors the policy never commands."""


def control_gains() -> tuple[np.ndarray, np.ndarray]:
    """Full PD gains in robot index space, policy-driven motors and held motors together.

    Returns:
        ``(kp, kd)``, each shape ``(29,)``, in [N·m/rad] and [N·m·s/rad]. Motors the policy drives
        get :data:`NOMINAL_KP` / :data:`NOMINAL_KD`; the rest get :data:`_UNMAPPED_HOLD_GAINS`.
    """
    kp, kd = NOMINAL_KP.copy(), NOMINAL_KD.copy()
    for motor in UNMAPPED_ROBOT_MOTORS.tolist():
        kp[motor], kd[motor] = _UNMAPPED_HOLD_GAINS[motor]
    return kp, kd


##
# Velocity command
##

HEADING_CONTROL_STIFFNESS = 0.5
"""``UniformVelocityCommandCfg.heading_control_stiffness`` for this task [1/s]."""

YAW_RATE_LIMITS = (-1.0, 1.0)
"""``ranges.ang_vel_z`` [rad/s]. During training this clips the heading controller, nothing else."""


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle [rad] to :math:`[-\\pi, \\pi]`, matching ``isaaclab.utils.math.wrap_to_pi``."""
    wrapped = (angle + np.pi) % (2.0 * np.pi)
    return float(np.pi if wrapped == 0.0 and angle > 0.0 else wrapped - np.pi)


def heading_from_quat_wxyz(quat_wxyz: np.ndarray) -> float:
    """Yaw of the body frame in the world frame [rad], from a ``(w, x, y, z)`` quaternion.

    This is Isaac Lab's ``heading_w``: the yaw of the body x-axis, ``atan2`` of the first column of
    the rotation matrix. Taking the body forward direction and taking the quaternion's yaw component
    are the same expression here, so there is no convention to get wrong -- unlike the quaternion
    order itself, which is WXYZ on the robot and XYZW in Isaac Lab (see
    :func:`quat_apply_inverse_wxyz`).
    """
    w, x, y, z = (float(v) for v in quat_wxyz)
    return float(np.arctan2(2.0 * (x * y + w * z), 1.0 - 2.0 * (y * y + z * z)))


def yaw_rate_from_heading(heading_target: float, quat_wxyz: np.ndarray) -> float:
    """Reproduce the yaw-rate command the way training generated it.

    .. attention::
        The third element of the command is **not** a user input during training. This task sets
        ``heading_command=True`` with ``rel_heading_envs=1.0``, so ``UniformVelocityCommand`` throws
        away the sampled ``ang_vel_z`` and overwrites it every step with a proportional controller on
        heading error. What the policy learned to read there is "how far off my heading target am I",
        not "how fast someone wants me to spin".

        Sending a constant yaw rate instead -- in particular a constant zero -- is therefore off
        distribution: zero only ever occurred when the robot was already pointing at its target, so
        the policy never saw "yaw command is zero *while* drifting off heading" and gets no signal to
        correct. A forward-only run drifted 0.51 m sideways against 0.57 m forward with a hard-wired
        ``wz = 0``.

    Args:
        heading_target: Desired yaw in the world frame [rad].
        quat_wxyz: Pelvis orientation from the IMU as ``(w, x, y, z)``.

    Returns:
        The yaw-rate command [rad/s], clipped to :data:`YAW_RATE_LIMITS`.
    """
    error = wrap_to_pi(heading_target - heading_from_quat_wxyz(quat_wxyz))
    return float(np.clip(HEADING_CONTROL_STIFFNESS * error, *YAW_RATE_LIMITS))


def quat_apply_inverse_wxyz(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate a world-frame vector into the body frame, taking a ``(w, x, y, z)`` quaternion.

    Transcribed from ``isaaclab.utils.math.quat_apply_inverse`` rather than derived, so there is no
    algebraically-equivalent-but-differently-rounded variant to argue about.

    .. attention::
        The two ends of this pipeline disagree on quaternion order. Isaac Lab 3.0 moved to **XYZW**
        (``root_quat_w`` is tagged ``QUAT_XYZW_ELEMENT_NAMES``); the Unitree SDK reports IMU
        orientation as **WXYZ**. This function takes the robot's convention and reorders internally,
        because the robot is what feeds it in production. Both orders hold normalized unit
        quaternions of the same shape, so getting this wrong passes every structural check and only
        shows up as a robot that leans the wrong way.

    Args:
        quat_wxyz: Body orientation as ``(w, x, y, z)`` -- the Unitree/``LowState_`` convention.
        vec: World-frame vector, shape ``(3,)``.

    Returns:
        The vector expressed in the body frame, shape ``(3,)``.
    """
    w, xyz = float(quat_wxyz[0]), np.asarray(quat_wxyz[1:], dtype=np.float32)
    t = np.cross(xyz, vec) * 2.0
    return (vec - w * t + np.cross(xyz, t)).astype(np.float32)


class G1PolicyRunner:
    """Assembles observations, runs the policy, and emits joint targets at 50 Hz.

    The caller owns the transport. Feed it whatever ``rt/lowstate`` reported and hand the returned
    targets to ``rt/lowcmd``; the same object works unchanged against ``unitree_mujoco``.
    """

    def __init__(self, policy, *, num_robot_motors: int = NUM_ROBOT_MOTORS):
        """Initialize the runner.

        Args:
            policy: Callable mapping a ``(1, OBS_DIM)`` float32 array to a ``(1, 37)`` action array.
                A ``torch.jit`` module wrapped to accept and return numpy works, as does an ONNX
                session.
            num_robot_motors: 29 for the standard G1 low-level interface.
        """
        self._policy = policy
        self._num_robot_motors = num_robot_motors
        self._mapped = POLICY_TO_ROBOT >= 0
        self._history: list[np.ndarray] = []
        self._last_action = np.zeros(NUM_POLICY_JOINTS, dtype=np.float32)

    def reset(self) -> None:
        """Clear the observation history and the remembered action.

        Call this before the first control step and after any interruption of the 50 Hz loop -- a
        stale history is a silent way to feed the policy a discontinuity it never saw in training.
        """
        self._history.clear()
        self._last_action[:] = 0.0

    def observe(
        self,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        quat_wxyz: np.ndarray,
        gyro: np.ndarray,
        command: np.ndarray,
        hand_joint_pos_rel: np.ndarray | None = None,
        hand_joint_vel: np.ndarray | None = None,
    ) -> np.ndarray:
        """Build the flat observation vector for the current step.

        The layout is term-major with each term's own history inside it, because Isaac Lab buffers
        history per term and concatenates afterwards. Frame-major stacking -- the intuitive guess --
        produces a vector of the right length that the policy reads as noise.

        Args:
            joint_pos: Measured joint positions in robot index space [rad], shape ``(29,)``.
            joint_vel: Measured joint velocities in robot index space [rad/s], shape ``(29,)``.
            quat_wxyz: Pelvis orientation from the IMU as ``(w, x, y, z)``.
            gyro: Pelvis angular velocity from the IMU, body frame [rad/s], shape ``(3,)``.
            command: Velocity command ``(vx, vy, wz)`` in [m/s, m/s, rad/s].
            hand_joint_pos_rel: Optional deviation from default [rad] for the 14 finger joints, in
                policy-space order. ``None`` assumes they sit at their default pose, which is what a
                robot without a Dex3 hand effectively reports. Supply the true values only when
                cross-checking against a simulation that does model them.
            hand_joint_vel: Optional finger joint velocities [rad/s]; ``None`` assumes zero.

        Returns:
            Observation of shape ``(OBS_DIM,)``, float32.
        """
        # Robot -> policy space. Unmapped slots default to zero, which reads as "this joint sits at
        # its default" -- the pose the joint-deviation penalties hold the hands in during training.
        q_rel = np.zeros(NUM_POLICY_JOINTS, dtype=np.float32)
        dq = np.zeros(NUM_POLICY_JOINTS, dtype=np.float32)
        robot_idx = POLICY_TO_ROBOT[self._mapped]
        q_rel[self._mapped] = joint_pos[robot_idx] - DEFAULT_JOINT_POS[self._mapped]
        dq[self._mapped] = joint_vel[robot_idx]
        if hand_joint_pos_rel is not None:
            q_rel[~self._mapped] = hand_joint_pos_rel
        if hand_joint_vel is not None:
            dq[~self._mapped] = hand_joint_vel

        projected_gravity = quat_apply_inverse_wxyz(
            np.asarray(quat_wxyz, dtype=np.float32), np.array([0.0, 0.0, -1.0], dtype=np.float32)
        )

        frame = [
            np.asarray(gyro, dtype=np.float32),
            projected_gravity,
            np.asarray(command, dtype=np.float32),
            q_rel,
            dq,
            self._last_action.copy(),
        ]
        # On the first step Isaac Lab's circular buffer is filled with copies of the current frame,
        # so a fresh policy sees a constant history rather than zeros.
        if not self._history:
            self._history = [frame] * HISTORY_LENGTH
        else:
            self._history = self._history[1:] + [frame]

        return np.concatenate([np.concatenate([f[term] for f in self._history]) for term in range(len(frame))]).astype(
            np.float32
        )

    def step(
        self,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        quat_wxyz: np.ndarray,
        gyro: np.ndarray,
        command: np.ndarray,
        *,
        action_limit: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run one 50 Hz control step.

        Args:
            joint_pos: See :meth:`observe`.
            joint_vel: See :meth:`observe`.
            quat_wxyz: See :meth:`observe`.
            gyro: See :meth:`observe`.
            command: See :meth:`observe`.
            action_limit: Symmetric clamp on the raw policy output, or ``None`` to leave it alone.

                Defaults to ``None`` because training clips nothing (``clip_actions=None``,
                ``JointPositionActionCfg.clip=None``) and a clamp here is therefore a change of
                behaviour, not a safety net. It was 5.0, which sounded harmless and was not: on a
                recorded rollout of this task's own checkpoint 7.6% of actions exceeded it -- the
                95th percentile is 7.25 and the maximum 11.39 -- so the deployment was quietly
                driving a different policy from the one that was trained.

                Set it on hardware, where an unbounded target really is dangerous, but set it from
                measured action statistics rather than from a round number, and treat whatever it
                clips as train/deploy divergence to be reported.

        Returns:
            ``(joint_target, action)`` -- the position target in robot index space [rad], shape
            ``(29,)``, with unmapped motors at zero; and the raw 37-dim policy output, which is fed
            back as the next step's ``last_action``.
        """
        obs = self.observe(joint_pos, joint_vel, quat_wxyz, gyro, command)
        action = np.asarray(self._policy(obs[None, :]), dtype=np.float32).reshape(NUM_POLICY_JOINTS)
        if action_limit is not None:
            action = np.clip(action, -action_limit, action_limit)
        self._last_action = action

        target_policy = DEFAULT_JOINT_POS + ACTION_SCALE * action
        target_robot = np.zeros(self._num_robot_motors, dtype=np.float32)
        target_robot[POLICY_TO_ROBOT[self._mapped]] = target_policy[self._mapped]
        return target_robot, action


def build_default_pose() -> np.ndarray:
    """Trained default joint pose in ``rt/lowstate`` motor order [rad], shape ``(29,)``."""
    mapped = POLICY_TO_ROBOT >= 0
    pose = np.zeros(NUM_ROBOT_MOTORS, dtype=np.float32)
    pose[POLICY_TO_ROBOT[mapped]] = DEFAULT_JOINT_POS[mapped]
    return pose


def load_policy(path: str):
    """Load a TorchScript policy and wrap it as a numpy callable.

    Raises:
        ValueError: If the checkpoint's input or output width does not match the deployment contract,
            which almost always means the wrong run or a policy that still expects ``base_lin_vel``.
    """
    import torch

    module = torch.jit.load(path)
    module.eval()
    probe = module(torch.zeros(1, OBS_DIM))
    if probe.shape[-1] != NUM_POLICY_JOINTS:
        raise ValueError(
            f"policy maps {OBS_DIM} observations to {probe.shape[-1]} actions, expected {NUM_POLICY_JOINTS}"
        )

    def run(obs: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            return module(torch.from_numpy(np.ascontiguousarray(obs, dtype=np.float32))).numpy()

    return run
