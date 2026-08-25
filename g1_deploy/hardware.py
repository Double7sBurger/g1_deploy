# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bring a physical G1 from power-on to policy control, and put it down safely afterwards.

Everything here is hardware-only. In simulation none of it is needed -- there is no factory
controller to displace, the robot starts in the pose you asked for, and nothing breaks if a target
is wrong -- which is exactly why it is easy to leave out and find out the hard way.

The order matters and each step exists for a reason:

1. :func:`release_motion_mode` -- the G1 boots running Unitree's own motion controller, and that
   controller is itself writing ``rt/lowcmd``. Two writers on one topic means the motors execute
   whichever sample landed last, so your policy and the factory controller alternate. Release first,
   and poll until the release is confirmed, because the request is asynchronous.
2. Read ``mode_machine`` off ``rt/lowstate`` -- the robot rejects commands that do not echo it.
3. :func:`ramp_to_pose` -- interpolate from wherever the robot currently is to the pose the policy
   expects. Handing control straight to a network from an arbitrary pose is a step in the position
   error and a torque spike.
4. :func:`check_upright` -- if the ramp did not end with the robot standing, do not engage.
5. :func:`damp_down` -- on exit, on a fall, and on Ctrl-C. Zero *gain*, nominal damping. Cutting
   torque outright drops the robot; damping lets it settle.

.. attention::
    :func:`clamp_to_joint_limits` is available but **off by default, and that is deliberate**. On a
    recorded rollout of this task's checkpoint the ankle targets ran up to 1.3 rad outside the
    mechanical travel on 58% of control steps -- yet every *achieved* ankle angle stayed inside it
    (pitch reached -0.625 rad against a -0.873 limit). The policy is using a far-away target as a
    torque command on a soft joint: ankle kp is 20 N·m/rad, so a target 0.9 rad away is simply how it
    asks for 18 N·m. Clamping that target to the limit cuts the same command to about 4 N·m.

    So clamping would trade a 4x weaker ankle for protection against a hard-stop impact the data says
    does not occur. Measure first: the loop reports how many steps *would* have been clamped, and you
    turn it on only if the robot actually reaches its stops.
"""

from __future__ import annotations

import time

import numpy as np

JOINT_POS_LIMITS = np.array(
    [
        # Mechanical travel [rad] in G1JointIndex order, from the official g1_29dof MJCF's joint ranges,
        # which match Unitree's current URDF exactly.
        (-2.530700, +2.879800),  # 0  left_hip_pitch
        (-0.523600, +2.967100),  # 1  left_hip_roll
        (-2.757600, +2.757600),  # 2  left_hip_yaw
        (-0.087267, +2.879800),  # 3  left_knee
        (-0.872670, +0.523600),  # 4  left_ankle_pitch
        (-0.261800, +0.261800),  # 5  left_ankle_roll
        (-2.530700, +2.879800),  # 6  right_hip_pitch
        (-2.967100, +0.523600),  # 7  right_hip_roll
        (-2.757600, +2.757600),  # 8  right_hip_yaw
        (-0.087267, +2.879800),  # 9  right_knee
        (-0.872670, +0.523600),  # 10 right_ankle_pitch
        (-0.261800, +0.261800),  # 11 right_ankle_roll
        (-2.618000, +2.618000),  # 12 waist_yaw
        (-0.520000, +0.520000),  # 13 waist_roll
        (-0.520000, +0.520000),  # 14 waist_pitch
        (-3.089200, +2.670400),  # 15 left_shoulder_pitch
        (-1.588200, +2.251500),  # 16 left_shoulder_roll
        (-2.618000, +2.618000),  # 17 left_shoulder_yaw
        (-1.047200, +2.094400),  # 18 left_elbow
        (-1.972220, +1.972220),  # 19 left_wrist_roll
        (-1.614430, +1.614430),  # 20 left_wrist_pitch
        (-1.614430, +1.614430),  # 21 left_wrist_yaw
        (-3.089200, +2.670400),  # 22 right_shoulder_pitch
        (-2.251500, +1.588200),  # 23 right_shoulder_roll
        (-2.618000, +2.618000),  # 24 right_shoulder_yaw
        (-1.047200, +2.094400),  # 25 right_elbow
        (-1.972220, +1.972220),  # 26 right_wrist_roll
        (-1.614430, +1.614430),  # 27 right_wrist_pitch
        (-1.614430, +1.614430),  # 28 right_wrist_yaw
    ]
)
"""``(lower, upper)`` per motor [rad], shape ``(29, 2)``."""

UPRIGHT_GRAVITY_Z = -0.9
"""Projected-gravity z at or below which the robot counts as standing; -1.0 is perfectly upright."""


def clamp_to_joint_limits(target: np.ndarray, margin: float = 0.05) -> tuple[np.ndarray, int]:
    """Clamp joint targets into the mechanical travel, keeping a margin off the hard stops.

    Args:
        target: Position targets in motor order [rad], shape ``(29,)``.
        margin: Keep this far inside each limit [rad], so the drive never parks on a hard stop.

    Returns:
        ``(clamped, n_clamped)`` -- the safe target and how many joints had to be moved. A nonzero
        count on hardware is worth printing: it means the policy asked for travel the robot does not
        have.
    """
    lower = JOINT_POS_LIMITS[:, 0] + margin
    upper = JOINT_POS_LIMITS[:, 1] - margin
    clamped = np.clip(target, lower, upper)
    return clamped.astype(np.float32), int(np.count_nonzero(np.abs(clamped - target) > 1e-6))


def release_motion_mode(timeout: float = 20.0) -> None:
    """Take low-level control away from the G1's factory motion controller.

    Polls until the robot reports no active mode. ``ReleaseMode`` is an asynchronous request, so
    calling it once and moving on can leave the factory controller still publishing ``rt/lowcmd``
    alongside yours.

    Args:
        timeout: Give up after this long [s].

    Raises:
        TimeoutError: If a mode is still active at the end. Do not send ``rt/lowcmd`` in that case.
    """
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

    client = MotionSwitcherClient()
    client.SetTimeout(5.0)
    client.Init()

    deadline = time.monotonic() + timeout
    while True:
        status, result = client.CheckMode()
        if result is None:
            raise TimeoutError(f"motion switcher did not answer CheckMode (status {status})")
        active = result.get("name") or ""
        if not active:
            print("[ok] factory motion mode released; rt/lowcmd is ours")
            return
        if time.monotonic() > deadline:
            raise TimeoutError(f"factory mode {active!r} still active after {timeout:.0f}s")
        print(f"[..] releasing factory mode {active!r}")
        client.ReleaseMode()
        time.sleep(1.0)


def ramp_to_pose(
    link, target: np.ndarray, kp: np.ndarray, kd: np.ndarray, mode_machine: int, control_dt: float, seconds: float = 3.0
) -> None:
    """Interpolate from the robot's current joint positions to ``target``.

    The *target* is interpolated at full stiffness the whole way. Ramping the gains instead is the
    obvious alternative and it is wrong: it leaves the robot with almost no stiffness for the first
    fraction of a second, which is long enough to collapse before the ramp has gone anywhere.

    Args:
        link: A :class:`~dds_controller.G1ControlLink`.
        target: Destination joint positions in motor order [rad], shape ``(29,)``.
        kp: Gains to hold throughout [N·m/rad].
        kd: Gains to hold throughout [N·m·s/rad].
        mode_machine: Echoed from ``rt/lowstate``.
        control_dt: Loop period [s].
        seconds: Ramp duration.
    """
    from g1_deploy.controller import read_joint_state

    start, _ = read_joint_state(link.wait_for_state(), len(target))
    steps = max(1, int(round(seconds / control_dt)))
    print(f"[..] ramping to the start pose over {seconds:.1f}s  (max move {np.abs(target - start).max():.3f} rad)")
    for k in range(steps):
        alpha = (k + 1) / steps
        blended, _ = clamp_to_joint_limits(start * (1.0 - alpha) + target * alpha)
        link.send(blended, kp, kd, mode_machine)
        time.sleep(control_dt)


def gravity_z(link) -> float:
    """Projected gravity z from the latest ``rt/lowstate``; -1.0 is upright, 0 is on its side."""
    from g1_deploy.core import quat_apply_inverse_wxyz

    quat = np.asarray(link.wait_for_state().imu_state.quaternion[:4], dtype=np.float32)
    return float(quat_apply_inverse_wxyz(quat, np.array([0.0, 0.0, -1.0], dtype=np.float32))[2])


def check_upright(link) -> float:
    """Return projected gravity z, raising if the robot is not standing.

    Raises:
        RuntimeError: If the robot is not upright. Engaging a locomotion policy from a heap on the
            floor feeds it observations it was never trained on.
    """
    value = gravity_z(link)
    if value > UPRIGHT_GRAVITY_Z:
        raise RuntimeError(f"robot is not upright (gravity_z {value:+.3f}, need <= {UPRIGHT_GRAVITY_Z})")
    print(f"[ok] robot is upright (gravity_z {value:+.3f})")
    return value


def damp_down(link, kd: np.ndarray, mode_machine: int, control_dt: float, seconds: float = 1.5) -> None:
    """Bleed off energy with zero stiffness and nominal damping, then stop commanding.

    This is the only correct way to end a run. Cutting the command outright leaves the motors
    unpowered and the robot falls; holding position keeps fighting whatever put it in trouble.

    Args:
        link: A :class:`~dds_controller.G1ControlLink`.
        kd: Damping gains to hold [N·m·s/rad].
        mode_machine: Echoed from ``rt/lowstate``.
        control_dt: Loop period [s].
        seconds: How long to damp.
    """
    print("[..] damping down")
    zero_target = np.zeros(len(kd), dtype=np.float32)
    zero_kp = np.zeros(len(kd), dtype=np.float32)
    for _ in range(max(1, int(round(seconds / control_dt)))):
        link.send(zero_target, zero_kp, kd, mode_machine)
        time.sleep(control_dt)


def confirm(prompt: str) -> None:
    """Block until the operator types ``go``.

    .. attention::
        Call this only *before* the robot is under your command. Nothing sends ``rt/lowcmd`` while
        this is waiting, and on hardware the motor watchdog times out and drops the robot; in
        simulation it falls on its own within about 1.5 s. A prompt between the ramp and engaging the
        policy is therefore a way to break the robot, not a safety feature.

    Raises:
        RuntimeError: If anything else is typed, so a stray keypress cannot start the robot.
    """
    answer = input(f"\n>>> {prompt}\n>>> type 'go' and press Enter (anything else aborts): ").strip()
    if answer != "go":
        raise RuntimeError("aborted by operator")
