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
5. :func:`damp_down` -- on exit, on a fall, on Ctrl-C, and on :func:`check_abort`. Zero *gain*,
   nominal damping. Cutting torque outright drops the robot; damping lets it settle.

The G1 has **no hardware emergency stop**, so :func:`check_abort` is the stop: every loop that
commands the robot reads the remote out of ``rt/lowstate`` and bails on the abort combo. See
:mod:`g1_deploy.remote` for why this has to live in our own program rather than rely on the factory
controller still listening.

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

MOTOR_NAMES = (
    # ``G1JointIndex`` order, the names the official g1_29dof MJCF and URDF use.
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
    "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
    "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
)
"""Motor names in ``rt/lowstate`` order, for readable bring-up output."""

RAMP_SPEED_WARN = 0.75
"""Average ramp rate [rad/s] above which :func:`ramp_to_pose` warns.

Not a limit -- there is no safe universal one, because how fast is too fast depends on what the
robot is hanging from. It is a prompt to look: at 0.75 rad/s a joint crosses 43 deg per second, and
anything much past that on a 35 kg humanoid is worth deciding on deliberately rather than by
inheriting the 3 s default.
"""

UPRIGHT_GRAVITY_Z = -0.9
"""Projected-gravity z at or below which the robot counts as standing; -1.0 is perfectly upright."""


class OperatorAbort(RuntimeError):
    """Raised when the remote's abort combo is seen. Every caller must damp down."""


def check_abort(link) -> None:
    """Raise :class:`OperatorAbort` if the operator is holding an abort combo on the remote.

    Called from every loop that commands the robot -- the ramp, the hold and the policy loop -- so
    the stop works during the motion that most needs it rather than only once the policy is running.

    Args:
        link: A :class:`~g1_deploy.controller.G1ControlLink`.

    Raises:
        OperatorAbort: If :func:`~g1_deploy.remote.abort_pressed` matches.
    """
    from g1_deploy.remote import abort_pressed

    state = link.state
    if state is None:
        return
    combo = abort_pressed(state.wireless_remote)
    if combo is not None:
        raise OperatorAbort(f"operator pressed {combo} on the remote")


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


def lowcmd_traffic(seconds: float = 3.0) -> int:
    """Count ``rt/lowcmd`` samples from *anyone else* over ``seconds``. Read-only.

    The property that actually matters before commanding a robot is "is another writer on this
    topic", and this measures it directly instead of inferring it from a mode name.

    That distinction is not academic. On a G1 in debug mode the motion switcher keeps reporting the
    previously selected mode -- measured: ``CheckMode() -> {'form': '0', 'name': 'ai'}`` -- while
    ``rt/lowcmd`` is provably silent, because debug mode suspends the high-level service without
    deselecting it. Believing the name means :func:`release_motion_mode` spins until it times out and
    the run dies before it starts; believing the topic means you proceed, correctly.

    Args:
        seconds: Listening window.

    Returns:
        Samples observed. Zero means the topic is free.
    """
    from unitree_sdk2py.core.channel import ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_

    count = {"n": 0}
    sub = ChannelSubscriber("rt/lowcmd", LowCmd_)
    sub.Init(lambda _msg: count.__setitem__("n", count["n"] + 1), 10)
    time.sleep(seconds)
    return count["n"]


def take_lowcmd(probe_s: float = 3.0, timeout: float = 20.0) -> None:
    """Make sure nothing else is writing ``rt/lowcmd``, releasing the factory controller if needed.

    Evidence first, mode registry second:

    1. Listen. If ``rt/lowcmd`` is silent, there is nothing to release -- which is the normal case
       once the operator has put the robot in debug mode with the remote, and is a *better* state to
       start from than a released one, because what the joints do after ``ReleaseMode()`` is not
       documented anywhere while debug mode's damping is.
    2. Only if somebody is writing, ask the motion switcher to release, then listen again.

    Args:
        probe_s: How long to listen each time.
        timeout: Give up on releasing after this long [s].

    Raises:
        RuntimeError: If another writer is still there after the release. Do not command the robot.
    """
    print(f"[..] listening {probe_s:.0f}s for another writer on rt/lowcmd ...")
    n = lowcmd_traffic(probe_s)
    if n == 0:
        print("[ok] rt/lowcmd is free -- nothing else is commanding the robot")
        return
    print(f"[!!] {n} rt/lowcmd samples from another writer; releasing the factory controller")
    release_motion_mode(timeout)
    n = lowcmd_traffic(probe_s)
    if n:
        raise RuntimeError(
            f"still {n} rt/lowcmd samples from another writer after the release -- refusing to add"
            " a second writer to the topic"
        )
    print("[ok] rt/lowcmd is free")


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
    move = np.abs(target - start)
    steps = max(1, int(round(seconds / control_dt)))
    print(f"[..] ramping to the start pose over {seconds:.1f}s  (max move {move.max():.3f} rad)")
    for j in np.argsort(move)[::-1][:3]:
        if move[j] > 1e-3:
            print(
                f"       {j:2d} {MOTOR_NAMES[j]:<20s} {start[j]:+7.3f} -> {target[j]:+7.3f} rad"
                f"  ({move[j] / seconds:5.2f} rad/s)"
            )
    # The robot starts wherever the factory controller left it, which on hardware is not knowable in
    # advance -- a G1 parked sitting has to unfold a long way to reach this crouch. The ramp is a
    # constant-rate interpolation, so a long move at a short duration is a fast move, and the whole
    # point of ramping was to avoid handing the drives a step.
    if move.max() / seconds > RAMP_SPEED_WARN:
        print(
            f"[warn] the ramp averages {move.max() / seconds:.2f} rad/s on the worst joint, above"
            f" {RAMP_SPEED_WARN:.2f}. The robot is far from the start pose -- consider a longer"
            " --ramp_s, and be sure it is supported."
        )
    for k in range(steps):
        check_abort(link)
        alpha = (k + 1) / steps
        blended, _ = clamp_to_joint_limits(start * (1.0 - alpha) + target * alpha)
        link.send(blended, kp, kd, mode_machine)
        time.sleep(control_dt)


def hold_pose(
    link, target: np.ndarray, kp: np.ndarray, kd: np.ndarray, mode_machine: int, control_dt: float,
    seconds: float = 5.0
) -> np.ndarray:
    """Hold ``target`` for ``seconds`` and return the joint positions actually reached.

    The bring-up equivalent of :func:`ramp_to_pose`'s destination: once the ramp is done, keep
    commanding the same pose so the operator has time to look at the robot, then report where the
    joints ended up.

    Holding a fixed pose is not the same problem as standing. A hoisted robot has nothing to
    balance, so the shipped gains hold this pose indefinitely -- the 1.5 s collapse in the README is
    about a robot on its own feet, where ankle kp of 20 N·m/rad cannot keep the centre of mass over
    the support polygon. Never run this with the feet loaded.

    Args:
        link: A :class:`~g1_deploy.controller.G1ControlLink`.
        target: Joint positions to hold in motor order [rad], shape ``(29,)``.
        kp: Gains [N·m/rad].
        kd: Gains [N·m·s/rad].
        mode_machine: Echoed from ``rt/lowstate``.
        control_dt: Loop period [s].
        seconds: How long to hold.

    Returns:
        Measured joint positions at the end of the hold [rad], shape ``(29,)``.
    """
    from g1_deploy.controller import read_joint_state

    print(f"[..] holding the start pose for {seconds:.1f}s -- look at the robot now")
    for _ in range(max(1, int(round(seconds / control_dt)))):
        check_abort(link)
        link.send(target, kp, kd, mode_machine)
        time.sleep(control_dt)
    reached, _ = read_joint_state(link.wait_for_state(), len(target))
    return reached


def hold_until_confirmed(
    link, target: np.ndarray, kp: np.ndarray, kd: np.ndarray, mode_machine: int, control_dt: float,
    word: str = "go2", timeout: float = 300.0, countdown: float = 3.0, stream=None
) -> np.ndarray:
    """Hold ``target`` and keep publishing while the operator repositions the robot.

    The gap this closes: between the ramp and engaging the policy there was nowhere to stand the
    robot up, square its feet, or lower it onto the ground, because the only tool for pausing was
    :func:`confirm` and that blocks. **Nothing sends ``rt/lowcmd`` while ``input()`` waits**, so on
    hardware the motor watchdog times out and drops the robot -- which is why the original sequence
    deliberately ran ramp, check and engage back to back.

    So this pauses without stopping: the hold target goes out at the full control rate throughout,
    stdin is polled rather than read, and the abort combo is checked every step. It is Unitree's own
    pattern -- ``unitree_rl_gym`` holds the default pose and waits on a button while publishing --
    and it is what makes lowering the hoist onto the feet a step you can take your time over.

    Args:
        link: A :class:`~g1_deploy.controller.G1ControlLink`.
        target: Pose to hold in motor order [rad], shape ``(29,)``.
        kp: Gains [N·m/rad].
        kd: Gains [N·m·s/rad].
        mode_machine: Echoed from ``rt/lowstate``.
        control_dt: Loop period [s].
        word: What the operator types to proceed. Anything else is ignored rather than treated as
            consent, so a stray keypress cannot start the robot.
        timeout: Give up holding after this long [s]. Not optional -- a hold with no end is a robot
            cooking its own motors while nobody is watching.
        countdown: Seconds between the confirmation and the policy taking over, still publishing.
        stream: Input stream; ``sys.stdin`` by default.

    Returns:
        Measured joint positions at the end of the hold [rad], shape ``(29,)``.

    Raises:
        OperatorAbort: If the remote's abort combo is pressed.
        TimeoutError: If ``word`` never arrives.
    """
    import select
    import sys

    from g1_deploy.controller import read_joint_state

    handle = sys.stdin if stream is None else stream
    interactive = False
    try:
        interactive = handle.isatty()
    except (AttributeError, ValueError):
        interactive = False

    print(f"\n>>> HOLDING the start pose. Square the robot up, lower it onto its feet.")
    print(f">>> Type '{word}' and press Enter when ready; the policy engages {countdown:.0f}s later.")
    print(f">>> rt/lowcmd keeps flowing throughout, so there is no watchdog to race.")
    if not interactive:
        print("[warn] stdin is not a terminal; holding for the full timeout instead")

    deadline = time.monotonic() + timeout
    pending, confirmed_at = "", None
    while True:
        check_abort(link)
        link.send(target, kp, kd, mode_machine)

        if confirmed_at is not None:
            if time.monotonic() - confirmed_at >= countdown:
                break
        elif interactive and select.select([handle], [], [], 0)[0]:
            chunk = handle.readline()
            if not chunk:
                interactive = False
            else:
                pending += chunk
                line, pending = pending.strip(), ""
                if line == word:
                    confirmed_at = time.monotonic()
                    print(f"[ok] confirmed; engaging in {countdown:.0f}s -- hands clear")
                elif line:
                    print(f"[..] ignored {line!r}; type {word!r} to proceed")
        elif time.monotonic() > deadline:
            raise TimeoutError(f"{word!r} not received within {timeout:.0f}s of holding")

        time.sleep(control_dt)

    reached, _ = read_joint_state(link.wait_for_state(), len(target))
    return reached


def report_tracking(target: np.ndarray, reached: np.ndarray, joint_names: list[str], worst: int = 6) -> float:
    """Print how far each joint ended from its target, worst first.

    What this proves: every motor answers, the gains hold the pose, and nothing is parked against a
    stop. A dead motor, a miswired joint or a limit collision all show up as one large row.

    .. attention::
        What it does **not** prove is the policy's joint ordering. ``--policy_physics`` selects the
        order of the policy's own observation and action vectors;
        :func:`~g1_deploy.core.build_default_pose` is assembled by joint *name*, so it comes out
        byte-identical under ``physx`` and ``newton`` and the ramp target is the same either way.
        A wrong backend is invisible here and only bites once the policy is running. The thing that
        actually validates it is the MuJoCo benchmark -- a scrambled action order does not walk.

        Nor does a small error mean the pose is *correct*, only that it is the pose that was asked
        for. Looking at the robot is not an optional extra step.

    Args:
        target: Commanded positions [rad], shape ``(29,)``.
        reached: Measured positions [rad], shape ``(29,)``.
        joint_names: Motor names in the same order.
        worst: How many of the largest errors to print.

    Returns:
        The largest absolute error [rad].
    """
    error = np.abs(reached - target)
    order = np.argsort(error)[::-1][:worst]
    print(f"[..] joint tracking, worst {worst} of {len(error)}:")
    for i in order:
        print(
            f"       {i:2d} {joint_names[i]:<24s} target {target[i]:+7.3f}  reached {reached[i]:+7.3f}"
            f"  error {error[i]:+7.3f} rad ({np.rad2deg(error[i]):+6.1f} deg)"
        )
    print(f"[..] max error {error.max():.3f} rad, mean {error.mean():.3f} rad")
    return float(error.max())


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
