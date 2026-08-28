# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read-only inspection of a G1 over ``rt/lowstate``. Writes nothing, ever.

The step between "the network is up" and "something commands the robot". It answers, without the
robot moving:

* is ``rt/lowstate`` arriving, how fast, and is the robot's clock advancing
* is ``mode_machine`` what this deployment assumes (5 on a 29-DoF G1)
* is ``mode_pr`` the PR ankle convention the policy was trained against
* is the robot upright, and where are its joints *right now* -- which is how far
  :func:`~g1_deploy.hardware.ramp_to_pose` would have to move it, the one number you cannot know in
  advance because it depends on what the factory controller left behind
* **does the remote's abort combo actually decode on this firmware**

That last one is the reason to run this before anything else. The G1 has no hardware emergency stop,
so :mod:`g1_deploy.remote` is the stop -- and the remote's mapping changed at Motion Control 8.2.0.0.
Press the combo here, watch it register, and only then let something write ``rt/lowcmd``. Verifying a
stop after you need it is not verifying it.

Deliberately built on a bare :class:`ChannelSubscriber` rather than
:class:`~g1_deploy.controller.G1ControlLink`, because that class constructs an ``rt/lowcmd``
publisher. It would never write one, but "provably has no writer on the topic" is a stronger claim
than "has a writer that is not used", and this is the tool whose whole value is being harmless.

Usage::

    python scripts/check_robot.py --domain_id 0 --interface en6
"""

from __future__ import annotations

import argparse
import time

from g1_deploy.bootstrap import ensure_cyclonedds

ensure_cyclonedds()

import numpy as np  # noqa: E402

from g1_deploy import core  # noqa: E402
from g1_deploy import hardware as hw  # noqa: E402
from g1_deploy import remote as rc  # noqa: E402
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber  # noqa: E402
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_  # noqa: E402

EXPECTED_MODE_MACHINE = 5
"""29-DoF G1.

Unitree's `Basic Services Interface
<https://support.unitree.com/home/en/G1_developer/basic_services_interface>`_ says 4 = 23-DoF,
5 = 29-DoF, 6 = 27-DoF, and their own ``unitree_rl_lab`` and ``unitree_mujoco`` agree. Their `Joint
Motor Sequence <https://support.unitree.com/home/en/G1_developer/joint_motor_sequence>`_ page says
1/2/9 instead; it is stale and outvoted three to one.
"""


def diagnose(args) -> int:
    """Say *why* there is no ``rt/lowstate``, by asking DDS discovery what it can see.

    "No samples" has three very different causes and the fix is different for each:

    * nothing discovered at all -- wrong interface, wrong domain, or no link
    * the robot's participants are visible but nobody publishes ``rt/lowstate`` -- the low-level
      service is not running, which is a robot-side state question (debug mode, initialisation),
      not a networking one
    * a publisher exists and the samples still do not arrive -- then it is worth suspecting QoS or
      the ``cyclonedds`` version

    Distinguishing them from the outside costs one builtin-topic subscription, and not doing it costs
    an hour of tcpdump.
    """
    from cyclonedds.builtin import BuiltinDataReader, BuiltinTopicDcpsPublication
    from cyclonedds.domain import DomainParticipant

    print("[..] asking DDS discovery what it can see ...")
    dp = DomainParticipant(args.domain_id)
    reader = BuiltinDataReader(dp, BuiltinTopicDcpsPublication)
    writers: dict[str, str] = {}
    end = time.monotonic() + 15.0
    while time.monotonic() < end:
        for sample in reader.take(500) or []:
            writers[sample.topic_name] = sample.type_name
        time.sleep(0.2)

    if not writers:
        print("[FAIL] no DDS publishers of any kind discovered.")
        print("       -> wrong --interface or --domain_id (0 on a robot), or the link is down.")
        print("       Check: ifconfig shows an address on the robot's subnet, and ping reaches it.")
        return 1

    print(f"[ok] the robot's DDS is reachable: {len(writers)} topics have publishers")
    if "rt/lowstate" in writers:
        print("[FAIL] rt/lowstate HAS a publisher but no sample arrived. That is a QoS or library")
        print("       mismatch rather than a robot state problem -- note unitree_sdk2py pins")
        print("       cyclonedds==0.10.2 and a newer one is installed here.")
        return 1

    print("[FAIL] nothing on this robot is publishing rt/lowstate.")
    print("       This is a robot-side state problem, not a network one -- the link is fine.")
    print("       The low-level state service is not running. Check, in order:")
    print("         1. LED strip colour. Debug mode is SOLID YELLOW. Damping is orange, zero-torque")
    print("            is purple, normal operation is blue, error is red.")
    print("         2. If it is not yellow: L2+B (damping) first, then L2+R2. Unitree's docs say")
    print("            debug mode can only be entered from zero-torque or damping, and that you may")
    print("            need several presses of L2+R2. Confirm with L2+A (diagnostic pose), L2+B back.")
    print("         3. Initialisation really finished: about 1 minute after power-on until all")
    print("            joints go zero-torque, and then another 30 seconds.")
    print("       Related topics seen, as a hint about what IS running:")
    for name in sorted(writers):
        if any(k in name for k in ("low", "sportmode", "odommode", "servicestate", "wireless")):
            print(f"         {name:<26s} {writers[name]}")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--domain_id", type=int, default=0, help="0 on a real G1.")
    parser.add_argument("--interface", required=True, help="Interface on the robot's subnet, from ifconfig.")
    parser.add_argument("--seconds", type=float, default=30.0, help="How long to watch the remote.")
    parser.add_argument("--policy_physics", choices=("physx", "newton"), default="newton")
    args = parser.parse_args()

    core.set_policy_backend(args.policy_physics)
    default_pose = core.build_default_pose()

    ChannelFactoryInitialize(args.domain_id, args.interface)
    latest: dict = {}
    count = {"n": 0}

    def on_state(msg: LowState_) -> None:
        latest["msg"] = msg
        count["n"] += 1

    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(on_state, 10)

    print(f"[..] listening on rt/lowstate, domain {args.domain_id} / {args.interface} -- NOTHING is published")
    deadline = time.monotonic() + 15.0
    while "msg" not in latest:
        if time.monotonic() > deadline:
            print("[FAIL] no rt/lowstate in 15s.")
            return diagnose(args)
        time.sleep(0.05)

    # ---- rate --------------------------------------------------------------------------------
    count["n"] = 0
    t0, tick0 = time.monotonic(), int(latest["msg"].tick)
    time.sleep(2.0)
    dt = time.monotonic() - t0
    tick1 = int(latest["msg"].tick)
    print(f"[ok] rt/lowstate at {count['n'] / dt:.0f} Hz")
    if tick1 == tick0:
        print(f"[FAIL] tick is frozen at {tick0} -- the robot is publishing but its clock is not advancing")
        return 1
    print(f"[ok] tick advancing: {tick0} -> {tick1} ({(tick1 - tick0) / dt / 1000:.2f} robot-seconds per wall second)")

    msg = latest["msg"]

    # ---- identity ----------------------------------------------------------------------------
    mm, pr = int(msg.mode_machine), int(msg.mode_pr)
    ok_mm = mm == EXPECTED_MODE_MACHINE
    print(f"[{'ok' if ok_mm else '!!'}] mode_machine = {mm}" + ("" if ok_mm else f"  EXPECTED {EXPECTED_MODE_MACHINE}"))
    if not ok_mm:
        print("     4 = 23-DoF, 5 = 29-DoF, 6 = 27-DoF. This deployment is built for the 29-DoF joint order;")
        print("     anything else means the joint mapping in core.py does not describe this robot. Stop.")
    ok_pr = pr == 0
    pr_txt = "PR, matches training" if ok_pr else "AB -- NOT what the policy assumes"
    print(f"[{'ok' if ok_pr else '!!'}] mode_pr = {pr} ({pr_txt})")

    # ---- posture -----------------------------------------------------------------------------
    quat = np.asarray(msg.imu_state.quaternion[:4], dtype=np.float32)
    g_z = float(core.quat_apply_inverse_wxyz(quat, np.array([0.0, 0.0, -1.0], np.float32))[2])
    upright = g_z <= hw.UPRIGHT_GRAVITY_Z
    up_txt = "upright" if upright else "NOT upright"
    print(f"[{'ok' if upright else '!!'}] gravity_z = {g_z:+.3f} ({up_txt}; -1.0 is vertical)")

    # ---- where the ramp would have to go -----------------------------------------------------
    from g1_deploy.controller import read_joint_state

    q, dq = read_joint_state(msg, core.NUM_ROBOT_MOTORS)
    move = np.abs(default_pose - q)
    print(f"[..] the robot is {move.max():.3f} rad from the policy's start pose; largest moves a ramp would make:")
    for j in np.argsort(move)[::-1][:6]:
        print(
            f"       {j:2d} {hw.MOTOR_NAMES[j]:<20s} now {q[j]:+7.3f} -> {default_pose[j]:+7.3f} rad"
            f"   ({move[j] / 3.0:5.2f} rad/s over a 3 s ramp)"
        )
    if float(np.abs(dq).max()) > 0.05:
        print(f"[!!] joints are moving ({np.abs(dq).max():.3f} rad/s max) -- something is still driving the robot")
    else:
        print(f"[ok] joints are still (max |dq| {np.abs(dq).max():.4f} rad/s)")

    # ---- who owns rt/lowcmd --------------------------------------------------------------------
    from g1_deploy import hardware as hw2

    n_cmd = hw2.lowcmd_traffic(3.0)
    if n_cmd:
        print(f"[!!] {n_cmd} rt/lowcmd samples in 3s -- something else is commanding the robot.")
        print("     --real will try to release it; do not run anything until this reads 0.")
    else:
        print("[ok] rt/lowcmd is free: nothing else is commanding the robot")
        print("     (the motion switcher may still report an old mode name here -- measured, that")
        print("      happens in debug mode and the topic, not the name, is what matters)")

    # ---- the stop ----------------------------------------------------------------------------
    if args.seconds <= 0:
        print("\n[..] --seconds 0: skipping the remote check. Run this again with the remote in hand")
        print("     before anything writes rt/lowcmd -- an unverified stop is not a stop.")
        print("\n[ok] read-only check complete; this process never published rt/lowcmd")
        return 0

    raw = bytes(bytearray(msg.wireless_remote))
    if not any(raw):
        print("[!!] wireless_remote is all zeros. Either no key is pressed (normal), or this firmware")
        print("     does not populate it -- the presses below will tell you which.")
    print()
    print(f"[..] PRESS THE ABORT COMBO NOW -- L2+B (firmware >= 8.2.0.0) or L1+A (older). {args.seconds:.0f}s.")
    print("     Nothing here commands the robot; this only proves the stop decodes on your firmware.")
    print("     Also try L2+A and L2+R2 -- those must NOT register as an abort.")
    print("     Then push the sticks: check that LEFT-FORWARD gives vx>0 and RIGHT-LEFT turns left.")
    print("     If a sign is inverted on your remote, fix it in teleop.apply_sticks before --remote.")

    seen_abort: set[str] = set()
    seen_any: set[str] = set()
    seen_sticks: set[bool] = set()
    last = ""
    end = time.monotonic() + args.seconds
    while time.monotonic() < end:
        m = latest["msg"]
        buttons = rc.parse_buttons(m.wireless_remote)
        held = sorted(n for n, v in buttons.items() if v)
        combo = rc.abort_pressed(m.wireless_remote)
        st = rc.parse_sticks(m.wireless_remote)
        cmd = ""
        if st and any(st.values()):
            vx = max(0.0, min(1.0, st.get("ly", 0.0)))
            vy = max(-0.5, min(0.5, -st.get("lx", 0.0)))
            cmd = (
                f"  |  ly{st.get('ly', 0.0):+.2f} lx{st.get('lx', 0.0):+.2f} rx{st.get('rx', 0.0):+.2f}"
                f"  ->  vx{vx:+.2f} vy{vy:+.2f} turn{-st.get('rx', 0.0):+.2f}"
            )
        line = f"       held: {'+'.join(held) or '(none)':<20s}  abort: {combo or '-':<7s}{cmd}"
        if line != last:
            print(line + ("   <== ABORT WOULD FIRE" if combo else ""))
            last = line
        seen_any.update(held)
        if st and any(st.values()):
            seen_sticks.add(True)
        if combo:
            seen_abort.add(combo)
        time.sleep(0.05)

    print()
    if seen_abort:
        print(f"[ok] abort decoded: {', '.join(sorted(seen_abort))} -- the stop works on this firmware")
    elif seen_any:
        print(f"[!!] buttons decode ({', '.join(sorted(seen_any))}) but no abort combo was seen.")
        print("     Do not run the policy until you have pressed one and watched it register here.")
    else:
        print("[FAIL] no button press decoded at all. Either nothing was pressed, or this firmware does")
        print("       not put the remote in rt/lowstate -- in which case g1_deploy.remote cannot stop the")
        print("       robot and you need a different abort before running anything.")
    if seen_sticks:
        print("[ok] analog sticks decode -- --remote has something to read")
    else:
        print("[..] no stick deflection seen; push them if you intend to use --remote")
    print("\n[ok] read-only check complete; this process never published rt/lowcmd")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
