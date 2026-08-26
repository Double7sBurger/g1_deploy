# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run an Isaac Lab G1 policy over DDS, against MuJoCo or against the robot.

The control-loop half of the Sonic-style stack (see :mod:`dds_sim_env` for the simulator half),
modelled on ``decoupled_wbc/control/main/teleop/run_g1_control_loop.py``. The important thing copied
from Sonic is their ``sim_sync_mode``: when this process owns the simulator, the *controller* decides
when physics advances, exactly ``decimation`` steps per control period. DDS still carries every byte
-- the same ``rt/lowcmd`` and ``rt/lowstate`` a real G1 speaks -- but nothing races.

Three modes, same code path for the policy:

``--sim sync``
    Own a :class:`~dds_sim_env.G1SimEnv` in this process and drive it. Deterministic, and the only
    mode where a sim-to-sim number means anything, because a free-running simulator adds 12-16 ms of
    transport jitter that has nothing to do with the policy.
``--sim none``
    Attach to whatever is already publishing ``rt/lowstate``: :mod:`run_sim_loop` running free on
    wall-clock time, ``unitree_mujoco``, or a physical G1. This is the mode that measures whether the
    policy survives real transport delay; it is not the mode to compare against Isaac Lab with.

Usage::

    uv run --no-project python deploy/run_policy_loop.py \\
        --policy logs/.../deploy_2999/policy.pt --policy_physics newton --vx 0.5 --viz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from g1_deploy.bootstrap import ensure_cyclonedds

ensure_cyclonedds()

from g1_deploy import core
from g1_deploy.core import build_default_pose, load_policy
from g1_deploy import hardware as hw
import numpy as np  # noqa: E402
from g1_deploy.controller import G1ControlLink, read_joint_state
from unitree_sdk2py.core.channel import ChannelFactoryInitialize  # noqa: E402

# dds_sim_bridge and dds_sim_env are imported lazily, inside the branch that needs them, because they
# pull in MuJoCo. A robot does not need a simulator: with --sim none this process runs on numpy,
# torch and unitree_sdk2py alone, which is what makes it deployable on a laptop or on the G1's own
# computer instead of a workstation.
SIM_DT = 0.002
"""Mirror of :data:`dds_sim_env.SIM_DT`, duplicated so the CLI can default without importing MuJoCo.

:func:`main` asserts the two agree whenever the simulator is actually loaded.
"""

DEFAULT_XML = Path.home() / "workspace/unitree_mujoco/unitree_robots/g1/scene_29dof.xml"
"""Flat-ground 29-DoF G1 scene. Not ``scene.xml``, which terrain_tool rewrote into an obstacle course."""

TRUNK_BODIES = ("pelvis", "waist_yaw_link", "waist_roll_link", "torso_link")
"""Bodies making up the trunk, the one segment where the MJCF and the USD disagree materially.

Measured at the same default pose: the MJCF trunk is 13.702 kg against the USD's 10.381 kg (+32%),
while the legs agree to -4% and the arms to +3%. ``--trunk_mass_scale`` exists so that gap can be
closed for system identification without touching the asset.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--policy", required=True, help="TorchScript policy exported by isaaclab play.")
    parser.add_argument(
        "--policy_physics",
        choices=("physx", "newton", "g1_29dof"),
        default="newton",
        help="Backend the checkpoint was trained with; PhysX and Newton enumerate the same USD's"
        " joints in different orders and the wrong choice silently drives the wrong motors.",
    )
    parser.add_argument("--sim", choices=("sync", "none"), default="sync")
    parser.add_argument("--xml", default=str(DEFAULT_XML))
    parser.add_argument("--vx", type=float, default=0.5, help="Forward velocity command [m/s].")
    parser.add_argument("--vy", type=float, default=0.0, help="Left velocity command [m/s].")
    parser.add_argument(
        "--heading",
        type=float,
        default=0.0,
        help="Target heading in the world frame [rad]. The yaw-rate command is derived from the"
        " error against it, which is how training generated that element.",
    )
    parser.add_argument("--duration", type=float, default=15.0, help="Seconds of policy control.")
    parser.add_argument("--viz", action="store_true", help="Open a viewer (--sim sync/free only).")
    parser.add_argument("--domain_id", type=int, default=1)
    parser.add_argument("--interface", default="lo")
    parser.add_argument(
        "--sim_dt",
        type=float,
        default=SIM_DT,
        help="Physics timestep [s]. Isaac Lab uses 0.005; MuJoCo's contact solver needs the solref"
        " time constant to stay above about 2x this, so the two knobs are not independent.",
    )
    parser.add_argument(
        "--contact_timeconst",
        type=float,
        default=0.005,
        help="Contact solref time constant [s]; 0 keeps the model's own. A sweep over 0.002-0.020"
        " moved walking survival by at most 13 points and left standing at 0%%, so this is a"
        " second-order knob, not the transfer gap.",
    )
    parser.add_argument(
        "--trunk_mass_scale",
        type=float,
        default=1.0,
        help="Multiply pelvis/waist/torso mass and inertia. 0.758 matches the USD the policy trained"
        " against; 1.0 keeps the MJCF, which is the one that matches a real ~35 kg G1.",
    )
    parser.add_argument(
        "--integrator",
        choices=("euler", "rk4", "implicit", "implicitfast"),
        default=None,
        help="Override the model's integrator. Only useful with a larger --sim_dt; the shipped Euler"
        " is stable at the shipped 0.002 s and is not at 0.005 s.",
    )
    parser.add_argument("--fall_gravity_z", type=float, default=-0.7, help="Stop above this.")
    parser.add_argument(
        "--real",
        action="store_true",
        help="Physical robot. Releases the factory motion controller, ramps from the robot's current"
        " pose to the policy's start pose, refuses to engage unless it ends upright, clamps every"
        " target into the mechanical travel, and damps down on exit. Implies --sim none.",
    )
    parser.add_argument("--ramp_s", type=float, default=3.0, help="Ramp duration on hardware [s].")
    parser.add_argument(
        "--clamp_targets",
        action="store_true",
        help="Clamp joint targets into the mechanical travel. Off by default: the policy drives the"
        " soft ankles with deliberately far-away targets, and clamping them cuts ankle torque about"
        " fourfold. The run reports how many steps would have been clamped either way -- turn this on"
        " only if the robot actually reaches its stops.",
    )
    parser.add_argument(
        "--skip_release_mode",
        action="store_true",
        help="Skip the factory-controller release. Only for rehearsing the --real sequence against"
        " run_sim_loop.py, which has no motion-switcher service. Never pass this to a robot: your"
        " commands would compete with the factory controller for rt/lowcmd.",
    )
    parser.add_argument(
        "--action_limit",
        type=float,
        default=None,
        help="Symmetric clamp on the raw policy output. Training clips nothing, so any value here is"
        " a change of behaviour -- on a recorded rollout of this task's checkpoint the 95th"
        " percentile was 7.25 and the maximum 11.39. Joint targets are clamped to the mechanical"
        " travel regardless when --real is set.",
    )
    args = parser.parse_args()
    if args.real:
        args.sim = "none"

    core.set_policy_backend(args.policy_physics)
    policy = load_policy(args.policy)
    default_pose = build_default_pose()
    kp, kd = core.control_gains()

    ChannelFactoryInitialize(args.domain_id, args.interface)

    bridge = env = None
    if args.sim != "none":
        from dds_sim_bridge import G1SimBridge
        from dds_sim_env import SIM_DT as ENV_SIM_DT
        from dds_sim_env import G1SimEnv

        if SIM_DT != ENV_SIM_DT:
            raise ValueError(f"SIM_DT mirror is stale: {SIM_DT} here vs {ENV_SIM_DT} in dds_sim_env")
        bridge = G1SimBridge(num_motors=core.NUM_ROBOT_MOTORS)
        mass_scale = {b: args.trunk_mass_scale for b in TRUNK_BODIES} if args.trunk_mass_scale != 1.0 else None
        env = G1SimEnv(
            args.xml,
            bridge,
            sim_dt=args.sim_dt,
            decimation=int(round(core.CONTROL_DT / args.sim_dt)),
            onscreen=args.viz,
            contact_timeconst=args.contact_timeconst,
            body_mass_scale=mass_scale,
            integrator=args.integrator,
        )
        if env.joint_names != [
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
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]:
            raise ValueError(f"{args.xml} does not list joints in G1JointIndex order: {env.joint_names}")
        # Reset before any endpoint exists, so every sample ever published is post-reset. Publishing
        # the model's own qpos0 during discovery and only then resetting let the controller seed its
        # five-frame observation history from a pose the robot was never in.
        root_z = env.reset(default_pose)
    else:
        root_z = float("nan")

    if args.real:
        # Before any endpoint of ours writes rt/lowcmd, the factory controller has to stop writing it.
        if args.skip_release_mode:
            print("[warn] --skip_release_mode: NOT releasing the factory controller (rehearsal only)")
        else:
            hw.release_motion_mode()

    link = G1ControlLink(num_motors=core.NUM_ROBOT_MOTORS)
    if args.sim == "sync":
        mode_machine = await_discovery(link, bridge, env, default_pose, kp, kd)
    else:
        mode_machine = int(link.wait_for_state().mode_machine)

    if args.real:
        # The one and only operator gate, and it comes before anything moves. There must be no
        # blocking prompt between the ramp and engaging the policy: nothing sends rt/lowcmd while
        # input() waits, so on hardware the motor watchdog times out and drops the robot, and in
        # simulation it simply falls -- it cannot hold this pose for much over 1.5 s. Ramp, check and
        # engage therefore run back to back.
        hw.confirm(
            "Robot supported / on a stand, area clear, e-stop in hand.\n"
            f">>> This ramps over {args.ramp_s:.1f}s and then runs the policy IMMEDIATELY for"
            f" {args.duration:.0f}s at vx={args.vx:+.2f} vy={args.vy:+.2f}."
        )
        hw.ramp_to_pose(link, default_pose, kp, kd, mode_machine, core.CONTROL_DT, args.ramp_s)
        try:
            hw.check_upright(link)
        except RuntimeError as exc:
            print(f"[FAIL] {exc}")
            hw.damp_down(link, kd, mode_machine, core.CONTROL_DT)
            return 1
    runner = core.G1PolicyRunner(policy)
    runner.reset()

    physics = (
        "external simulator or robot"
        if env is None
        else f"physics {env.sim_dt * 1000:.1f} ms x {env.decimation}"
        f"  solref {env.model.geom_solref[1, 0]:.4f}  root_z {root_z:.3f} m"
    )
    print(
        f"[..] {Path(args.policy).parent.name}  mode={args.sim}  order={args.policy_physics}"
        f"  mode_machine={mode_machine}\n"
        f"     {physics}  policy at {1 / core.CONTROL_DT:.0f} Hz"
        f"  cmd=({args.vx:+.2f}, {args.vy:+.2f}) m/s, heading {np.rad2deg(args.heading):+.0f} deg"
    )

    n_steps = int(round(args.duration / core.CONTROL_DT))
    wall_start = time.monotonic()
    fell_at = None
    clamped_steps = 0
    try:
        for k in range(n_steps):
            if args.sim == "sync":
                state = link.wait_for_tick(int(round(env.data.time * 1e3)))
            else:
                state = link.state

            quat = np.asarray(state.imu_state.quaternion[:4], dtype=np.float32)
            gyro = np.asarray(state.imu_state.gyroscope[:3], dtype=np.float32)
            q, dq = read_joint_state(state, core.NUM_ROBOT_MOTORS)
            command = np.array([args.vx, args.vy, core.yaw_rate_from_heading(args.heading, quat)], dtype=np.float32)

            target, _ = runner.step(q, dq, quat, gyro, command, action_limit=args.action_limit)
            target[core.UNMAPPED_ROBOT_MOTORS] = 0.0
            if args.real:
                safe, n_clamped = hw.clamp_to_joint_limits(target)
                clamped_steps += n_clamped > 0
                if args.clamp_targets:
                    target = safe

            gravity_z = float(core.quat_apply_inverse_wxyz(quat, np.array([0.0, 0.0, -1.0], np.float32))[2])
            if gravity_z > args.fall_gravity_z:
                fell_at = k * core.CONTROL_DT
                print(f"[warn] fell at t={fell_at:.2f}s (gravity_z={gravity_z:+.3f})")
                break

            if args.sim == "sync":
                # Wait for the simulator's own subscriber to have the command before advancing, so a
                # control period never runs on the previous period's target because DDS lagged.
                seen = bridge.cmd_count
                link.send(target, kp, kd, mode_machine)
                deadline = time.monotonic() + 1.0
                while bridge.cmd_count == seen:
                    if time.monotonic() > deadline:
                        raise TimeoutError("simulator did not receive rt/lowcmd within 1s")
                    time.sleep(0.0002)
                env.step_control()
            else:
                link.send(target, kp, kd, mode_machine)
                target_wall = wall_start + (k + 1) * core.CONTROL_DT
                lag = target_wall - time.monotonic()
                if lag > 0:
                    time.sleep(lag)
    except KeyboardInterrupt:
        print("\n[..] interrupted")
    finally:
        if args.real:
            # Every exit path damps: normal end, fall, operator Ctrl-C, or an exception mid-loop.
            hw.damp_down(link, kd, mode_machine, core.CONTROL_DT)
        if env is not None:
            env.close()

    survived = fell_at if fell_at is not None else n_steps * core.CONTROL_DT
    real_time = time.monotonic() - wall_start
    print(f"[ok] survived {survived:.2f}s of {args.duration:.0f}s  (wall {real_time:.1f}s)")
    if args.real and clamped_steps:
        verb = "were clamped" if args.clamp_targets else "would have been clamped (--clamp_targets off)"
        print(
            f"[warn] {clamped_steps}/{n_steps} control steps {verb}: the policy commanded joint"
            " targets outside the mechanical travel. Expected on the ankles -- check whether the"
            " joints actually reached their stops before deciding to clamp."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
