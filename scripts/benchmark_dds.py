# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the sim-to-sim benchmark through the DDS loop and write per-episode metrics.

Third runner for the protocol in :mod:`benchmark_protocol`, alongside :mod:`benchmark_isaaclab` and
:mod:`benchmark_mujoco`. The difference from :mod:`benchmark_mujoco` is the path, not the physics:
every observation arrives as an ``rt/lowstate`` sample and every action leaves as ``rt/lowcmd``,
through the same :class:`~dds_controller.G1ControlLink` that would talk to the robot. If a number
here disagrees with :mod:`benchmark_mujoco`, the transport is the difference.

Episodes run serially on one DDS link. ``ChannelFactoryInitialize`` is process-global and two workers
on one domain would hear each other's traffic, so there is no worker pool here; the simulator time
instead runs monotonically across episodes, which is what keeps ``tick`` usable as an identifier
after a reset.

Usage::

    uv run --no-project python deploy/benchmark_dds.py \\
        --policy logs/.../deploy_2999/policy.pt --suite hold --repeats 3 --out bench_dds_hold.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from g1_deploy.bootstrap import ensure_cyclonedds

ensure_cyclonedds()

from g1_deploy import benchmark as bp
from g1_deploy import core
import numpy as np  # noqa: E402
from g1_deploy.controller import G1ControlLink, read_joint_state
from g1_deploy.sim.bridge import G1SimBridge
from g1_deploy.sim.env import SIM_DT, G1SimEnv
from g1_deploy.core import build_default_pose, load_policy
from g1_deploy.sim.bridge import await_discovery
from g1_deploy.sim.env import DEFAULT_XML, TRUNK_BODIES
from unitree_sdk2py.core.channel import ChannelFactoryInitialize  # noqa: E402

EPISODE_GAP_S = 1.0
"""Simulated seconds skipped between episodes, so no two episodes share a ``tick``."""


def run_episode(episode, env, bridge, link, runner, kp, kd, mode_machine, default_pose) -> dict:
    """Run one protocol episode over DDS and return its scored metrics."""
    jitter_motor = np.zeros(core.NUM_ROBOT_MOTORS, dtype=np.float32)
    mapped = core.POLICY_TO_ROBOT >= 0
    motor_names = [core.POLICY_JOINT_NAMES[i] for i in np.flatnonzero(mapped)]
    jitter_motor[core.POLICY_TO_ROBOT[mapped]] = episode.joint_jitter(motor_names)

    env.reset(default_pose + jitter_motor, start_time=env.data.time + EPISODE_GAP_S)
    bridge.publish_low_state(env.prepare_obs())  # nothing else publishes until the first step
    runner.reset()

    n_steps = int(round(bp.EPISODE_S / bp.CONTROL_DT))
    rec = {k: [] for k in ("cmd", "lin_vel_b", "ang_vel_b", "gravity_z", "root_z")}
    for k in range(n_steps):
        state = link.wait_for_tick(int(round(env.data.time * 1e3)))
        quat = np.asarray(state.imu_state.quaternion[:4], dtype=np.float32)
        gyro = np.asarray(state.imu_state.gyroscope[:3], dtype=np.float32)
        q, dq = read_joint_state(state, core.NUM_ROBOT_MOTORS)

        vx, vy, heading, standing = episode.planar_command(k)
        command = (
            np.zeros(3, dtype=np.float32)
            if standing
            else np.array([vx, vy, core.yaw_rate_from_heading(heading, quat)], dtype=np.float32)
        )

        # Ground truth for scoring only, read off the simulator rather than off the wire: a real G1
        # publishes neither of these, and routing them through DDS would put them one import away
        # from becoming a policy input.
        rec["cmd"].append(command)
        rec["lin_vel_b"].append(env.base_lin_vel_b())
        # .copy(): np.asarray on a MuJoCo view aliases the live buffer, so every recorded sample
        # would end up holding the last step's value.
        rec["ang_vel_b"].append(np.array(env.data.qvel[3:6]))
        rec["gravity_z"].append(env.gravity_z)
        rec["root_z"].append(float(env.data.qpos[2]))

        target, _ = runner.step(q, dq, quat, gyro, command)
        target[core.UNMAPPED_ROBOT_MOTORS] = 0.0

        seen = bridge.cmd_count
        link.send(target, kp, kd, mode_machine)
        deadline = time.monotonic() + 2.0
        while bridge.cmd_count == seen:
            if time.monotonic() > deadline:
                raise TimeoutError("simulator did not receive rt/lowcmd within 2s")
            time.sleep(0.0002)
        env.step_control()

    return bp.score({key: np.array(v) for key, v in rec.items()})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--policy", required=True)
    parser.add_argument("--policy_physics", choices=("physx", "newton", "g1_29dof"), default="newton")
    parser.add_argument("--suite", choices=("hold", "sequence"), default="hold")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--xml", default=str(DEFAULT_XML))
    parser.add_argument("--sim_dt", type=float, default=SIM_DT)
    parser.add_argument("--contact_timeconst", type=float, default=0.005)
    parser.add_argument("--trunk_mass_scale", type=float, default=1.0)
    parser.add_argument(
        "--command_delay_steps",
        type=int,
        default=0,
        help="Physics steps of rt/lowcmd latency; 2 ms each at the default sim_dt. Real transport is"
        " 12-16 ms, so 0 flatters the policy.",
    )
    parser.add_argument(
        "--align_legs_to_usd",
        default=None,
        help="Path to deploy/g1_usd_rest.json to rewrite the leg rest transforms to the USD's, which"
        " reproduces Isaac Lab's leg kinematics exactly. See kinematics_align.",
    )
    parser.add_argument("--integrator", choices=("euler", "rk4", "implicit", "implicitfast"), default=None)
    parser.add_argument("--domain_id", type=int, default=7)
    parser.add_argument("--interface", default="lo")
    parser.add_argument("--out", default="deploy/benchmark_dds.npz")
    args = parser.parse_args()

    core.set_policy_backend(args.policy_physics)
    policy = load_policy(args.policy)
    default_pose = build_default_pose()
    kp, kd = core.control_gains()

    ChannelFactoryInitialize(args.domain_id, args.interface)
    bridge = G1SimBridge(num_motors=core.NUM_ROBOT_MOTORS)
    mass_scale = {b: args.trunk_mass_scale for b in TRUNK_BODIES} if args.trunk_mass_scale != 1.0 else None
    env = G1SimEnv(
        args.xml,
        bridge,
        sim_dt=args.sim_dt,
        decimation=int(round(bp.CONTROL_DT / args.sim_dt)),
        contact_timeconst=args.contact_timeconst,
        body_mass_scale=mass_scale,
        align_legs_to_usd=args.align_legs_to_usd,
        command_delay_steps=args.command_delay_steps,
        integrator=args.integrator,
    )
    env.reset(default_pose)
    link = G1ControlLink(num_motors=core.NUM_ROBOT_MOTORS)
    mode_machine = await_discovery(link, bridge, env, default_pose, kp, kd)
    runner = core.G1PolicyRunner(policy)

    n = bp.num_episodes(args.suite, args.repeats)
    print(
        f"[..] dds  suite={args.suite}  {n} episodes x {bp.EPISODE_S:.0f}s"
        f"  physics {args.sim_dt * 1000:.1f} ms x {env.decimation}  domain {args.domain_id}"
    )
    started = time.monotonic()
    results = []
    for i in range(n):
        results.append(
            run_episode(bp.Episode(args.suite, i), env, bridge, link, runner, kp, kd, mode_machine, default_pose)
        )
        if (i + 1) % 10 == 0 or i + 1 == n:
            done = np.mean([r["success"] for r in results])
            print(f"     {i + 1}/{n}  success so far {done:.0%}  ({time.monotonic() - started:.0f}s)")

    keys = list(results[0])
    np.savez(
        args.out,
        suite=args.suite,
        policy=args.policy,
        side="dds",
        **{key: np.array([r[key] for r in results]) for key in keys},
    )
    print(
        f"[ok] {args.out}  success {np.mean([r['success'] for r in results]):.1%}"
        f"  lin err {np.nanmean([r['lin_vel_err'] for r in results]):.3f} m/s"
    )
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
