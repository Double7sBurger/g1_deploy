# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Serve a MuJoCo G1 on ``rt/lowstate`` / ``rt/lowcmd``, free-running on wall-clock time.

The direct counterpart of ``gear_sonic/scripts/run_sim_loop.py``: start this, then point any
controller at the same DDS domain and it cannot tell the simulator from a robot. The loop is Sonic's
``BaseSimulator.start`` -- step physics, publish, sleep the remainder of the timestep.

This is the *emulation* mode. It reproduces a real robot's asynchrony, including the transport delay
the controller has to tolerate, which makes it the right tool for asking "will this survive on
hardware". It is the wrong tool for asking "does MuJoCo agree with Isaac Lab", because the answer
then includes 12-16 ms of jitter that is a property of the transport rather than of the policy --
use ``run_policy_loop.py --sim sync`` for that.

Usage::

    # terminal 1
    uv run --no-project python deploy/run_sim_loop.py --viz

    # terminal 2
    uv run --no-project python deploy/run_policy_loop.py --sim none \\
        --policy logs/.../deploy_2999/policy.pt --policy_physics newton --vx 0.5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from g1_deploy.bootstrap import ensure_cyclonedds

ensure_cyclonedds()

from g1_deploy import core
from g1_deploy.sim.bridge import G1SimBridge
from g1_deploy.sim.env import DECIMATION, SIM_DT, G1SimEnv
from g1_deploy.core import build_default_pose, load_policy
from g1_deploy.sim.bridge import await_discovery
from g1_deploy.sim.env import DEFAULT_XML, TRUNK_BODIES
from unitree_sdk2py.core.channel import ChannelFactoryInitialize  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--xml", default=str(DEFAULT_XML))
    parser.add_argument(
        "--policy_physics",
        choices=("physx", "newton", "g1_29dof"),
        default="newton",
        help="Only decides which default pose the robot is reset into; the joint order on the wire"
        " is always G1JointIndex.",
    )
    parser.add_argument("--viz", action="store_true", help="Open a viewer tracking the pelvis.")
    parser.add_argument("--domain_id", type=int, default=1)
    parser.add_argument("--interface", default="lo")
    parser.add_argument("--contact_timeconst", type=float, default=0.005)
    parser.add_argument("--trunk_mass_scale", type=float, default=1.0)
    parser.add_argument(
        "--align_legs_to_usd",
        default=None,
        help="Path to deploy/g1_usd_rest.json. Rewrites the leg rest transforms to the USD the policy"
        " trained on -- use it to reproduce Isaac Lab, not to predict hardware, because the MJCF is"
        " the description that matches a real G1.",
    )
    parser.add_argument(
        "--command_delay_steps",
        type=int,
        default=0,
        help="Physics steps of rt/lowcmd latency, 2 ms each. Leave at 0 here: this server is already"
        " free-running, so real transport delay is in the loop for free.",
    )
    parser.add_argument(
        "--reset_on_fall",
        action="store_true",
        help="Reset to the default pose when the pelvis drops below --fall_height, the way Sonic's"
        " check_fall does. Off by default: a controller measuring survival wants the fall to stand.",
    )
    parser.add_argument("--fall_height", type=float, default=0.2, help="Pelvis height [m] counting as a fall.")
    parser.add_argument(
        "--hoist_s",
        type=float,
        default=0.0,
        help="Hold the pelvis up for this many seconds after the controller connects, then let go."
        " Stands in for the gantry a G1 is brought up on. Needed to rehearse a realistic ramp: the"
        " default pose is not statically stable, so a free-standing robot falls in about 1.5 s.",
    )
    args = parser.parse_args()

    core.set_policy_backend(args.policy_physics)
    ChannelFactoryInitialize(args.domain_id, args.interface)

    bridge = G1SimBridge(num_motors=core.NUM_ROBOT_MOTORS)
    mass_scale = {b: args.trunk_mass_scale for b in TRUNK_BODIES} if args.trunk_mass_scale != 1.0 else None
    env = G1SimEnv(
        args.xml,
        bridge,
        sim_dt=SIM_DT,
        decimation=DECIMATION,
        onscreen=args.viz,
        contact_timeconst=args.contact_timeconst,
        body_mass_scale=mass_scale,
        align_legs_to_usd=args.align_legs_to_usd,
        command_delay_steps=args.command_delay_steps,
    )
    default_pose = build_default_pose()
    root_z = env.reset(default_pose)
    print(
        f"[..] serving rt/lowstate on domain {args.domain_id} / {args.interface}\n"
        f"     {Path(args.xml).name}  root_z={root_z:.3f} m  physics {SIM_DT * 1000:.0f} ms"
        f"  contact timeconst={env.model.geom_solref[1, 0]:.4f}\n"
        f"[..] waiting for rt/lowcmd -- physics is frozen at the reset pose until a controller connects"
    )

    # Freeze at the reset pose until a controller connects; see G1SimEnv.publish_only for why a PD
    # hold is not an option. Without this the robot is flat on the floor by the time a controller
    # finishes importing torch, and every run reads as an instant failure.
    while not bridge.cmd_received:
        if env.viewer is not None:
            if not env.viewer.is_running():
                env.close()
                return 0
            env.viewer.sync()
        env.publish_only()
        time.sleep(SIM_DT)
    print(f"[ok] controller connected at t={env.data.time:.2f}s, physics running")

    step = 0
    viewer_every = max(1, int(round(0.02 / SIM_DT)))
    hoist_steps = int(round(args.hoist_s / SIM_DT))
    hoist_z = float(env.data.qpos[2])
    if hoist_steps:
        print(f"[..] hoist holding the pelvis at {hoist_z:.3f} m for {args.hoist_s:.1f}s, then releasing")
    start = time.monotonic()
    try:
        while env.viewer is None or env.viewer.is_running():
            if step < hoist_steps:
                env.apply_hoist(hoist_z)
            elif step == hoist_steps and hoist_steps:
                env.release_hoist()
                print(f"[..] hoist released at t={env.data.time:.2f}s")
            env.sim_step()
            step += 1
            if env.viewer is not None and step % viewer_every == 0:
                env.viewer.sync()
            if args.reset_on_fall and env.data.qpos[2] < args.fall_height:
                print(f"[warn] fell at t={env.data.time:.2f}s, resetting")
                env.reset(default_pose)
                start, step = time.monotonic(), 0
            lag = start + step * SIM_DT - time.monotonic()
            if lag > 0:
                time.sleep(lag)
    except KeyboardInterrupt:
        print("\n[..] interrupted")
    finally:
        env.close()

    elapsed = time.monotonic() - start
    print(
        f"[ok] {env.data.time:.1f}s simulated in {elapsed:.1f}s wall"
        f"  (real-time factor {env.data.time / max(1e-9, elapsed):.2f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
