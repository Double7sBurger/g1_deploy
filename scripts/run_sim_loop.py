# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Serve a MuJoCo G1 on ``rt/lowstate`` / ``rt/lowcmd``, free-running on wall-clock time.

The direct counterpart of ``gear_sonic/scripts/run_sim_loop.py``: start this, then point any
controller at the same DDS domain and it cannot tell the simulator from a robot. A worker steps
physics and publishes at 500 Hz; the caller renders snapshots at 50 Hz. Neither loop replays
overdue periods after a stall.

This is the *emulation* mode. It reproduces a real robot's asynchrony, including the transport delay
the controller has to tolerate, which makes it the right tool for asking "will this survive on
hardware". It is the wrong tool for asking "does MuJoCo agree with Isaac Lab", because the answer
then includes wall-clock scheduling and transport jitter as well as policy behaviour --
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
import copy
import sys
from pathlib import Path

from g1_deploy.bootstrap import ensure_cyclonedds

ensure_cyclonedds()

from g1_deploy import core
from g1_deploy.sim.bridge import G1SimBridge
from g1_deploy.sim.env import DECIMATION, SIM_DT, G1SimEnv
from g1_deploy.core import build_default_pose, load_policy
from g1_deploy.sim.bridge import await_discovery
from g1_deploy.sim.env import DEFAULT_XML, TRUNK_BODIES
from g1_deploy.sim.realtime import RealtimePhysics
from g1_deploy.timing import PeriodicDeadline
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
    parser.add_argument(
        "--depth",
        default=None,
        metavar="EXPORT_DIR",
        help="Also render the chest depth camera and publish it on the same UDP wire format"
        " scripts/depth_publisher.py uses, so a vision student can be rehearsed against this"
        " simulator with the identical receive path it will use on the robot.",
    )
    parser.add_argument(
        "--camera_pitch",
        type=float,
        default=None,
        help="Render the depth camera at this downward pitch [deg] instead of the contract's."
        " Measured on this robot with scripts/fit_camera_pose.py the physical mount is about 51"
        " degrees against the contract's 47.6; overriding shows what the policy will actually be"
        " fed on hardware.",
    )
    parser.add_argument("--depth_host", default="127.0.0.1", help="Where to send depth frames.")
    parser.add_argument("--depth_port", type=int, default=None)
    parser.add_argument(
        "--lock_waist", action="store_true",
        help="Lock waist roll and pitch, making the simulated robot the 27-DoF G1 that reports"
        " mode_machine 6. Those two motor slots are empty on that variant while every other joint"
        " keeps its index, so this is the whole difference between it and the 29-DoF robot.")
    parser.add_argument(
        "--foot_plate",
        action="store_true",
        help="Swap each foot's four contact spheres for the solid plate training uses. MuJoCo"
        " cannot load the training USD; this reproduces the one override that governs lateral"
        " stability.",
    )
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

    # Build the model before the env so camera and foot-plate edits survive; G1SimEnv would
    # otherwise reload the XML and drop them.
    depth_pub = model = renderer = None
    if args.depth is not None or args.foot_plate or args.lock_waist:
        from pathlib import Path as _Path

        from g1_deploy.sim.mjcf import (
            compile_model,
            foot_plate_override,
            lock_waist,
            resolve_includes,
        )

        root = resolve_includes(_Path(args.xml))
        if args.foot_plate:
            report = foot_plate_override(root)
            n = sum(v["spheres_disabled"] for v in report.values())
            print(f"[..] foot plate: {n} contact spheres disabled, one plate per foot")
        if args.depth is not None:
            from g1_deploy.sim.depth_camera import add_camera_element, contract_camera
            from g1_deploy.depth import load_contract

            spec = contract_camera(load_contract(args.depth), args.camera_pitch)
            add_camera_element(root, spec)
        model = compile_model(root, args.xml)
        if args.lock_waist:
            lock_waist(model)
            print("[..] waist roll and pitch locked: this is the 27-DoF G1 (mode_machine 6)")

    bridge = G1SimBridge(num_motors=core.NUM_ROBOT_MOTORS)
    mass_scale = {b: args.trunk_mass_scale for b in TRUNK_BODIES} if args.trunk_mass_scale != 1.0 else None
    env = G1SimEnv(
        args.xml,
        bridge,
        sim_dt=SIM_DT,
        decimation=DECIMATION,
        onscreen=False,
        contact_timeconst=args.contact_timeconst,
        body_mass_scale=mass_scale,
        align_legs_to_usd=args.align_legs_to_usd,
        command_delay_steps=args.command_delay_steps,
        model=model,
    )
    default_pose = build_default_pose()
    root_z = env.reset(default_pose)
    # The GL thread never renders live physics data or shares a mutable model with it.
    # GL remains on this thread, as required by mjpython on macOS.
    viewer = render_model = render_data = None
    if args.viz or args.depth is not None:
        import mujoco
        import mujoco.viewer

        render_model = copy.copy(env.model)
        render_data = mujoco.MjData(render_model)
        mujoco.mj_copyData(render_data, render_model, env.data)
    if args.viz:
        viewer = mujoco.viewer.launch_passive(
            render_model, render_data, show_left_ui=False, show_right_ui=False,
        )
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = env.pelvis_id
        viewer.cam.distance = 2.5
        viewer.cam.azimuth = 130.0
        viewer.cam.elevation = -20.0
    if args.depth is not None:
        import socket

        from g1_deploy.depth_link import DEFAULT_PORT, pack_frame
        from g1_deploy.sim.depth_camera import DepthRenderer

        renderer = DepthRenderer(render_model, spec)
        depth_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        depth_dest = (args.depth_host, args.depth_port or DEFAULT_PORT)
        print(f"[..] publishing {spec['width']}x{spec['height']} depth to"
              f" {depth_dest[0]}:{depth_dest[1]}")
    print(
        f"[..] serving rt/lowstate on domain {args.domain_id} / {args.interface}\n"
        f"     {Path(args.xml).name}  root_z={root_z:.3f} m  physics {SIM_DT * 1000:.0f} ms"
        f"  contact timeconst={env.model.geom_solref[1, 0]:.4f}\n"
        f"[..] waiting for rt/lowcmd -- physics is frozen at the reset pose until a controller connects"
    )

    physics = RealtimePhysics(
        env, default_pose, hoist_s=args.hoist_s,
        reset_on_fall=args.reset_on_fall, fall_height=args.fall_height,
    )
    physics.start()
    graphics_pacer = PeriodicDeadline(core.CONTROL_DT)
    depth_seq = 0
    try:
        while viewer is None or viewer.is_running():
            physics.check()
            if render_data is not None:
                physics.copy_state(render_model, render_data)
            if renderer is not None:
                # Frames also stream while physics is frozen, before the controller connects.
                frame = renderer.render(render_data)
                depth_sock.sendto(pack_frame(depth_seq, frame), depth_dest)
                depth_seq += 1
            if viewer is not None:
                viewer.sync()
            graphics_pacer.wait()
    except KeyboardInterrupt:
        print("\n[..] interrupted")
    finally:
        physics.stop()
        if viewer is not None:
            viewer.close()
        if renderer is not None:
            renderer.close()
            depth_sock.close()
        env.close()

    physics.check()
    if physics.started_at is not None:
        elapsed = physics.finished_at - physics.started_at
        simulated = physics.steps * SIM_DT
        print(f"[ok] {simulated:.1f}s simulated in {elapsed:.1f}s wall"
              f"  (real-time factor {simulated / max(1e-9, elapsed):.2f})")
        print(f"[..] physics timing: {physics.pacer.overruns} overruns,"
              f" worst {physics.pacer.max_lateness * 1000:.1f} ms; no catch-up bursts")
    if render_data is not None:
        print(f"[..] graphics timing: {graphics_pacer.overruns} overruns,"
              f" worst {graphics_pacer.max_lateness * 1000:.1f} ms; independent of physics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
