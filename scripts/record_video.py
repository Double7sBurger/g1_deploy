# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record the DDS sim-to-sim loop to an mp4 while stepping through a command schedule.

Offscreen rendering the way Sonic does it (``DefaultEnv.init_renderers`` /
``update_render_caches``), but driving :mod:`run_policy_loop`'s synchronous DDS loop so what ends up
in the video is exactly what :mod:`benchmark_dds` measures -- same transport, same physics, same 50
Hz policy period. Nothing here touches the control path; the renderer only reads state.

The schedule uses step changes rather than ramps because that is what training saw: the command
generator resamples every 10 s and never interpolates, so a step is in distribution and a smooth ramp
would be the unusual input.

Usage::

    uv run --no-project python deploy/record_video.py \\
        --policy logs/.../deploy_2999/policy.pt --out /tmp/g1_sim2sim.mp4
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from g1_deploy.bootstrap import ensure_cyclonedds

ensure_cyclonedds()

from g1_deploy import benchmark as bp
import cv2  # noqa: E402
from g1_deploy import core
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from g1_deploy.controller import G1ControlLink, read_joint_state
from g1_deploy.sim.bridge import G1SimBridge
from g1_deploy.sim.env import SIM_DT, G1SimEnv
from g1_deploy.core import build_default_pose, load_policy
from g1_deploy.sim.bridge import await_discovery
from g1_deploy.sim.env import DEFAULT_XML, TRUNK_BODIES
from unitree_sdk2py.core.channel import ChannelFactoryInitialize  # noqa: E402


def annotate(frame: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> np.ndarray:
    """Draw an overlay so the video is readable without the console beside it."""
    out = frame.copy()
    box_h = 26 * len(lines) + 12
    cv2.rectangle(out, (0, 0), (430, box_h), (0, 0, 0), -1)
    out = cv2.addWeighted(out, 0.75, frame, 0.25, 0)
    for i, (text, colour) in enumerate(lines):
        cv2.putText(out, text, (12, 26 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 1, cv2.LINE_AA)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--policy", required=True)
    parser.add_argument("--policy_physics", choices=("physx", "newton"), default="newton")
    parser.add_argument("--xml", default=str(DEFAULT_XML))
    parser.add_argument("--out", default="/tmp/mj_video/clip_0000.mp4")
    parser.add_argument("--csv", default=None, help="Per-step overlay data; defaults beside --out.")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=int, default=50, help="Frames per second; 50 is one per control step.")
    parser.add_argument("--trunk_mass_scale", type=float, default=0.7576)
    parser.add_argument(
        "--align_legs_to_usd",
        default=None,
        help="Path to deploy/g1_usd_rest.json to rewrite the leg rest transforms to the USD's, which"
        " reproduces Isaac Lab's leg kinematics exactly. See kinematics_align.",
    )
    parser.add_argument("--contact_timeconst", type=float, default=0.005)
    parser.add_argument("--domain_id", type=int, default=17)
    parser.add_argument("--interface", default="lo")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Write frames without the text overlay. Use when feeding stitch_videos.py, which draws"
        " both panels' overlays itself so the two sides cannot drift apart visually.",
    )
    parser.add_argument(
        "--stop_on_fall",
        action="store_true",
        help="End the recording at the first fall instead of filming the aftermath.",
    )
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
        sim_dt=SIM_DT,
        decimation=int(round(core.CONTROL_DT / SIM_DT)),
        contact_timeconst=args.contact_timeconst,
        body_mass_scale=mass_scale,
        align_legs_to_usd=args.align_legs_to_usd,
    )
    env.reset(default_pose)
    link = G1ControlLink(num_motors=core.NUM_ROBOT_MOTORS)
    mode_machine = await_discovery(link, bridge, env, default_pose, kp, kd)
    runner = core.G1PolicyRunner(policy)
    runner.reset()

    # The MJCF declares a 640x480 offscreen framebuffer and Renderer refuses to exceed it. Raise it
    # on the loaded model, not in the file -- same in-memory-override rule as every other knob here.
    env.model.vis.global_.offwidth = max(env.model.vis.global_.offwidth, args.width)
    env.model.vis.global_.offheight = max(env.model.vis.global_.offheight, args.height)
    renderer = mujoco.Renderer(env.model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = env.pelvis_id
    camera.distance, camera.azimuth, camera.elevation = 3.0, 130.0, -15.0

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError(f"could not open {args.out} for writing")

    total = bp.demo_duration()
    n_steps = int(round(total / core.CONTROL_DT))
    render_every = max(1, int(round(1.0 / (args.fps * core.CONTROL_DT))))
    print(f"[..] recording {total:.0f}s -> {args.out}  {args.width}x{args.height} @ {args.fps} fps")

    fell_at = None
    rows = []
    for k in range(n_steps):
        elapsed = k * core.CONTROL_DT
        state = link.wait_for_tick(int(round(env.data.time * 1e3)))
        quat = np.asarray(state.imu_state.quaternion[:4], dtype=np.float32)
        gyro = np.asarray(state.imu_state.gyroscope[:3], dtype=np.float32)
        q, dq = read_joint_state(state, core.NUM_ROBOT_MOTORS)

        vx, vy, heading, caption = bp.demo_command(elapsed)
        command = np.array([vx, vy, core.yaw_rate_from_heading(heading, quat)], dtype=np.float32)

        measured = env.base_lin_vel_b()
        gravity_z = env.gravity_z
        if fell_at is None and gravity_z > -0.7:
            fell_at = elapsed
            print(f"[warn] fell at t={elapsed:.2f}s during '{caption}'")
            if args.stop_on_fall:
                break

        rows.append(
            {
                "t": elapsed,
                "caption": caption,
                "cmd_vx": vx,
                "cmd_vy": vy,
                "heading_deg": float(np.rad2deg(heading)),
                "cmd_wz": float(command[2]),
                "meas_vx": float(measured[0]),
                "meas_vy": float(measured[1]),
                "meas_wz": float(env.data.qvel[5]),
                "root_z": float(env.data.qpos[2]),
                "gravity_z": gravity_z,
            }
        )

        if k % render_every == 0:
            renderer.update_scene(env.data, camera=camera)
            frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            status = ("FALLEN", (80, 80, 255)) if fell_at is not None else ("upright", (140, 255, 140))
            writer.write(
                frame
                if args.raw
                else annotate(
                    frame,
                    [
                        (f"t = {elapsed:5.2f} s   {caption}", (255, 255, 255)),
                        (
                            f"cmd  vx {vx:+.2f}  vy {vy:+.2f}  yaw* {np.rad2deg(heading):+.0f} deg"
                            f"  -> wz {command[2]:+.2f}",
                            (120, 220, 255),
                        ),
                        (
                            f"meas vx {measured[0]:+.2f}  vy {measured[1]:+.2f}"
                            f"  wz {float(env.data.qvel[5]):+.2f}   z {float(env.data.qpos[2]):.2f} m",
                            (255, 220, 120),
                        ),
                        (f"MuJoCo <- DDS rt/lowstate / rt/lowcmd   {status[0]}", status[1]),
                    ],
                )
            )

        target, _ = runner.step(q, dq, quat, gyro, command)
        target[core.UNMAPPED_ROBOT_MOTORS] = 0.0
        seen = bridge.cmd_count
        link.send(target, kp, kd, mode_machine)
        while bridge.cmd_count == seen:
            pass
        env.step_control()

    csv_path = Path(args.csv) if args.csv else Path(args.out).with_name("overlay.csv")
    with open(csv_path, "w", newline="") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        csv_writer.writeheader()
        csv_writer.writerows(rows)

    writer.release()
    renderer.close()
    env.close()
    survived = fell_at if fell_at is not None else total
    print(f"[ok] {args.out}  upright for {survived:.2f}s of {total:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
