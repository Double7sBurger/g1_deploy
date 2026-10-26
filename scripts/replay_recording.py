# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay a deployment recording as motion in MuJoCo, and render it to an mp4.

Kinematic playback, not a re-simulation: the joint angles and the base orientation are written
straight into ``qpos`` and no physics runs. What you see is the posture the robot actually held.

Nothing in the recording measures where the base *was* -- there is no external tracking -- so the
base translation is reconstructed: the height puts the lowest foot contact on the ground, and the
horizontal position assumes the lowest contact does not slide. Posture and limb motion are
measured; travel across the floor is an estimate and drifts whenever a foot scuffs.

Two recordings replay side by side, aligned on the clip's reference time, which is how a hardware
run and its simulation rehearsal are compared.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import mujoco
import numpy as np

from g1_deploy.sim.env import DEFAULT_XML

FPS = 50


def contact_geoms(model):
    """The foot collision spheres, which are what touches the floor."""
    return [g for g in range(model.ngeom) if model.geom_contype[g]
            and "ankle_roll" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                                   model.geom_bodyid[g]) or "")]


class Playback:
    """One recording, posed into its own copy of the model."""

    def __init__(self, directory, xml):
        from g1_deploy.analysis import analyze, window_arrays
        from g1_deploy.hardware import MOTOR_NAMES

        metadata, summary, arrays = analyze(directory)
        arrays = window_arrays(arrays, summary["policy_window_s"])
        self.name = Path(directory).resolve().name
        self.real = bool(metadata["arguments"].get("real"))
        self.t = arrays["t"]
        self.q = arrays["q"][:, :29]
        self.quat = arrays["quaternion"]
        self.reference = (arrays.get("reference_time"), arrays.get("tracking_t"))
        self.model = mujoco.MjModel.from_xml_path(xml)
        self.data = mujoco.MjData(self.model)
        joints = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n + "_joint")
                  for n in MOTOR_NAMES]
        self.qadr = np.array([self.model.jnt_qposadr[j] for j in joints])
        self.feet = contact_geoms(self.model)
        self.offset = np.zeros(3)
        self.planted = None

    def clip_time(self, robot_time):
        """The clip's reference time at a robot-clock instant, for aligning two recordings."""
        reference, stamps = self.reference
        if reference is None:
            return robot_time
        return float(np.interp(robot_time, stamps, reference))

    def pose(self, robot_time):
        """Pose the model at ``robot_time`` and return the estimated base height."""
        i = int(np.clip(np.searchsorted(self.t, robot_time), 0, len(self.t) - 1))
        quat = self.quat[i]
        norm = np.linalg.norm(quat)
        self.data.qpos[3:7] = quat / norm if norm > 1e-6 else (1, 0, 0, 0)
        self.data.qpos[self.qadr] = self.q[i]
        self.data.qpos[:3] = self.offset
        mujoco.mj_forward(self.model, self.data)
        points = self.data.geom_xpos[self.feet]
        lowest = int(np.argmin(points[:, 2]))
        self.offset[2] -= points[lowest, 2]
        if self.planted is not None and self.planted[0] == lowest:
            # The planted foot is assumed not to slide, so any horizontal travel it shows is the
            # base moving the other way. This is the whole of the position estimate.
            self.offset[:2] -= points[lowest, :2] - self.planted[1]
        self.data.qpos[:3] = self.offset
        mujoco.mj_forward(self.model, self.data)
        self.planted = (lowest, self.data.geom_xpos[self.feet][lowest, :2].copy())
        return float(self.data.qpos[2])


def label(frame, lines, width):
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (width, 26 * len(lines) + 12), (0, 0, 0), -1)
    out = cv2.addWeighted(out, 0.75, frame, 0.25, 0)
    for i, (text, colour) in enumerate(lines):
        cv2.putText(out, text, (12, 26 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.58, colour, 1, cv2.LINE_AA)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("recordings", nargs="+", help="One or two recording directories.")
    parser.add_argument("--out", default=None)
    parser.add_argument("--xml", default=str(DEFAULT_XML))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--azimuth", type=float, default=135.0)
    parser.add_argument("--elevation", type=float, default=-12.0)
    parser.add_argument("--distance", type=float, default=3.2)
    args = parser.parse_args()
    if len(args.recordings) > 2:
        parser.error("one or two recordings")

    from g1_deploy.bootstrap import ensure_cyclonedds

    ensure_cyclonedds()
    shots = [Playback(directory, args.xml) for directory in args.recordings]
    out = Path(args.out or f"recordings/{shots[0].name}_replay.mp4")
    span = min(shot.t[-1] - shot.t[0] for shot in shots)
    steps = int(span * FPS)

    renderers = []
    for shot in shots:
        shot.model.vis.global_.offwidth = max(shot.model.vis.global_.offwidth, args.width)
        shot.model.vis.global_.offheight = max(shot.model.vis.global_.offheight, args.height)
        renderers.append(mujoco.Renderer(shot.model, height=args.height, width=args.width))

    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                             (args.width * len(shots), args.height))
    if not writer.isOpened():
        raise RuntimeError(f"could not open {out} for writing")

    for k in range(steps):
        panels = []
        for shot, renderer in zip(shots, renderers):
            when = shot.t[0] + k / FPS
            height = shot.pose(when)
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            camera.trackbodyid = mujoco.mj_name2id(shot.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
            camera.distance, camera.azimuth, camera.elevation = args.distance, args.azimuth, args.elevation
            renderer.update_scene(shot.data, camera=camera)
            frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            panels.append(label(frame, [
                (("hardware" if shot.real else "simulation") + f"   {shot.name}", (255, 255, 255)),
                (f"clip t = {shot.clip_time(when):5.2f} s", (120, 220, 255)),
                (f"pelvis {height:.3f} m (estimated)", (255, 220, 120)),
            ], args.width))
        writer.write(np.hstack(panels) if len(panels) > 1 else panels[0])
    writer.release()
    print(f"[ok] {out}  {steps} frames at {FPS} fps  ({out.stat().st_size / 1e6:.1f} MB)")
    print("[note] posture and joints are measured; the base translation is reconstructed from foot"
          " contact and drifts when a foot slides")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
