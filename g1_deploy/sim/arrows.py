# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Draw the commanded and achieved base velocity as arrows over the robot, the way Isaac Lab does.

Isaac Lab's velocity task hangs two arrows above the base -- one for the command, one for what the
robot is actually doing -- and the gap between them is the whole task at a glance. This reproduces
that in MuJoCo's passive viewer.

The arrows are drawn into ``viewer.user_scn``, a scratch scene the viewer composites on top of the
model's own geometry. They are decoration: nothing here is part of the model, collides, or is
visible to the depth camera, which renders from the model and never sees ``user_scn``.

Both arrows leave the same point, parallel, scaled by the same metres-per-unit factor, so the
tracking error is the bit of arrow sticking out past the other one. Two arrows scaled independently
would always look like good tracking, and two drawn from different points would have to be measured
rather than seen.

They are separated vertically by a few centimetres and nothing more. Exactly coincident is worse,
not better: the usual case is a forward command with no lateral component, where the two are
coaxial and the shorter one disappears inside the longer.

Yaw rate gets its own pair, well clear of the linear ones, because it is a different quantity in
different units -- rad/s, not m/s -- and stacking it alongside invites reading the three as one
vector. It is drawn across the body, pointing to the robot's left when the yaw rate is positive,
rather than as a curved arrow, which at this size reads as noise.
"""

from __future__ import annotations

import numpy as np

import mujoco

COMMAND_RGBA = (0.2, 0.85, 0.25, 0.85)
"""Green for the command, matching Isaac Lab's."""

ACHIEVED_RGBA = (0.25, 0.5, 0.95, 0.85)
"""Blue for what the robot managed."""

YAW_RGBA = (0.95, 0.65, 0.15, 0.85)
"""Amber for yaw rate, which is a different quantity in different units and should not be mistaken
for a third linear velocity."""

ARROW_HEIGHT = 0.75
"""How far above the pelvis the arrows float [m]. The pelvis sits near 0.78 and the head reaches
about 1.3, so anything under half a metre draws the arrows through the head."""

ARROW_SCALE = 1.0
"""Metres of arrow per m/s. At 1.0 a 0.5 m/s command is half a metre, comfortably readable against
a robot 1.3 m tall; shorter than that and the gap between the two arrows is hard to see, which is
the only thing the picture is for."""

ARROW_WIDTH = 0.022
"""Shaft radius [m]."""

ARROW_PAIR_GAP = 0.07
"""Vertical gap between the two arrows of a pair [m]. Enough that neither hides the other when they
point the same way, small enough that their tails still read as the same origin."""

YAW_STACK = 0.30
"""How far above the linear pair the yaw pair sits [m]."""

MIN_LENGTH = 0.02
"""Arrows shorter than this [m] are skipped. ``mjv_connector`` degenerates at zero length, and a
stub pointing in the direction of sensor noise is worse than nothing."""


def _arrow(scene, start: np.ndarray, vec: np.ndarray, rgba, width: float = ARROW_WIDTH) -> None:
    """Append one arrow to a scene, if there is room and it is long enough.

    Args:
        scene: An ``mjvScene``, normally ``viewer.user_scn``.
        start: Tail of the arrow in world coordinates [m].
        vec: Arrow as a displacement [m]; its length is the drawn length.
        rgba: Colour.
        width: Shaft radius [m].
    """
    if scene.ngeom >= scene.maxgeom or float(np.linalg.norm(vec)) < MIN_LENGTH:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
                        np.zeros(9), np.asarray(rgba, dtype=np.float32))
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, width,
                         np.asarray(start, dtype=np.float64),
                         np.asarray(start + vec, dtype=np.float64))
    scene.ngeom += 1


def draw_velocity(viewer, model, data, pelvis_id: int, command=None,
                  scale: float = ARROW_SCALE) -> None:
    """Redraw the velocity arrows for this frame.

    Clears ``user_scn`` first: these arrows are a snapshot of one instant, and appending to it every
    frame would leave a trail that looks like a plot of the robot's history and is really a leak
    until ``maxgeom`` is reached.

    The velocities are rotated into the robot's frame and back out, rather than drawn along world
    axes, because the command *is* in the robot's frame -- "forward at 0.5" follows the robot around
    a turn, and an arrow along world +X would disagree with it for no reason.

    Args:
        viewer: A passive viewer with a ``user_scn``. Ignored if ``None``.
        model: Compiled model, for the body position.
        data: Current state.
        pelvis_id: Body id to hang the arrows over.
        command: ``(vx, vy, yaw_rate)`` in the body frame, or ``None`` to draw only what the robot
            is doing -- which is the honest picture when no controller has said anything yet.
        scale: Metres of arrow per m/s.
    """
    if viewer is None or not hasattr(viewer, "user_scn"):
        return
    scene = viewer.user_scn
    scene.ngeom = 0

    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, data.qpos[3:7])
    rot = rot.reshape(3, 3)
    # Heading only. Tipping the arrows with the trunk's pitch and roll would make them swing with
    # every stride and read as velocity that is not there.
    heading = np.arctan2(rot[1, 0], rot[0, 0])
    forward = np.array([np.cos(heading), np.sin(heading), 0.0])
    left = np.array([-np.sin(heading), np.cos(heading), 0.0])

    base = np.array(data.xpos[pelvis_id]) + np.array([0.0, 0.0, ARROW_HEIGHT])
    gap = np.array([0.0, 0.0, ARROW_PAIR_GAP])
    achieved = rot.T @ data.qvel[0:3]
    _arrow(scene, base, (achieved[0] * forward + achieved[1] * left) * scale, ACHIEVED_RGBA)
    if command is None:
        return
    command = np.asarray(command, dtype=float)
    _arrow(scene, base + gap, (command[0] * forward + command[1] * left) * scale, COMMAND_RGBA)

    # Yaw rate, in rad/s on the same arrow scale, kept well above the linear pair.
    yaw_base = base + np.array([0.0, 0.0, YAW_STACK])
    _arrow(scene, yaw_base, float(data.qvel[5]) * left * scale * 0.5,
           ACHIEVED_RGBA, ARROW_WIDTH * 0.7)
    _arrow(scene, yaw_base + gap * 0.7, command[2] * left * scale * 0.5,
           YAW_RGBA, ARROW_WIDTH * 0.7)
