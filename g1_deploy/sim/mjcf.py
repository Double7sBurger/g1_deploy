# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Edit a G1 MJCF before compiling it, so MuJoCo can be brought closer to the training asset.

The training runs use Isaac Sim's own ``Robots/Unitree/G1/g1.usd`` with a USD override layer, and
MuJoCo has no USD importer -- there is no way to load that asset here. What *can* be reproduced is
each individual override, applied to Unitree's official MJCF, which starts from the same robot.

:func:`foot_plate_override` is the one that matters for locomotion. Both descriptions ship the same
foot contact approximation -- four 5 mm spheres per sole, at the same coordinates -- and training
replaces them with a solid plate. Four points and a 203 x 66 mm surface are very different things to
walk on: the spheres give almost no resistance to roll about the foot's long axis, which is exactly
the axis a lateral velocity command loads.

Measured consequence of *not* applying it, on the depth student over the 15-episode hold grid:
forward-only commands track at 0.8-0.96 of commanded speed, while every ``vy = -0.5`` episode runs
away at 2.1-2.4x and every ``vy = +0.5`` episode stalls. The failure is entirely in the lateral
axis, which is what the foot geometry governs.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

FOOT_PLATE_SIZE_M = (0.203100, 0.065500, 0.018500)
"""Foot plate as trained: length x width x thickness [m].

Not chosen here. ``scripts/make_g1_ladder.py`` on the training side measures it from the old
``g1_minimal.usd``'s ankle collision mesh -- point-cloud bounding box, with the transform's scale
folded in -- so the plate is geometrically the same foot the previous asset had.
"""

SPHERE_RADIUS_M = 0.005
"""Radius of the contact spheres being replaced. Used to keep the sole at the same height."""


def resolve_includes(path: Path, depth: int = 0) -> ET.Element:
    """Inline every ``<include>`` so the whole scene is one editable tree.

    MuJoCo splits the G1 scene across files -- ``scene_29dof.xml`` includes ``g1_29dof.xml``, which
    holds the bodies -- and both a camera and a foot geom have to be added to a body's own element.
    Compiling from a string also loses the file's directory, so includes could not be followed at
    compile time anyway.

    Args:
        path: File to read.
        depth: Recursion guard.

    Returns:
        The root element with includes replaced by their contents.

    Raises:
        ValueError: If includes nest more than eight deep, which means a cycle.
    """
    if depth > 8:
        raise ValueError(f"include nesting deeper than 8 at {path}")
    root = ET.parse(path).getroot()
    for parent in list(root.iter()):
        for i, child in reversed(list(enumerate(parent))):
            if child.tag != "include":
                continue
            sub = resolve_includes(path.parent / child.get("file"), depth + 1)
            parent.remove(child)
            for j, element in enumerate(list(sub)):
                parent.insert(i + j, element)
    return root


def assets_for(xml_path: str | Path) -> dict:
    """Read the meshes an MJCF references, so it can be compiled from a string.

    Keyed by bare filename only. Adding both the bare name and the relative path makes MuJoCo reject
    the dict outright with "Repeated file name in assets", and dropping the wrong one loses a mesh
    silently.
    """
    base = Path(xml_path).parent
    compiler = ET.parse(xml_path).getroot().find(".//compiler")
    sub = compiler.get("meshdir", "") if compiler is not None else ""
    folder = (base / sub) if sub else base
    assets = {}
    if folder.is_dir():
        for f in sorted(folder.rglob("*")):
            if f.suffix.lower() in (".stl", ".obj", ".png"):
                assets[f.name] = f.read_bytes()
    return assets


def foot_plate_override(root: ET.Element, bodies: tuple[str, ...] = ("left_ankle_roll_link", "right_ankle_roll_link"),
                        size_m: tuple[float, float, float] = FOOT_PLATE_SIZE_M) -> dict:
    """Replace each foot's contact spheres with one solid plate, as training does.

    The plate is placed from the spheres it replaces rather than from a hand-picked offset: centred
    on their footprint, and with its underside at the spheres' contact plane so the robot stands at
    exactly the same height. Getting that height wrong would change the leg configuration at
    touchdown and confound the very comparison this is for.

    The spheres are disabled by clearing ``contype``/``conaffinity`` rather than deleted, so the
    model still reports them and the change is visible in a diff of the compiled model.

    Args:
        root: Resolved MJCF tree, from :func:`resolve_includes`.
        bodies: Bodies whose spheres to replace.
        size_m: Plate length, width, thickness [m].

    Returns:
        Per body, the number of spheres disabled and the plate's ``pos``/``size``.

    Raises:
        ValueError: If a body is missing, or has no contact spheres to replace -- either means this
            MJCF does not describe the feet the way the override assumes, and silently adding a
            second contact surface on top of the existing one would be worse than stopping.
    """
    report = {}
    for name in bodies:
        body = next((e for e in root.iter("body") if e.get("name") == name), None)
        if body is None:
            raise ValueError(f"no body named {name!r} in this MJCF")

        # MuJoCo defaults an untyped geom to a sphere, and Unitree's MJCF relies on that -- the four
        # contact spheres carry only `size` and `pos`. Matching on an explicit type="sphere" finds
        # none of them and would silently leave the spheres in place beside the new plate.
        spheres = [
            g for g in body.findall("geom")
            if g.get("type", "sphere") == "sphere"
            and len(g.get("size", "").split()) == 1
            and g.get("contype", "1") != "0"
            and g.get("conaffinity", "1") != "0"
        ]
        if not spheres:
            raise ValueError(f"{name!r} has no enabled contact spheres; this override does not apply")

        centres = [[float(v) for v in g.get("pos", "0 0 0").split()] for g in spheres]
        radius = float(spheres[0].get("size", str(SPHERE_RADIUS_M)).split()[0])
        xs = [c[0] for c in centres]
        zs = [c[2] for c in centres]
        plate_pos = (
            (min(xs) + max(xs)) / 2.0,
            0.0,
            min(zs) - radius + size_m[2] / 2.0,
        )
        for g in spheres:
            g.set("contype", "0")
            g.set("conaffinity", "0")
        ET.SubElement(body, "geom", {
            "name": f"{name}_foot_plate",
            "type": "box",
            "pos": " ".join(f"{v:.6f}" for v in plate_pos),
            "size": " ".join(f"{v / 2.0:.6f}" for v in size_m),
            "contype": "1",
            "conaffinity": "1",
        })
        report[name] = {"spheres_disabled": len(spheres), "pos": plate_pos,
                        "half_size": tuple(v / 2.0 for v in size_m)}
    return report


LOCKED_WAIST_JOINTS = ("waist_roll_joint", "waist_pitch_joint")
"""The two joints a 27-DoF G1 does not have. Motor slots 13 and 14, which it reports as empty."""


def lock_waist(model: mujoco.MjModel) -> None:
    """Make the waist rigid in roll and pitch, the way a 27-DoF G1 is.

    Two changes, and both are needed to match the robot rather than merely restrain it. The joint is
    pinned to zero travel, so the link above the waist is rigid; and the actuator's gear is zeroed,
    so the torque the policy asks for goes nowhere instead of fighting a limit. A run that only did
    the first would measure a policy straining against a hard stop, which is not what a robot without
    the motor does.

    Joint indices are left in place on purpose. The 27-DoF variant does not renumber -- measured on
    the robot, slots 0-12 and 15-28 carry live data at their 29-DoF positions and 13-14 read zero --
    so the mapping in :mod:`g1_deploy.core` still describes it and only these two joints differ.

    Args:
        model: Compiled model, modified in place.

    Raises:
        ValueError: If a joint or its actuator is missing, rather than silently locking nothing.
    """
    for name in LOCKED_WAIST_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"{name} is not in this model; --lock_waist has nothing to lock")
        model.jnt_limited[jid] = 1
        model.jnt_range[jid] = (0.0, 0.0)
        aid = int(np.flatnonzero(model.actuator_trnid[:, 0] == jid)[0]) if np.any(
            model.actuator_trnid[:, 0] == jid) else -1
        if aid < 0:
            raise ValueError(f"no actuator drives {name}; --lock_waist would leave it powered")
        model.actuator_gear[aid, 0] = 0.0


def compile_model(root: ET.Element, xml_path: str | Path) -> mujoco.MjModel:
    """Compile an edited tree, pulling meshes from the original file's directory."""
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"), assets_for(xml_path))


def ground_height(model: mujoco.MjModel, x: float = 0.0, y: float = 0.0) -> float:
    """Height of the world's collision geometry under ``(x, y)``, by raycast.

    Zero on a plane, where the raycast is redundant. Not zero on generated terrain: Isaac Lab's
    ``random_rough`` spans 0 to +0.1 m and ``boxes`` spans -1.0 to +0.125 m, so a spawn routine that
    assumes a floor at z=0 buries the robot, the solver ejects it, and every episode ends in a
    fraction of a second.

    Args:
        model: Compiled model.
        x: Query point [m].
        y: Query point [m].

    Returns:
        Surface height [m]; 0.0 if the ray hits nothing.
    """
    # Park the robot far below before casting. mj_ray has no world-geometry-only mode and its
    # bodyexclude takes a single body, so a robot at its spawn pose is simply the first thing the ray
    # meets -- measured, that returns its own pelvis at 1.324 m on every terrain.
    data = mujoco.MjData(model)
    data.qpos[2] = -1000.0
    mujoco.mj_forward(model, data)
    geomid = np.zeros(1, dtype=np.int32)
    dist = mujoco.mj_ray(model, data, np.array([x, y, 20.0]), np.array([0.0, 0.0, -1.0]),
                         None, 1, -1, geomid)
    return 0.0 if dist < 0 else float(20.0 - dist)
