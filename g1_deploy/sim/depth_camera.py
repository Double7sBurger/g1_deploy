# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render the chest depth camera in MuJoCo, matching the camera the policy trained against.

This is what lets a vision student be tested in closed loop without the robot: MuJoCo renders the
same 64x38 depth image Isaac Lab did, :mod:`g1_deploy.depth` preprocesses it the same way, and the
policy cannot tell which simulator it is in.

MuJoCo's depth buffer is **image-plane depth** -- verified against a flat wall normal to the optical
axis, where every pixel reads the same 1.99 m rather than growing toward the corners -- which is the
same quantity as Isaac Lab's ``distance_to_image_plane`` annotator. Pixels that hit nothing come back
at the far clipping plane and are mapped to :data:`~g1_deploy.depth_link.INVALID`, matching a
RealSense's 0 and Isaac Lab's ``+inf``.

.. attention::
    **Read ``convention`` before the quaternion.** The same physical camera is a different rotation
    under each: this rig is ``+47.6`` degrees about ``+Y`` in ``ros`` and ``-132.4`` in ``world``.
    Taking the raw angle without the convention aims ``depth_student_cl``'s camera 132 degrees down,
    at the ground just past its own feet -- which produced a visibly limping robot while every array
    shape downstream still checked out. See :data:`CONVENTION_OFFSET_DEG`.

"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

from g1_deploy.depth_link import INVALID
from g1_deploy.sim.mjcf import assets_for, compile_model, foot_plate_override, resolve_includes

CONVENTION_OFFSET_DEG = {"ros": 0.0, "world": 180.0}
"""Degrees to add to the signed rotation about ``+Y`` to get the camera's downward pitch.

``convention`` says which local axis the camera looks along -- ROS optical frames use ``+Z``, the
world convention uses ``+X`` -- so the same physical camera is a different quaternion in each. Two
exports of the same rig make that concrete: ``depth_student_w100`` records ``ros`` with ``+47.6``
degrees about ``+Y``, ``depth_student_cl`` records ``world`` with ``-132.4``, and both are a camera
pitched **47.6 degrees down**.

.. attention::
    These offsets were pinned against measurement, not derived. The anchor is the real D435i on this
    robot: at 1.26 m with a 47.6-degree downward pitch, flat ground puts the image centre at 1.71 m,
    and the camera reads 1.79 against MuJoCo's 1.80 with the same pitch applied. Deriving the view
    direction from each convention's forward axis instead gives 42.4 and 47.6 degrees *upward*, which
    matches neither the hardware nor either contract, so one of those axis definitions is wrong and
    the empirical table is used until it is resolved.

    A convention not in this table raises rather than defaulting. Reading a quaternion with the wrong
    convention is exactly the failure that produced a visibly limping robot: ``depth_student_cl`` was
    run with the raw 132.4-degree angle, aiming the camera at the ground a metre in front of the
    feet, and every shape downstream still checked out.
"""


def contract_camera(contract: dict) -> dict:
    """Pull the camera geometry out of a contract into plain numbers.

    Args:
        contract: Parsed ``contract.json``.

    Returns:
        ``pos`` [m] relative to the parent body, ``pitch_deg`` below horizontal, ``fovy_deg``,
        ``width``, ``height``, and the ``max_range`` the policy clips at.

    Raises:
        ValueError: If the rotation is not about the pitch axis, or the convention is unknown --
            both mean the camera is not the one this deployment knows how to reproduce.
    """
    cam = contract["camera"]
    quat = np.asarray(cam["offset_rot_wxyz"], dtype=float)
    quat = quat / (np.linalg.norm(quat) or 1.0)
    axis = quat[1:]
    norm = float(np.linalg.norm(axis))
    if norm > 1e-9 and (abs(axis[0]) > 1e-6 * norm or abs(axis[2]) > 1e-6 * norm):
        raise ValueError(
            f"camera rotation is about {np.round(axis / norm, 3)}, not the pitch axis; this loader"
            " only reproduces a camera pitched about Y"
        )

    convention = cam.get("convention", "ros")
    if convention not in CONVENTION_OFFSET_DEG:
        raise ValueError(
            f"unknown camera convention {convention!r}; known: {sorted(CONVENTION_OFFSET_DEG)}."
            " Guessing would silently aim the camera somewhere else."
        )
    angle = math.degrees(2.0 * math.acos(max(-1.0, min(1.0, quat[0]))))
    signed = angle * (1.0 if norm < 1e-9 or axis[1] >= 0 else -1.0)
    pitch = signed + CONVENTION_OFFSET_DEG[convention]
    if not 0.0 < pitch < 90.0:
        raise ValueError(
            f"camera works out to {pitch:+.1f} degrees, which is not a downward pitch. Either the"
            f" convention table is wrong for {convention!r} or this export describes a camera aimed"
            " somewhere this loader does not expect."
        )

    hfov = math.degrees(2.0 * math.atan(cam["horizontal_aperture_mm"] / (2.0 * cam["focal_length_mm"])))
    aspect = cam["width"] / cam["height"]
    vfov = math.degrees(2.0 * math.atan(math.tan(math.radians(hfov) / 2.0) / aspect))
    return {
        "pos": np.asarray(cam["offset_pos_m"], dtype=float),
        "pitch_deg": pitch,
        "fovy_deg": vfov,
        "width": int(cam["width"]),
        "height": int(cam["height"]),
        "max_range": float(contract["depth_max_range_m"]),
    }


def build_model_with_camera(xml_path: str, spec: dict, body: str = "torso_link",
                            name: str = "depth_camera", foot_plate: bool = False) -> tuple[mujoco.MjModel, str]:
    """Load the scene and attach the depth camera to ``body``.

    The camera is written into the MJCF as an ``<camera>`` child of the body, which is the only way
    MuJoCo lets a camera ride a moving link. Its frame follows MuJoCo's convention -- looking along
    ``-Z`` with ``+Y`` up -- so a camera that looks forward and level has ``xyaxes`` of
    ``(0 -1 0, 0 0 1)``: ``+X_cam`` to the robot's right, ``+Y_cam`` up, hence ``-Z_cam`` forward.
    The pitch then rotates that pair about the camera's own ``+X``.

    Args:
        xml_path: G1 MJCF scene.
        spec: Output of :func:`contract_camera`.
        body: Body to mount on.
        name: Camera name.

    Returns:
        ``(model, camera_name)``.
    """
    import xml.etree.ElementTree as ET

    root = resolve_includes(Path(xml_path))
    target = next((e for e in root.iter("body") if e.get("name") == body), None)
    if target is None:
        raise ValueError(f"no body named {body!r} in {xml_path} or anything it includes")

    add_camera_element(root, spec, body=body, name=name)
    if foot_plate:
        foot_plate_override(root)
    return compile_model(root, xml_path), name


def add_camera_element(root, spec: dict, body: str = "torso_link", name: str = "depth_camera") -> None:
    """Insert the ``<camera>`` element into a resolved MJCF tree, in place.

    Split out from :func:`build_model_with_camera` so a caller that is already editing the tree for
    other reasons -- ``run_sim_loop.py`` swapping the foot geometry -- can add the camera to the
    same tree instead of compiling twice and losing one set of edits.

    Args:
        root: Resolved MJCF tree.
        spec: Output of :func:`contract_camera`.
        body: Body to mount on.
        name: Camera name.

    Raises:
        ValueError: If the body is absent.
    """
    import xml.etree.ElementTree as ET

    target = next((e for e in root.iter("body") if e.get("name") == body), None)
    if target is None:
        raise ValueError(f"no body named {body!r} in this MJCF")
    pitch = math.radians(spec["pitch_deg"])
    # Level, forward-looking MuJoCo camera: +X_cam = -Y_body (right), +Y_cam = +Z_body (up), so
    # -Z_cam is +X_body (forward). Pitching down by `pitch` about +X_cam tips -Z_cam toward the
    # ground and carries +Y_cam forward with it.
    x_axis = np.array([0.0, -1.0, 0.0])
    y_axis = np.array([math.sin(pitch), 0.0, math.cos(pitch)])
    ET.SubElement(
        target,
        "camera",
        {
            "name": name,
            "pos": " ".join(f"{v:.6f}" for v in spec["pos"]),
            "xyaxes": " ".join(f"{v:.6f}" for v in np.concatenate([x_axis, y_axis])),
            "fovy": f"{spec['fovy_deg']:.6f}",
            "mode": "fixed",
        },
    )


class DepthRenderer:
    """Render the policy's depth observation from a MuJoCo state.

    Args:
        model: Model built by :func:`build_model_with_camera`.
        spec: Output of :func:`contract_camera`.
        camera: Camera name.
    """

    def __init__(self, model: mujoco.MjModel, spec: dict, camera: str = "depth_camera") -> None:
        self.spec = spec
        self.camera = camera
        self._renderer = mujoco.Renderer(model, height=spec["height"], width=spec["width"])
        self._renderer.enable_depth_rendering()
        # Anything at or past the far plane hit nothing. MuJoCo scales the far plane by the model's
        # extent, so the threshold has to come from the model rather than being a constant.
        self._far = float(model.stat.extent * model.vis.map.zfar)

    def render(self, data: mujoco.MjData) -> np.ndarray:
        """Depth in metres at the policy's resolution.

        Args:
            data: Current simulator state.

        Returns:
            ``(height, width)`` float32 metres, :data:`~g1_deploy.depth_link.INVALID` where nothing
            was hit.
        """
        self._renderer.update_scene(data, camera=self.camera)
        depth = np.asarray(self._renderer.render(), dtype=np.float32)
        return np.where(depth >= self._far * 0.99, np.float32(INVALID), depth)

    def close(self) -> None:
        """Release the renderer's GL context."""
        self._renderer.close()


def load_camera_spec(export_dir: str | Path) -> dict:
    """Convenience: read ``contract.json`` from an export directory and return the camera spec."""
    path = Path(export_dir)
    if path.is_dir():
        path = path / "contract.json"
    with open(path) as handle:
        return contract_camera(json.load(handle))
