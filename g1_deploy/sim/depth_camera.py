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
    **Resolve the rotation into a view direction; do not read its angle.** The same physical camera
    is a different quaternion under each ``convention``, and two exports differing only in the sign
    of the forward component -- one aimed at the ground, one into the robot's own torso -- carry the
    *same* rotation angle. Reading the angle alone maps both onto a plausible view of the floor. See
    :func:`camera_view_dir`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

from g1_deploy.depth_link import INVALID
from g1_deploy.sim.mjcf import assets_for, compile_model, foot_plate_override, resolve_includes


CONVENTION_INVERTED = {"world": True, "ros": False}
"""Whether a contract's ``offset_rot_wxyz`` maps parent-to-camera and so must be inverted.

Anchored on exports whose intended aim is known, not derived from Isaac Lab's source. See
:func:`camera_view_dir` for the evidence and for why the rotation *angle* is not enough.
"""

SCENE_AXIS = np.array([1.0, 0.0, 0.0])
"""The camera axis that looks into the scene, in its own frame: forward, under both conventions."""


def camera_view_dir(quat_wxyz, convention: str) -> np.ndarray:
    """Direction the camera looks, in the parent body's frame.

    Four exports here have a known intended aim, and exactly one reading satisfies all four::

        mjd_v2, yhkd_v2, ymsd_v2   [ 0.9150, 0, -0.4035, 0]  world   ->  [ 0.674, 0, -0.738]
        depth_student_w100         [ 0.9150, 0, +0.4035, 0]  ros     ->  [ 0.674, 0, -0.738]
        the seven older exports    [ 0.4035, 0, -0.9150, 0]  world   ->  [-0.674, 0, -0.738]

    The first four are forward and 47.6 degrees down, which is the mount measured on this robot at
    48.2. The last is *backward* and down -- into the robot's own torso -- which is what the training
    config's docstring says the superseded value did, and it read 0.00 to 0.17 m on every pixel.

    The ``ros`` export is the exact conjugate of the ``world`` one: the same physical camera stored
    the other way round. That is the whole content of :data:`CONVENTION_INVERTED`. The scene-facing
    axis is forward in both -- ``ros``'s optical ``+Z`` is not it, and reading it that way yields
    42.4 degrees, a plausible-looking angle that is simply not this camera.

    .. attention::
        The rotation's *angle* cannot decide this. The correct and the torso-facing exports are both
        47.6 degrees; they differ only in the sign of the forward component. An earlier version of
        this module read the angle and added a per-convention constant, so a camera aimed backward
        rendered as an unremarkable view of the ground -- and five students were distilled against
        it before anything downstream noticed.

    Args:
        quat_wxyz: ``offset_rot_wxyz`` from the contract.
        convention: ``world`` or ``ros``.

    Returns:
        Unit view direction in the parent frame, ``x`` forward and ``z`` up.

    Raises:
        ValueError: On an unknown convention. Guessing would aim the camera silently.
    """
    if convention not in CONVENTION_INVERTED:
        raise ValueError(
            f"unknown camera convention {convention!r}; known: {sorted(CONVENTION_INVERTED)}"
        )
    quat = np.asarray(quat_wxyz, dtype=float)
    quat = quat / (np.linalg.norm(quat) or 1.0)
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, quat)
    rot = rot.reshape(3, 3)
    return (rot.T if CONVENTION_INVERTED[convention] else rot) @ SCENE_AXIS


def contract_camera(contract: dict, pitch_override_deg: float | None = None) -> dict:
    """Pull the camera geometry out of a contract into plain numbers.

    Args:
        contract: Parsed ``contract.json``.
        pitch_override_deg: Use this downward pitch instead of the contract's. Measured on this
            robot with ``scripts/fit_camera_pose.py`` the real mount is 48.2 degrees against the
            contract's 47.6 -- agreement, so this exists for the case where they genuinely diverge,
            and for running an export whose rotation this loader rejects.

    Returns:
        ``pos`` [m] relative to the parent body, ``pitch_deg`` below horizontal, ``fovy_deg``,
        ``width``, ``height``, and the ``max_range`` the policy clips at.

    Raises:
        ValueError: If the camera does not look forward and down. Both halves matter: a rotation
            that aims it backward is the known broken export, and one that aims it up is not a
            terrain camera. Either way rendering it anyway would produce a plausible picture of
            something the policy was not trained on.
    """
    cam = contract["camera"]
    view = camera_view_dir(cam["offset_rot_wxyz"], cam.get("convention", "ros"))
    pitch = math.degrees(math.asin(-max(-1.0, min(1.0, view[2]))))

    if pitch_override_deg is not None:
        pitch = float(pitch_override_deg)
    elif not (view[0] > 0.0 and 0.0 < pitch < 90.0):
        raise ValueError(
            f"the camera's view direction is {np.round(view, 3)} in the parent frame: "
            + ("it points backward, into the robot" if view[0] <= 0 else f"its pitch is {pitch:+.1f} deg")
            + ". This export was made with a camera rotation that does not look forward and down."
            " Pass --camera_pitch to render it at a chosen angle anyway."
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
