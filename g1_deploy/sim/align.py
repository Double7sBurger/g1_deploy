# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rewrite a MuJoCo G1's leg rest transforms to match the USD the policy was trained on.

The two robot descriptions disagree about the legs. Measured in the pelvis frame at identical joint
angles, the MJCF's ankle sits 5.6 cm below the USD's, so the same joint command stands the MuJoCo
robot about 6 cm taller -- which is the whole of the "MuJoCo walks upright, Isaac Lab crouches"
difference, and most of the stability difference with it, because a leg near full extension has poor
leverage.

**It is not different bones.** Every leg segment length agrees to 0.1 mm (thigh 0.1938 vs 0.1939 m,
shank 0.3001 vs 0.3000 m) and every joint axis agrees to within 0.04 of canonical. What differs is
how the three hip joints are stacked: the USD places hip roll coincident in z with hip pitch, the
MJCF places it 3 cm lower, and the offset compounds down the chain. So the fix is the parent-relative
*rest transform* of each leg body, and nothing else.

Only ``body_pos`` and ``body_quat`` are touched. Joint axes already match, joint anchors are all at
the child body origin in both, and masses and inertias are left alone -- this aligns kinematics, not
dynamics, and :func:`align_legs` says so by refusing to guess about the rest.

.. attention::
    For sim-to-sim this closes a real gap. For sim-to-real it points the wrong way: a G1 EDU is
    specified at about 35 kg and the MJCF is the description that matches the hardware, so the model
    that needs correcting is the USD used for training. Use this to explain a discrepancy, not to
    declare one fixed.
"""

from __future__ import annotations

import json

import numpy as np

LEG_CHAIN = tuple(
    f"{side}_{link}_link"
    for side in ("left", "right")
    for link in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
)
"""Bodies whose rest transform is rewritten, in parent-before-child order."""


def _mat_from_xyzw(quat: np.ndarray) -> np.ndarray:
    """Rotation matrix from an ``(x, y, z, w)`` quaternion -- Isaac Lab 3.0's convention."""
    x, y, z, w = (float(v) for v in quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _wxyz_from_mat(rot: np.ndarray) -> np.ndarray:
    """``(w, x, y, z)`` quaternion from a rotation matrix -- MuJoCo's convention."""
    trace = np.trace(rot)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q = np.array([0.25 / s, (rot[2, 1] - rot[1, 2]) * s, (rot[0, 2] - rot[2, 0]) * s, (rot[1, 0] - rot[0, 1]) * s])
    else:
        i = int(np.argmax(np.diag(rot)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + rot[i, i] - rot[j, j] - rot[k, k])
        q = np.zeros(4)
        q[0] = (rot[k, j] - rot[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (rot[j, i] + rot[i, j]) / s
        q[1 + k] = (rot[k, i] + rot[i, k]) / s
    return q / np.linalg.norm(q)


def load_usd_rest(path: str) -> dict:
    """Load a rest-pose dump: body names plus pelvis-frame positions and ``xyzw`` orientations."""
    with open(path) as handle:
        data = json.load(handle)
    return {
        "names": list(data["body_names"]),
        "pos": np.asarray(data["pos_rest"], dtype=float),
        "quat": np.asarray(data["quat_rest_xyzw"], dtype=float),
    }


def align_legs(model, usd_rest: dict) -> dict[str, float]:
    """Rewrite the leg bodies' ``body_pos`` and ``body_quat`` in place from the USD rest pose.

    Args:
        model: A loaded ``mujoco.MjModel``. Modified in place; the MJCF file is untouched.
        usd_rest: Output of :func:`load_usd_rest`, taken with every joint at zero and the root
            orientation identity, so it is the pure rest kinematics.

    Returns:
        Per-body translation change [m], for logging.

    Raises:
        ValueError: If the dump is missing a body the MuJoCo model has in its leg chain, or if the
            USD's root frame was not identity when it was taken.
    """
    names = usd_rest["names"]
    pos, quat = usd_rest["pos"], usd_rest["quat"]
    pelvis = names.index("pelvis")
    if not np.allclose(pos[pelvis], 0.0, atol=1e-6):
        raise ValueError("rest dump is not expressed relative to the pelvis")

    rot = {name: _mat_from_xyzw(quat[i]) for i, name in enumerate(names)}
    moved = {}
    for body in LEG_CHAIN:
        if body not in names:
            raise ValueError(f"rest dump has no body {body!r}")
        bid = model.body(body).id
        parent = model.body(model.body_parentid[bid]).name
        if parent not in names:
            raise ValueError(f"rest dump has no parent body {parent!r} for {body!r}")

        # Parent-relative rest transform, exactly what MuJoCo stores in body_pos / body_quat.
        rel_pos = rot[parent].T @ (pos[names.index(body)] - pos[names.index(parent)])
        rel_rot = rot[parent].T @ rot[body]
        moved[body] = float(np.linalg.norm(rel_pos - model.body_pos[bid]))
        model.body_pos[bid] = rel_pos
        model.body_quat[bid] = _wxyz_from_mat(rel_rot)
    return moved
