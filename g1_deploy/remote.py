# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read the G1's remote controller out of ``rt/lowstate``, so the operator can abort.

**The G1 has no hardware emergency stop.** Unitree's documented emergency stop is a button
combination on the stock remote -- ``L2 + B`` on Motion Control 8.2.0.0 and later -- which asks the
*factory* controller to drop into damping. Unitree's docs claim that combo "remains effective even in
debug mode", but once a custom program owns ``rt/lowcmd`` nothing in their published code depends on
that, and neither should this. Unitree's own RL deployment handles the combo *inside the user
program*: ``unitree_rl_lab`` maps ``LT + B`` to its Passive (kp=0, kd=3) state from every state.

So does this module. Every ``rt/lowstate`` carries 40 bytes of remote state; the control loop parses
them each step and bails to :func:`~g1_deploy.hardware.damp_down` when it sees an abort combo. That
costs one struct unpack per control period and is the only stop that is guaranteed to be under your
own program's control.

The byte layout is transcribed from Unitree's own parser,
``unitree_rl_gym/deploy/deploy_real/common/remote_controller.py``: a 16-bit button mask at offset 2,
then four little-endian floats for the sticks.

.. attention::
    **The remote's mapping changed at Motion Control 8.2.0.0.** Damping is ``L2 + B`` on 8.2.0.0 and
    later, and ``L1 + A`` on earlier firmware -- where ``L2 + B`` means nothing and ``L1 + B`` is
    *zero torque*, which drops a hanging robot with no damping at all. :data:`ABORT_COMBOS`
    deliberately accepts both mappings plus ``select``, because on any given firmware the combos that
    are not the abort are also not things an operator presses mid-run, and guessing the firmware
    wrong must not be what decides whether the stop works.
"""

from __future__ import annotations

import struct

BUTTON_BITS = {
    "R1": 0,
    "L1": 1,
    "start": 2,
    "select": 3,
    "R2": 4,
    "L2": 5,
    "F1": 6,
    "F2": 7,
    "A": 8,
    "B": 9,
    "X": 10,
    "Y": 11,
    "up": 12,
    "right": 13,
    "down": 14,
    "left": 15,
}
"""Bit index of each button in the mask at ``wireless_remote[2:4]``.

Matches ``KeyMap`` in ``unitree_rl_gym/deploy/deploy_real/common/remote_controller.py`` and the
layout on Unitree's `Remote Control Data
<https://support.unitree.com/home/en/G1_developer/remote_control_data>`_ page.
"""

ABORT_COMBOS = (
    ("L2", "B"),
    ("L1", "A"),
    ("select",),
)
"""Combinations that abort the run and damp the robot down.

* ``L2 + B`` -- the documented emergency stop on Motion Control >= 8.2.0.0.
* ``L1 + A`` -- the same function on firmware older than 8.2.0.0, where the mapping was L1-based.
* ``select`` -- what ``unitree_rl_gym``'s G1 example uses to leave its control loop.

All three are accepted regardless of firmware. The cost of a false positive is a damped robot on a
hoist; the cost of a false negative is a 35 kg humanoid nobody can stop.
"""


STICK_OFFSETS = {"lx": 4, "rx": 8, "ry": 12, "ly": 20}
"""Byte offset of each analog stick's float in ``wireless_remote``.

Note ``ly`` at 20 rather than 16 -- the layout is not four contiguous floats. Transcribed from
``RemoteController.set`` in ``unitree_rl_gym/deploy/deploy_real/common/remote_controller.py``.
"""

STICK_DEADZONE = 0.08
"""Deflection below which a stick reads as centred. Sticks rest slightly off zero and a locomotion
command that drifts by itself is not one you want on a robot."""


def parse_sticks(wireless_remote) -> dict[str, float]:
    """Decode the four analog stick axes.

    Args:
        wireless_remote: The 40-byte ``rt/lowstate`` field.

    Returns:
        ``lx``, ``ly``, ``rx``, ``ry`` in roughly ``[-1, 1]``, deadzoned to exactly 0 near centre.
        Empty if the field is too short.
    """
    data = bytes(bytearray(wireless_remote))
    if len(data) < 24:
        return {}
    out = {}
    for name, off in STICK_OFFSETS.items():
        value = struct.unpack("<f", data[off : off + 4])[0]
        out[name] = 0.0 if abs(value) < STICK_DEADZONE else float(value)
    return out


def parse_buttons(wireless_remote) -> dict[str, bool]:
    """Decode the button mask from a ``rt/lowstate`` ``wireless_remote`` field.

    Args:
        wireless_remote: The 40-byte field, as a list of ints or a bytes-like object.

    Returns:
        Button name -> pressed. Empty if the field is too short to decode, which is what a simulator
        that does not populate it looks like.
    """
    data = bytes(bytearray(wireless_remote))
    if len(data) < 4:
        return {}
    mask = struct.unpack("<H", data[2:4])[0]
    return {name: bool(mask >> bit & 1) for name, bit in BUTTON_BITS.items()}


def abort_pressed(wireless_remote) -> str | None:
    """Return the name of the abort combo being held, or ``None``.

    Args:
        wireless_remote: The 40-byte ``rt/lowstate`` field.

    Returns:
        Something like ``"L2+B"`` for logging, or ``None`` if no abort combo is down.
    """
    buttons = parse_buttons(wireless_remote)
    if not buttons:
        return None
    for combo in ABORT_COMBOS:
        if all(buttons.get(name) for name in combo):
            return "+".join(combo)
    return None
