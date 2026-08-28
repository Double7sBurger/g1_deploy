# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard command source for the policy loop, on the terminal and in the MuJoCo viewer.

The velocity command is otherwise a constant fixed at startup. This makes it live without changing
what the policy reads: the same ``(vx, vy, heading_target)`` triple goes to
:func:`~g1_deploy.core.yaw_rate_from_heading` every control step, so the third element stays the
heading-error controller training generated rather than becoming a raw yaw rate an operator types.

**The command is clamped to the training distribution by default.** ``UniformVelocityCommandCfg`` for
this task draws ``vx`` from ``(0.0, 1.0)`` and ``vy`` from ``(-0.5, 0.5)``, so there is no reverse
gear: a negative ``vx`` is a command the policy has never seen. :data:`VX_LIMITS` can be widened
deliberately, but the default refuses to hand the robot an off-distribution command by accident.

Two input paths feed one :class:`TeleopCommand`, because the two run modes put the operator in
different windows:

* the **terminal**, via :class:`RawTerminal` -- works in every mode, including ``--real`` against
  hardware, and is the only one that exists when the viewer lives in another process.
* the **viewer**, via ``launch_passive(key_callback=...)`` -- only in ``--sim sync --viz``, where the
  controller owns the window. Handy because that is the window you are already looking at.

Both are non-blocking. A control loop that blocks on ``input()`` stops sending ``rt/lowcmd``, and on
hardware the motor watchdog then drops the robot -- the same reason :func:`~g1_deploy.hardware.confirm`
is only ever called before anything moves.
"""

from __future__ import annotations

import os
import select
import sys
import termios
import threading
import tty

import numpy as np

from g1_deploy import remote as rc

VX_LIMITS = (0.0, 1.0)
"""Forward-velocity command range [m/s]; ``UniformVelocityCommandCfg.ranges.lin_vel_x`` for this task.

The lower bound is 0, not ``-1``. Training never sampled a backward command, so walking in reverse is
extrapolation, not a feature that happens to be unbound in the CLI.
"""

VY_LIMITS = (-0.5, 0.5)
"""Lateral-velocity command range [m/s]; ``ranges.lin_vel_y``."""

VX_STEP = 0.1
"""Forward-velocity increment per keypress [m/s]."""

VY_STEP = 0.1
"""Lateral-velocity increment per keypress [m/s]."""

HEADING_STEP = np.deg2rad(15.0)
"""Heading-target increment per keypress [rad].

Deliberately a step on the *target*, not on a yaw rate: the policy reads heading error, and the
turn-rate that error produces is capped by :data:`~g1_deploy.core.YAW_RATE_LIMITS` anyway. Holding
the key down walks the target around; the robot follows at whatever rate the trained controller asks
for.
"""

STICK_TURN_RATE = 1.0
"""How fast a fully deflected right stick sweeps the heading *target* [rad/s].

Sweeping the target rather than commanding a yaw rate keeps the third command element the
heading-error controller training generated. ``unitree_rl_gym`` feeds ``-rx`` straight in as a yaw
rate, which is right for a policy trained with ``heading_command=False``; this task is not one.
"""


HELP = (
    "w/s forward +/-   a/d left/right +/-   q/e turn left/right   "
    "space stop   r face current heading   x quit   ? help"
)
"""One-line key map, printed at startup and on ``?``."""

# Terminal arrow keys arrive as three bytes; fold them onto the letter keys before dispatch.
_ARROWS = {"\x1b[A": "w", "\x1b[B": "s", "\x1b[D": "a", "\x1b[C": "d"}

# GLFW key codes the MuJoCo viewer reports, for the keys that are not plain ASCII letters.
_GLFW_ARROWS = {265: "w", 264: "s", 263: "a", 262: "d", 32: " "}


class TeleopCommand:
    """Live ``(vx, vy, heading)`` command, mutated by keypresses from either input path.

    Thread-safe: the viewer delivers ``key_callback`` on its own UI thread while the control loop
    reads the fields at 50 Hz.
    """

    def __init__(
        self,
        vx: float = 0.0,
        vy: float = 0.0,
        heading: float = 0.0,
        vx_limits: tuple[float, float] = VX_LIMITS,
        vy_limits: tuple[float, float] = VY_LIMITS,
    ):
        """Seed the command with the values the CLI started from.

        Args:
            vx: Initial forward velocity [m/s].
            vy: Initial lateral velocity [m/s].
            heading: Initial world-frame heading target [rad].
            vx_limits: Clamp on ``vx`` [m/s]; the training range by default.
            vy_limits: Clamp on ``vy`` [m/s].
        """
        self._lock = threading.Lock()
        self.vx_limits = vx_limits
        self.vy_limits = vy_limits
        self.vx = float(np.clip(vx, *vx_limits))
        self.vy = float(np.clip(vy, *vy_limits))
        self.heading = float(heading)
        self.changed = True
        """Set on every accepted keypress, cleared by :meth:`status_if_changed`."""

        self.quit = False
        """Set by ``x``; the control loop polls it and exits through its normal shutdown.

        Not a nicety. Under ``mjpython`` -- which macOS requires for a viewer -- the script runs off
        the main thread, so Python delivers ``KeyboardInterrupt`` to mjpython's event loop and the
        control loop never sees it. Ctrl-C is therefore not a dependable stop there, and closing the
        viewer window or pressing ``x`` is.
        """

        self._face_current = False

    def as_array(self, quat_wxyz: np.ndarray) -> np.ndarray:
        """Build the policy's command vector from the live values.

        Args:
            quat_wxyz: Pelvis orientation from the IMU as ``(w, x, y, z)``; the yaw element is
                derived from the error against :attr:`heading`, exactly as in training.

        Returns:
            ``(vx, vy, yaw_rate)``, shape ``(3,)``, float32.
        """
        from g1_deploy import core

        with self._lock:
            if self._face_current:
                # 'r' means "stop turning": adopt the heading the robot is actually at, which zeroes
                # the error the policy is steering on.
                self.heading = core.heading_from_quat_wxyz(quat_wxyz)
                self._face_current = False
            vx, vy, heading = self.vx, self.vy, self.heading
        return np.array([vx, vy, core.yaw_rate_from_heading(heading, quat_wxyz)], dtype=np.float32)

    def face(self, heading: float) -> None:
        """Set the heading target outright, e.g. to the robot's measured heading before engaging.

        Args:
            heading: World-frame yaw [rad].
        """
        with self._lock:
            self.heading = float(heading)
            self._face_current = False
            self.changed = True

    def apply_sticks(self, sticks: dict[str, float], dt: float) -> None:
        """Drive the command from the remote's analog sticks.

        Sign convention taken from Unitree's own G1 deployment,
        ``unitree_rl_gym/deploy/deploy_real/deploy_real.py``::

            cmd[0] = ly        cmd[1] = lx * -1        cmd[2] = rx * -1

        so left stick forward is +vx, left stick left is +vy, right stick left turns left. Verify the
        physical directions with ``scripts/check_robot.py`` before letting this drive a robot -- it
        prints the live axes and costs nothing.

        **A centred stick holds the last command; it does not zero it.** A self-centring stick
        mapped straight onto velocity would command ``vx = 0`` the moment you let go, and this
        checkpoint cannot stand -- measured on the hardware code path it falls in about 1.4 s at zero
        command. So releasing would be a fall, and engaging without the stick already pushed would be
        a fall at step 0. Holding instead means the robot keeps doing the last thing it was told
        until told otherwise.

        .. attention::
            The consequence is that **letting go is not a stop**. The stop is
            :data:`~g1_deploy.remote.ABORT_COMBOS`, and the hoist. Pull the left stick back to slow
            down; it reaches ``vx = 0``, which is a fall, deliberately reachable but not the resting
            state.

        Args:
            sticks: Output of :func:`~g1_deploy.remote.parse_sticks`; ignored if empty.
            dt: Control period [s], used to integrate the heading target.
        """
        if not sticks:
            return
        # Deadzone applied here as well as in parse_sticks: whether a stick counts as centred
        # decides whether the robot keeps walking, which is too important to inherit from whatever
        # the caller happened to pass in.
        def live(axis: str) -> float:
            value = sticks.get(axis, 0.0)
            return 0.0 if abs(value) < rc.STICK_DEADZONE else value

        ly, lx, rx = live("ly"), live("lx"), live("rx")
        with self._lock:
            vx = float(np.clip(ly, *self.vx_limits)) if ly else self.vx
            vy = float(np.clip(-lx, *self.vy_limits)) if lx else self.vy
            heading = self.heading - rx * STICK_TURN_RATE * dt
            if (vx, vy, heading) != (self.vx, self.vy, self.heading):
                self.vx, self.vy, self.heading = vx, vy, heading
                self.changed = True

    def handle(self, key: str) -> None:
        """Apply one keypress. Unknown keys are ignored.

        Args:
            key: A single character, already lowercased.
        """
        with self._lock:
            if key == "w":
                self.vx = float(np.clip(self.vx + VX_STEP, *self.vx_limits))
            elif key == "s":
                self.vx = float(np.clip(self.vx - VX_STEP, *self.vx_limits))
            elif key == "a":
                self.vy = float(np.clip(self.vy + VY_STEP, *self.vy_limits))
            elif key == "d":
                self.vy = float(np.clip(self.vy - VY_STEP, *self.vy_limits))
            elif key == "q":
                self.heading += HEADING_STEP
            elif key == "e":
                self.heading -= HEADING_STEP
            elif key == " ":
                self.vx = self.vy = 0.0
            elif key == "r":
                self._face_current = True
            elif key == "x":
                self.quit = True
            elif key == "?":
                pass
            else:
                return
            self.changed = True

    def viewer_key_callback(self, keycode: int) -> None:
        """``mujoco.viewer.launch_passive`` key hook.

        Args:
            keycode: GLFW key code. Printable ASCII arrives as the uppercase letter's code.
        """
        key = _GLFW_ARROWS.get(keycode)
        if key is None and 0 < keycode < 128:
            key = chr(keycode).lower()
        if key is not None:
            self.handle(key)

    def status(self) -> str:
        """Human-readable one-liner for the current command."""
        with self._lock:
            return f"cmd vx={self.vx:+.2f} vy={self.vy:+.2f} m/s  heading={np.rad2deg(self.heading):+.0f} deg"

    def status_if_changed(self) -> str | None:
        """Return :meth:`status` once per change, so a 50 Hz loop does not flood the terminal."""
        with self._lock:
            if not self.changed:
                return None
            self.changed = False
        return self.status()


class RawTerminal:
    """Context manager putting ``stdin`` in cbreak mode so single keys arrive without Enter.

    ``tty.setcbreak`` rather than ``setraw`` on purpose: cbreak leaves ``ISIG`` on, so Ctrl-C still
    raises :class:`KeyboardInterrupt` and the loop's ``finally`` still damps the robot down.

    A non-tty ``stdin`` (piped input, a launcher without a console) is not an error -- the context
    becomes a no-op and :meth:`poll` returns nothing, leaving the viewer as the only input path.

    Both mode changes use ``TCSANOW``. ``tty.setcbreak`` defaults to ``TCSAFLUSH``, which blocks
    until the terminal's *output* queue has drained -- a wait that never ends if nothing is reading
    the other end, and one that would land between the ramp and the first ``rt/lowcmd``. Nothing
    here needs the queues flushed; the mode change is all that is wanted.
    """

    def __init__(self, stream=None):
        """Args:
        stream: Input stream; ``sys.stdin`` by default.
        """
        self.stream = stream if stream is not None else sys.stdin
        self.enabled = False
        self._saved = None

    def __enter__(self) -> RawTerminal:
        try:
            fd = self.stream.fileno()
            if os.isatty(fd):
                self._saved = termios.tcgetattr(fd)
                tty.setcbreak(fd, termios.TCSANOW)
                self.enabled = True
        except (OSError, ValueError, termios.error):
            self.enabled = False
        return self

    def __exit__(self, *exc) -> None:
        if self._saved is not None:
            termios.tcsetattr(self.stream.fileno(), termios.TCSANOW, self._saved)
            self._saved = None
        self.enabled = False

    def poll(self) -> list[str]:
        """Drain whatever has been typed since the last call, without ever blocking.

        Returns:
            Lowercased single characters, arrow keys folded onto ``w``/``a``/``s``/``d``.
        """
        if not self.enabled:
            return []
        fd = self.stream.fileno()
        chunk = ""
        while select.select([fd], [], [], 0)[0]:
            data = os.read(fd, 64)
            if not data:
                break
            chunk += data.decode(errors="ignore")
        for seq, key in _ARROWS.items():
            chunk = chunk.replace(seq, key)
        return [c.lower() for c in chunk]
