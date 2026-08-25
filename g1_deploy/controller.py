# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Controller half of the DDS loop: reads ``rt/lowstate``, writes ``rt/lowcmd``.

This is the piece that is identical against the simulator and against the robot. It knows nothing
about MuJoCo; it speaks only ``unitree_hg`` messages, so :mod:`run_policy_loop` can point it at
:mod:`dds_sim_env`, at ``unitree_mujoco``, or at a real G1 by changing a domain id.

The observation it builds is assembled by :class:`~g1_deploy_core.G1PolicyRunner` from joint
encoders and the pelvis IMU only -- everything in ``rt/lowstate``, nothing privileged.
"""

from __future__ import annotations

import threading
import time

import numpy as np
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

MOTOR_ENABLE = 1
""":cpp:`motor_cmd().mode() = 1` in the SDK's own G1 example; 0 disables the motor."""

MODE_PR = 0
"""Series ankle parameterization (pitch/roll), matching what the policy was trained against."""

_STATE_QUEUE_LEN = 128
"""``rt/lowstate`` subscriber queue depth; see :meth:`G1ControlLink.__init__`."""


class G1ControlLink:
    """The controller's two DDS endpoints, plus the handshakes the synchronous mode needs."""

    def __init__(self, num_motors: int = 29):
        """Create the ``rt/lowstate`` subscriber and the ``rt/lowcmd`` publisher.

        Args:
            num_motors: Motors to read and command.
        """
        self.num_motors = num_motors
        self._lock = threading.Lock()
        self._state: LowState_ | None = None
        self._crc = CRC()
        self._pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self._pub.Init()
        self._sub = ChannelSubscriber("rt/lowstate", LowState_)
        # The queue must be deep enough to hold one control period's burst. The simulator publishes
        # once per physics step -- four samples arriving back to back -- and the SDK's reader thread
        # drains asynchronously, so a length-1 queue drops the newest two: the controller then waits
        # forever for a tick that was published and thrown away.
        self._sub.Init(self._on_state, _STATE_QUEUE_LEN)

    def _on_state(self, msg: LowState_) -> None:
        with self._lock:
            self._state = msg

    @property
    def state(self) -> LowState_ | None:
        """Most recent ``rt/lowstate``, or ``None`` before the first sample."""
        with self._lock:
            return self._state

    def wait_for_state(self, timeout: float = 30.0) -> LowState_:
        """Block until the first ``rt/lowstate`` arrives.

        Raises:
            TimeoutError: If nothing publishes in time -- usually a domain id or interface mismatch,
                or the simulator is not running.
        """
        deadline = time.monotonic() + timeout
        while True:
            state = self.state
            if state is not None:
                return state
            if time.monotonic() > deadline:
                raise TimeoutError(f"no rt/lowstate within {timeout:.0f}s")
            time.sleep(0.002)

    def wait_for_tick(self, tick_ms: int, timeout: float = 2.0) -> LowState_:
        """Block until a state stamped at or after ``tick_ms`` arrives.

        This is what removes the race from the synchronous mode. The simulator stamps every sample
        with its own clock, so waiting on the tick means the controller provably sees the state at
        the control boundary it just advanced to -- not one physics step behind, and not whatever
        DDS happened to have delivered. Without it the loop still runs, but the observation lags by
        a jittery 0-3 physics steps and no two runs agree.

        Args:
            tick_ms: Simulator time to wait for [ms].
            timeout: Give up after this long [s].

        Returns:
            The first state at or after ``tick_ms``.

        Raises:
            TimeoutError: If no such sample arrives in time.
        """
        deadline = time.monotonic() + timeout
        while True:
            state = self.state
            if state is not None and int(state.tick) >= tick_ms:
                return state
            if time.monotonic() > deadline:
                got = None if state is None else int(state.tick)
                raise TimeoutError(f"no rt/lowstate with tick >= {tick_ms} within {timeout:.1f}s (latest {got})")
            time.sleep(0.0002)

    def send(self, q_target: np.ndarray, kp: np.ndarray, kd: np.ndarray, mode_machine: int) -> None:
        """Publish one ``rt/lowcmd``.

        Args:
            q_target: Position targets in motor order [rad], shape ``(num_motors,)``.
            kp: Proportional gains [N·m/rad], shape ``(num_motors,)``.
            kd: Derivative gains [N·m·s/rad], shape ``(num_motors,)``.
            mode_machine: Echoed from ``rt/lowstate``; a real G1 rejects a command that disagrees.
        """
        cmd = unitree_hg_msg_dds__LowCmd_()
        cmd.mode_pr = MODE_PR
        cmd.mode_machine = mode_machine
        for i in range(self.num_motors):
            motor = cmd.motor_cmd[i]
            motor.mode = MOTOR_ENABLE
            motor.q = float(q_target[i])
            motor.dq = 0.0
            motor.kp = float(kp[i])
            motor.kd = float(kd[i])
            motor.tau = 0.0
        cmd.crc = self._crc.Crc(cmd)
        self._pub.Write(cmd)


def read_joint_state(state: LowState_, num_motors: int = 29) -> tuple[np.ndarray, np.ndarray]:
    """Joint positions and velocities from a ``rt/lowstate`` sample, in motor order."""
    q = np.fromiter((state.motor_state[i].q for i in range(num_motors)), np.float32, num_motors)
    dq = np.fromiter((state.motor_state[i].dq for i in range(num_motors)), np.float32, num_motors)
    return q, dq
