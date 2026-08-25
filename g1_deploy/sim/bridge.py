# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DDS bridge that makes a MuJoCo G1 answer to the same topics as the physical robot.

Modelled on ``gear_sonic/utils/mujoco_sim/unitree_sdk2py_bridge.py`` (the Sonic sim2sim stack),
reduced to what a 29-DoF G1 without Dex3 hands actually needs. It subscribes ``rt/lowcmd`` and
publishes ``rt/lowstate`` and ``rt/secondary_imu`` with the ``unitree_hg`` message types, so the same
controller process drives the simulator and the robot without a code change.

**What is deliberately not published.** Sonic also publishes ``rt/odostate`` -- ground-truth base
position and linear velocity -- and their state processor subscribes to it in sim. That topic does
not exist on a real G1, and a policy that reads it cannot be deployed. This bridge omits it, so
"observable in simulation" and "observable on hardware" are the same set by construction. The
benchmark still gets ground truth, but from the simulator object directly rather than over DDS,
which keeps the evaluation channel out of the policy's input path.
"""

from __future__ import annotations

import threading
import time

import numpy as np
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__IMUState_ as IMUState_default,
)
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__LowCmd_ as LowCmd_default,
)
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__LowState_ as LowState_default,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import IMUState_, LowCmd_, LowState_

NUM_ROBOT_MOTORS = 29
"""``G1_NUM_MOTOR``; must match :data:`g1_deploy_core.NUM_ROBOT_MOTORS`."""

MODE_MACHINE = 5
"""Firmware mode the robot reports in ``rt/lowstate`` and expects echoed in ``rt/lowcmd``.

Sonic hardcodes 5 for a 29-DoF G1. A real robot reports its own value and the controller must echo
whatever it read, which is why :meth:`G1SimBridge.publish_low_state` puts it in every sample.
"""

MODE_PR = 0
"""Ankle coordinate convention: 0 is the PR (pitch/roll) series parameterization the policy assumes."""


class G1SimBridge:
    """Publishes simulated G1 state and holds the most recent ``rt/lowcmd``.

    The command is *held*, not consumed: between two controller updates the simulator keeps applying
    the last one, which is what a motor driver does. :meth:`low_cmd` returns the live message under a
    lock rather than a copy, because the PD evaluation reads it every physics step and copying 29
    motor commands at 200 Hz is pure overhead.
    """

    def __init__(self, num_motors: int = NUM_ROBOT_MOTORS, mode_machine: int = MODE_MACHINE):
        """Initialize the publishers and the ``rt/lowcmd`` subscriber.

        Args:
            num_motors: Motors reported in ``rt/lowstate``.
            mode_machine: Value stamped into every published ``rt/lowstate``.
        """
        self.num_motors = num_motors
        self.mode_machine = mode_machine

        self._cmd_lock = threading.Lock()
        self._low_cmd = LowCmd_default()
        self._cmd_received = False

        self.low_state = LowState_default()
        self.low_state.mode_machine = mode_machine
        self.low_state.mode_pr = MODE_PR
        self._low_state_puber = ChannelPublisher("rt/lowstate", LowState_)
        self._low_state_puber.Init()

        self.torso_imu_state = IMUState_default()
        self._torso_imu_puber = ChannelPublisher("rt/secondary_imu", IMUState_)
        self._torso_imu_puber.Init()

        self._cmd_count = 0
        self._low_cmd_suber = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self._low_cmd_suber.Init(self._on_low_cmd, 1)

    def _on_low_cmd(self, msg: LowCmd_) -> None:
        with self._cmd_lock:
            self._low_cmd = msg
            self._cmd_received = True
            self._cmd_count += 1

    @property
    def cmd_received(self) -> bool:
        """Whether any ``rt/lowcmd`` has arrived yet."""
        with self._cmd_lock:
            return self._cmd_received

    @property
    def cmd_count(self) -> int:
        """Number of ``rt/lowcmd`` samples delivered so far.

        A controller that owns the simulator waits on this before stepping, so a control period never
        runs against the previous period's command just because DDS delivery had not landed yet.
        Cheaper and less invasive than stamping a sequence number into a message field that means
        something else on hardware.
        """
        with self._cmd_lock:
            return self._cmd_count

    def read_cmd(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Snapshot the held command as arrays.

        Returns:
            ``(q, dq, kp, kd, tau_ff)``, each shape ``(num_motors,)``, in [rad], [rad/s],
            [N·m/rad], [N·m·s/rad] and [N·m].
        """
        with self._cmd_lock:
            cmd = self._low_cmd
            q = np.fromiter((cmd.motor_cmd[i].q for i in range(self.num_motors)), float, self.num_motors)
            dq = np.fromiter((cmd.motor_cmd[i].dq for i in range(self.num_motors)), float, self.num_motors)
            kp = np.fromiter((cmd.motor_cmd[i].kp for i in range(self.num_motors)), float, self.num_motors)
            kd = np.fromiter((cmd.motor_cmd[i].kd for i in range(self.num_motors)), float, self.num_motors)
            tau = np.fromiter((cmd.motor_cmd[i].tau for i in range(self.num_motors)), float, self.num_motors)
        return q, dq, kp, kd, tau

    def publish_low_state(self, obs: dict) -> None:
        """Publish one ``rt/lowstate`` and one ``rt/secondary_imu`` sample.

        Args:
            obs: Arrays from :meth:`~dds_sim_env.G1SimEnv.prepare_obs`.
        """
        q, dq, ddq, tau = obs["body_q"], obs["body_dq"], obs["body_ddq"], obs["body_tau_est"]
        for i in range(self.num_motors):
            motor = self.low_state.motor_state[i]
            motor.q = float(q[i])
            motor.dq = float(dq[i])
            motor.ddq = float(ddq[i])
            motor.tau_est = float(tau[i])

        self.low_state.imu_state.quaternion[:] = obs["base_quat_wxyz"]
        self.low_state.imu_state.gyroscope[:] = obs["base_gyro_b"]
        self.low_state.imu_state.accelerometer[:] = obs["base_accel_b"]
        # Sim time in milliseconds. The controller uses this to tell which physics step a sample came
        # from, which is what makes the synchronous mode deterministic instead of a race.
        self.low_state.tick = int(round(obs["time"] * 1e3))
        self._low_state_puber.Write(self.low_state)

        self.torso_imu_state.quaternion[:] = obs["torso_quat_wxyz"]
        self.torso_imu_state.gyroscope[:] = obs["torso_gyro_b"]
        self._torso_imu_puber.Write(self.torso_imu_state)


def await_discovery(link, bridge, env, default_pose, kp, kd, timeout: float = 30.0) -> int:
    """Block until both DDS endpoints have found each other, and return the robot's mode_machine.

    Needed because the default DDS durability is volatile: a sample published before the remote
    reader exists is simply dropped. In the synchronous mode the simulator only publishes when the
    controller tells it to step, and the controller only steps once it has a state, so without this
    the two sit waiting for each other forever.

    Args:
        link: Controller-side endpoints.
        bridge: Simulator-side endpoints.
        env: The simulator, republished from until the controller sees it.
        default_pose: Hold target used for the probe command [rad], shape ``(29,)``.
        kp: Probe gains [N·m/rad].
        kd: Probe gains [N·m·s/rad].
        timeout: Give up after this long [s].

    Returns:
        ``mode_machine`` as reported by the simulator.

    Raises:
        TimeoutError: If either direction never connects.
    """
    deadline = time.monotonic() + timeout
    while link.state is None:
        if time.monotonic() > deadline:
            raise TimeoutError("controller never received rt/lowstate from the in-process simulator")
        bridge.publish_low_state(env.prepare_obs())
        time.sleep(0.01)
    mode_machine = int(link.state.mode_machine)

    # And the other direction: the simulator must be able to hear rt/lowcmd before the first control
    # period, or that period would silently run on a zero-gain command.
    while bridge.cmd_count == 0:
        if time.monotonic() > deadline:
            raise TimeoutError("simulator never received rt/lowcmd from the in-process controller")
        link.send(default_pose, kp, kd, mode_machine)
        time.sleep(0.01)
    return mode_machine
