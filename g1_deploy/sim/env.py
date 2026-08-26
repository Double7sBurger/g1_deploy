# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MuJoCo G1 that answers to ``rt/lowcmd`` and publishes ``rt/lowstate``.

Modelled on ``DefaultEnv`` in ``gear_sonic/utils/mujoco_sim/base_sim.py``. The two things worth
copying from Sonic are here and the rest is dropped:

* **PD runs at the physics rate from a held command.** ``sim_step`` publishes state, evaluates
  ``tau = tau_ff + kp (q* - q) + kd (dq* - dq)`` against the *latest* ``rt/lowcmd``, clips, and steps
  once. Holding one torque across a whole control period freezes the damping term and rings.
* **The controller can drive the stepping.** :meth:`step_control` is Sonic's ``step_simulator``: it
  advances exactly ``decimation`` physics steps. Running the simulator free and letting DDS decide
  when the controller sees a state adds 12-16 ms of jitter that has nothing to do with the policy,
  and it is not reproducible run to run.

**The physics timestep is 0.005 s on purpose.** That is Isaac Lab's ``sim.dt`` for this task, and
with a decimation of 4 it lands the policy at the trained 50 Hz. The stock ``unitree_mujoco`` runs
0.002 s, which is a different integrator trajectory for the same model and a gratuitous difference
from training.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

SIM_DT = 0.002
"""Physics timestep [s]. The MJCF's own value, kept because it measures better.

Isaac Lab runs this task at ``sim.dt = 0.005`` and matching that number is the tempting choice, but
the two are different integrators and equal timesteps are not equal accuracy. On the 45-episode hold
suite the same checkpoint scores 75.6% success and 0.364 m/s tracking error at 0.002 s against 53.3%
and 0.448 m/s at 0.005 s. Only the policy period has to match training, and ``SIM_DT * DECIMATION``
is 0.02 s either way.

Both timesteps are stable. An earlier version of this file claimed 0.005 blew the integrator up, on
the evidence that the policy fell at 2.02 s there and walked 15 s at 0.002. That was a bug in the
handshake below, not in the physics -- 0.005 survives the full run once the controller stops seeding
its observation history from a pre-reset sample -- but it does cost real accuracy.
"""

DECIMATION = 10
"""Physics steps per control step; ``SIM_DT * DECIMATION`` is the trained 0.02 s policy period."""

DEFAULT_XML = Path.home() / "workspace/unitree_mujoco/unitree_robots/g1/scene_29dof.xml"
"""Flat-ground 29-DoF G1 scene.

Deliberately not ``scene.xml``: ``unitree_mujoco``'s terrain_tool rewrites that file in place with a
stairs/ramp/boulder course, so survival measured against it silently mixes locomotion with obstacle
avoidance.
"""

TRUNK_BODIES = ("pelvis", "waist_yaw_link", "waist_roll_link", "torso_link")
"""The trunk, the one segment where the MJCF and the USD disagree materially: 13.702 kg against
10.381 kg at the same pose, while the legs agree to -4% and the arms to +3%."""

USD_REST_JSON = Path(__file__).resolve().parent.parent / "data" / "g1_usd_rest.json"
"""Shipped USD rest-pose dump for :func:`~g1_deploy.sim.align.align_legs`."""

GRAVITY = np.array([0.0, 0.0, -9.81])
"""World-frame gravity [m/s^2], used to turn linear acceleration into IMU specific force."""

# Joint-name fragments that identify a driven body joint, in MuJoCo model order. The official
# g1_29dof MJCF lists them in exactly G1JointIndex order, which the constructor asserts.
_BODY_JOINT_PARTS = ("hip", "knee", "ankle", "waist", "shoulder", "elbow", "wrist")


class G1SimEnv:
    """MuJoCo G1 wired to a :class:`~dds_sim_bridge.G1SimBridge`."""

    def __init__(
        self,
        xml_path: str,
        bridge,
        *,
        sim_dt: float = SIM_DT,
        decimation: int = DECIMATION,
        onscreen: bool = False,
        key_callback=None,
        contact_timeconst: float = 0.0,
        body_mass_scale: dict[str, float] | None = None,
        integrator: str | None = None,
        joint_frictionloss: float | None = None,
        joint_damping: float | None = None,
        align_legs_to_usd: str | None = None,
        command_delay_steps: int = 0,
    ):
        """Load the model and place the robot on the ground in its default pose.

        Args:
            xml_path: G1 MJCF scene. The file is never modified; every override here is applied to
                the loaded model in memory.
            bridge: The DDS bridge that supplies commands and receives state.
            sim_dt: Physics timestep [s].
            decimation: Physics steps per :meth:`step_control` call.
            onscreen: Open a passive viewer tracking the pelvis.
            key_callback: Called with a GLFW key code on every viewer keypress. Only meaningful with
                ``onscreen``; the viewer owns the window and therefore the keyboard.
            contact_timeconst: Override every geom's ``solref`` time constant [s]; 0 keeps the
                model's own value. This is a system-identification knob, not a model edit.
            body_mass_scale: Per-body mass and inertia multipliers, for system identification
                against the USD the policy trained on. Unknown body names raise.
            integrator: ``"euler"``, ``"rk4"``, ``"implicit"`` or ``"implicitfast"``; ``None`` keeps
                the model's own. Only worth changing together with a larger ``sim_dt``.
            joint_frictionloss: Dry (Coulomb) friction on every driven joint [N·m]; ``None`` keeps
                the MJCF's 0.2. The USD assumes frictionless joints, so a nominal Isaac Lab robot has
                0 here and training randomized it over 0.0-0.3 -- the MJCF sits inside that band but
                not at its centre.
            joint_damping: Viscous damping on every driven joint [N·m·s/rad]; ``None`` keeps the
                MJCF's 0.05. Isaac Lab's legs carry 0, because their PD is applied as an external
                torque rather than folded into the solver.
            command_delay_steps: Hold each ``rt/lowcmd`` for this many physics steps before the PD
                acts on it, emulating bus and transport latency. 0 is the synchronous ideal and is
                what the sim-to-sim comparison uses; a real G1 is never 0. At ``sim_dt = 0.002``
                each step is 2 ms, so 6-8 steps brackets the 12-16 ms measured over DDS.
            align_legs_to_usd: Path to a USD rest-pose dump (``deploy/g1_usd_rest.json``); ``None``
                keeps the MJCF's own leg kinematics. The two descriptions disagree about the leg --
                at identical joint angles the MJCF's ankle sits 5.6 cm below the USD's, so the same
                joint command stands the MuJoCo robot about 6 cm taller. Passing the dump rewrites
                the leg rest transforms and reproduces the USD's forward kinematics exactly (0.000 mm
                across six joint configurations, four of them held out). See
                :mod:`kinematics_align`.

        Raises:
            ValueError: If the model's driven joints are not the expected 29, or a scaled body is
                absent.
        """
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model.opt.timestep = sim_dt
        if integrator is not None:
            self.model.opt.integrator = {
                "euler": mujoco.mjtIntegrator.mjINT_EULER,
                "rk4": mujoco.mjtIntegrator.mjINT_RK4,
                "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
                "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
            }[integrator]
        if contact_timeconst > 0:
            self.model.geom_solref[:, 0] = contact_timeconst
        for name, factor in (body_mass_scale or {}).items():
            body = self.model.body(name)  # raises KeyError-like on an unknown name
            self.model.body_mass[body.id] *= factor
            self.model.body_inertia[body.id] *= factor

        self.data = mujoco.MjData(self.model)
        self.command_delay_steps = command_delay_steps
        self._delay_queue: list[tuple] = []
        self.sim_dt = sim_dt
        self.decimation = decimation
        self.bridge = bridge

        self.body_joint_ids = np.array(
            [
                j
                for j in range(self.model.njnt)
                if any(part in (self.model.joint(j).name or "") for part in _BODY_JOINT_PARTS)
            ],
            dtype=np.int32,
        )
        if len(self.body_joint_ids) != bridge.num_motors:
            raise ValueError(
                f"{xml_path} exposes {len(self.body_joint_ids)} driven joints, expected {bridge.num_motors}"
            )
        # Free joint first: 7 qpos / 6 qvel of root before the hinge DOFs.
        self.qpos_adr = self.model.jnt_qposadr[self.body_joint_ids]
        self.qvel_adr = self.model.jnt_dofadr[self.body_joint_ids]
        if align_legs_to_usd is not None:
            from g1_deploy.sim.align import align_legs, load_usd_rest

            align_legs(self.model, load_usd_rest(align_legs_to_usd))
        if joint_frictionloss is not None:
            self.model.dof_frictionloss[self.qvel_adr] = joint_frictionloss
        if joint_damping is not None:
            self.model.dof_damping[self.qvel_adr] = joint_damping
        self.effort_limit = np.abs(self.model.actuator_ctrlrange[:, 1])

        self.pelvis_id = self.model.body("pelvis").id
        self.torso_id = self.model.body("torso_link").id
        self.viewer = None
        if onscreen:
            self.viewer = mujoco.viewer.launch_passive(
                self.model, self.data, show_left_ui=False, show_right_ui=False, key_callback=key_callback
            )
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.viewer.cam.trackbodyid = self.pelvis_id
            self.viewer.cam.distance = 2.5
            self.viewer.cam.azimuth = 130.0
            self.viewer.cam.elevation = -20.0

    @property
    def joint_names(self) -> list[str]:
        """Driven joint names in ``rt/lowstate`` motor order."""
        return [self.model.joint(j).name for j in self.body_joint_ids]

    def reset(self, joint_pos: np.ndarray, start_time: float | None = None) -> float:
        """Reset to ``joint_pos`` with the lowest robot geom resting on the floor.

        The MJCF's own spawn height assumes straight legs while the trained default pose is a slight
        crouch, so reusing it drops the robot ~10 cm and hands the policy an impact it never saw.

        Args:
            joint_pos: Target joint positions [rad] in motor order, shape ``(29,)``.
            start_time: Simulation time to resume from [s], instead of the 0 that ``mj_resetData``
                writes. A benchmark that replays many episodes over one DDS link needs the clock to
                keep increasing across resets, because the controller identifies samples by ``tick``
                and a clock that restarts makes the previous episode's last sample look current.

        Returns:
            The resulting pelvis height [m].
        """
        mujoco.mj_resetData(self.model, self.data)
        if start_time is not None:
            self.data.time = start_time
        self.data.qpos[self.qpos_adr] = joint_pos
        robot_geoms = [
            g
            for g in range(self.model.ngeom)
            if self.model.geom_bodyid[g] != 0 and self.model.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE
        ]
        self.data.qpos[2] = 1.0
        mujoco.mj_forward(self.model, self.data)
        self.data.qpos[2] = 1.0 - float(self.data.geom_xpos[robot_geoms, 2].min()) + 0.002
        mujoco.mj_forward(self.model, self.data)
        return float(self.data.qpos[2])

    def prepare_obs(self) -> dict:
        """Collect everything :meth:`~dds_sim_bridge.G1SimBridge.publish_low_state` needs.

        Every field corresponds to something a real G1 reports: joint encoders, the pelvis IMU, and
        the torso IMU. Ground truth base position and velocity are deliberately absent.
        """
        quat = self.data.qpos[3:7].copy()  # MuJoCo free joint: (w, x, y, z)
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, quat)
        rot = rot.reshape(3, 3)
        # A real accelerometer measures specific force, not coordinate acceleration: at rest it reads
        # +g, not zero. Sonic publishes raw qacc here, which reads zero at rest and would silently
        # mislead anything that ever starts using it.
        accel_b = rot.T @ (np.asarray(self.data.qacc[0:3]) - GRAVITY)

        torso_vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.torso_id, torso_vel, 1)

        return {
            "body_q": self.data.qpos[self.qpos_adr].copy(),
            "body_dq": self.data.qvel[self.qvel_adr].copy(),
            "body_ddq": self.data.qacc[self.qvel_adr].copy(),
            "body_tau_est": self.data.actuator_force.copy(),
            "base_quat_wxyz": quat,
            "base_gyro_b": self.data.qvel[3:6].copy(),  # free joint angular velocity is body frame
            "base_accel_b": accel_b,
            "torso_quat_wxyz": self.data.xquat[self.torso_id].copy(),
            "torso_gyro_b": torso_vel[:3].copy(),  # mj_objectVelocity returns [ang, lin]
            "time": float(self.data.time),
        }

    def publish_only(self) -> None:
        """Publish the current state without advancing physics.

        A free-running server needs this while it waits for a controller. Holding the default pose
        with a position PD instead does not work and is not a tuning problem: the trained default
        pose is a slight crouch that the shipped gains do not statically support, and *both*
        simulators drop the robot from it in about 1.5 s -- MuJoCo at 1.46 s, Isaac Lab at a
        comparable time with its contact termination firing on the way. Freezing is the honest
        equivalent of a robot supported on a stand until the operator engages the controller.
        """
        self.bridge.publish_low_state(self.prepare_obs())

    def compute_torques(self) -> np.ndarray:
        """PD torque from the held ``rt/lowcmd``, clipped to the model's own ``ctrlrange``."""
        if not self.bridge.cmd_received:
            return np.zeros(self.model.nu)
        snapshot = self.bridge.read_cmd()
        if self.command_delay_steps > 0:
            # A FIFO of held commands: the PD acts on the one from command_delay_steps ago, which is
            # what a bus round trip does to a real motor driver.
            self._delay_queue.append(snapshot)
            if len(self._delay_queue) > self.command_delay_steps:
                snapshot = self._delay_queue.pop(0)
        q_des, dq_des, kp, kd, tau_ff = snapshot
        tau = tau_ff + kp * (q_des - self.data.qpos[self.qpos_adr]) + kd * (dq_des - self.data.qvel[self.qvel_adr])
        return np.clip(tau, -self.effort_limit, self.effort_limit)

    def sim_step(self) -> None:
        """Apply the held command, advance one physics step, then publish the resulting state.

        Sonic publishes *before* stepping, which leaves the newest sample one physics step behind the
        control boundary. Isaac Lab builds its observation after all ``decimation`` physics steps, so
        publishing after the step is what makes the two agree; it is also what a real robot does,
        since its encoders report where the joints are now, not where they were 5 ms ago.
        """
        self.data.ctrl[:] = self.compute_torques()
        mujoco.mj_step(self.model, self.data)
        self.bridge.publish_low_state(self.prepare_obs())

    def step_control(self) -> None:
        """Advance one control period: :attr:`decimation` physics steps, then refresh the viewer."""
        for _ in range(self.decimation):
            self.sim_step()
        if self.viewer is not None:
            self.viewer.sync()

    def apply_hoist(
        self,
        target_z: float,
        kp: float = 4000.0,
        kd: float = 400.0,
        kp_ang: float = 400.0,
        kd_ang: float = 40.0,
    ) -> None:
        """Hold the pelvis up and roughly level, like a strap on a gantry.

        The rehearsal equivalent of hanging the robot up, which is how a G1 is brought up on
        hardware -- and it is not optional for rehearsing a realistic ramp: the trained default pose
        is not statically stable at these gains, so a free-standing robot collapses in about 1.5 s
        and any ramp longer than that ends on the floor.

        Vertical force plus an orientation restoring torque, following ``ElasticBand`` in Sonic's
        MuJoCo stack. Lifting alone is not enough -- a pure vertical force at the pelvis leaves the
        robot free to rotate and it ends up inverted, which no real strap allows.

        Args:
            target_z: Pelvis height to hold [m].
            kp: Strap stiffness [N/m].
            kd: Strap damping [N·s/m].
            kp_ang: Orientation restoring stiffness [N·m/rad].
            kd_ang: Orientation damping [N·m·s/rad].
        """
        force = kp * (target_z - float(self.data.qpos[2])) - kd * float(self.data.qvel[2])
        rotvec = np.zeros(3)
        mujoco.mju_quat2Vel(rotvec, np.asarray(self.data.qpos[3:7]), 1.0)
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, self.data.qpos[3:7])
        omega_world = rot.reshape(3, 3) @ np.asarray(self.data.qvel[3:6])
        self.data.xfrc_applied[self.pelvis_id, :3] = (0.0, 0.0, max(0.0, force))
        self.data.xfrc_applied[self.pelvis_id, 3:] = -kp_ang * rotvec - kd_ang * omega_world

    def release_hoist(self) -> None:
        """Stop pulling on the pelvis."""
        self.data.xfrc_applied[self.pelvis_id, :] = 0.0

    @property
    def gravity_z(self) -> float:
        """Projected gravity z in the body frame; below -0.9 is upright, above -0.7 is a fall."""
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, self.data.qpos[3:7])
        return float((rot.reshape(3, 3).T @ np.array([0.0, 0.0, -1.0]))[2])

    def base_lin_vel_b(self) -> np.ndarray:
        """Ground-truth base linear velocity in the body frame [m/s]. Evaluation only.

        Not published over DDS and not available to the policy -- a real G1 has no such measurement.
        The benchmark reads it straight off this object so the evaluation channel never touches the
        observation path.
        """
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, self.data.qpos[3:7])
        return rot.reshape(3, 3).T @ np.asarray(self.data.qvel[0:3])

    def close(self) -> None:
        """Close the viewer if one is open."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
