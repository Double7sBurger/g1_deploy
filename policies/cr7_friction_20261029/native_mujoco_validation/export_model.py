"""Export the frozen official USD physics model for standalone native MuJoCo."""

import argparse
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import torch
import warp as wp

wp.config.enable_backward = False
from isaaclab.app import launch_simulation
from isaaclab_newton.physics.newton_manager import NewtonManager
from isaaclab_newton.physics.mjwarp_manager import NewtonMJWarpManager

ROOT = Path(__file__).resolve().parent
PACKAGES = Path('/home/henry/workspace/g1_deploy/policies/cr7_friction_20261029')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['implicit200', 'explicit400'], required=True)
    mode = parser.parse_args().mode
    if mode == 'implicit200':
        from motion_tracking.tasks.motion_tracking.motion_tracking_allfive_env_cfg import MotionTrackingAllFiveEnvCfg
        cfg = MotionTrackingAllFiveEnvCfg()
        package = PACKAGES / 'fixed/mu_0.7'
    else:
        from motion_tracking.tasks.motion_tracking.motion_tracking_sysid_env_cfg import MotionTrackingSysIdEnvCfg
        cfg = MotionTrackingSysIdEnvCfg()
        package = PACKAGES / 'dr_shared/rank1_s45_i20000'
    contract = json.loads((package / 'contract.json').read_text())
    cfg.scene.num_envs = 1
    cfg.seed = 20261029
    cfg.sim.physics.solver_cfg.use_mujoco_cpu = True
    cfg.sim.physics.use_cuda_graph = False
    cfg.commands.motion.motion_file = str(package / 'motion.npz')
    cfg.events.add_base_mass = None
    cfg.events.push_robot = None
    cfg.events.physics_material = None
    cfg.scene.terrain.physics_material.static_friction = 1.
    cfg.scene.terrain.physics_material.dynamic_friction = 1.
    cfg.observations.policy.enable_corruption = False
    cfg.scene.self_collision = None
    cfg.scene.foot_ground = None
    cfg.rewards.self_collisions = None
    cfg.rewards.foot_slip = None
    # Native MuJoCo does not publish the Warp contact sensor buffers.
    NewtonMJWarpManager._initialize_contacts = classmethod(lambda cls: None)
    with launch_simulation(cfg, {'visualizer': None, 'visualizer_explicit': True}):
        env = gym.make('Motion-Tracking-G1-v0', cfg=cfg).unwrapped
        env.reset()
        solver = NewtonManager._solver
        model = solver.mj_model
        assert solver.use_mujoco_cpu and isinstance(model, mujoco.MjModel)
        # Newton updates this at its first step; the exported standalone model needs it now.
        model.opt.timestep = env.physics_dt
        robot = env.scene['robot']
        joints = {}
        bodies = {}
        for name in robot.joint_names:
            ids = [i for i in range(model.njnt) if model.joint(i).name.endswith(name)]
            assert len(ids) == 1, (name, ids)
            joints[name] = ids[0]
        for name in robot.body_names:
            ids = [i for i in range(model.nbody) if model.body(i).name.endswith(name)]
            assert len(ids) == 1, (name, ids)
            bodies[name] = ids[0]
        floors = np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
        assert len(floors) == 1
        floor = int(floors[0])
        model.geom_friction[:, 0] = 1.
        model.geom_priority[floor] = 1
        parameters = json.loads((package / 'training/sysid_parameters.json').read_text())
        efforts = torch.zeros((1, robot.num_joints), device=env.device)
        for actuator in robot.actuators.values():
            efforts[:, actuator.joint_indices] = actuator.effort_limit
        rows = []
        for name, jid in joints.items():
            dof = int(model.jnt_dofadr[jid])
            values = {k: float(getattr(model, k)[dof]) for k in ('dof_armature', 'dof_damping', 'dof_frictionloss')}
            p = parameters['joints'].get(name.removesuffix('_joint'), {}).get('parameters')
            if p:
                for field, key in [('dof_armature', 'armature'), ('dof_damping', 'viscous_friction'),
                                   ('dof_frictionloss', 'coulomb_friction')]:
                    np.testing.assert_allclose(values[field], p[key], rtol=1e-5, atol=1e-7)
            rows.append(dict(name=name, id=jid, qpos=int(model.jnt_qposadr[jid]), dof=dof, **values))
        # Check the standalone model's forward kinematics before any rollout.
        reference = np.load(package / 'motion.npz')
        metadata = json.loads((package / 'motion.json').read_text())
        qids = [model.jnt_qposadr[joints[n]] for n in contract['joint_names']]
        refq = [metadata['joint_names'].index(n) for n in contract['joint_names']]
        free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        assert len(free) == 1
        rootq = int(model.jnt_qposadr[free[0]])
        data = mujoco.MjData(model)
        max_fk = 0.
        for frame in [0, 74, 148, 222, 295]:
            pelvis = metadata['body_names'].index('pelvis')
            data.qpos[rootq:rootq+3] = reference['body_pos_w'][frame, pelvis]
            data.qpos[rootq+3:rootq+7] = reference['body_quat_w'][frame, pelvis]
            data.qpos[qids] = reference['joint_pos'][frame, refq]
            mujoco.mj_forward(model, data)
            expected = reference['body_pos_w'][frame, [metadata['body_names'].index(n) for n in bodies]]
            actual = data.xpos[list(bodies.values())]
            max_fk = max(max_fk, float(np.max(np.linalg.norm(actual - expected, axis=-1))))
        assert max_fk < 2e-4, max_fk
        out = ROOT / 'models' / mode
        out.mkdir(parents=True, exist_ok=True)
        mujoco.mj_saveModel(model, str(out / 'model.mjb'))
        actuators = [dict(id=i, name=model.actuator(i).name, joint=int(model.actuator_trnid[i, 0]),
                          gain=model.actuator_gainprm[i].tolist(), bias=model.actuator_biasprm[i].tolist(),
                          force_range=model.actuator_forcerange[i].tolist(), gear=model.actuator_gear[i].tolist())
                     for i in range(model.nu)]
        report = dict(mode=mode, mujoco_version=mujoco.__version__, physics_dt=model.opt.timestep,
                      integrator=int(model.opt.integrator), nq=model.nq, nv=model.nv, nu=model.nu,
                      model_sha256=hashlib.sha256((out / 'model.mjb').read_bytes()).hexdigest(),
                      asset_url=contract['asset_url'], asset_variants=contract['asset_variants'],
                      sysid_sha256=contract['sysid_sha256'], joints=rows, bodies=bodies, floor_geom=floor,
                      free_joint=int(free[0]), actuators=actuators, fk_max_body_error_m=max_fk,
                      mass_kg=float(model.body_mass.sum()),
                      effort_limits=efforts[0].cpu().tolist(), joint_order=robot.joint_names,
                      soft_joint_limits=robot.data.soft_joint_pos_limits.torch[0].cpu().tolist(),
                      tracked_bodies=cfg.commands.motion.body_names,
                      setup='Frozen official USD imported once by Newton; exported MJB stepped directly by MuJoCo in a separate process.')
        (out / 'model.json').write_text(json.dumps(report, indent=2) + '\n')
        print('MODEL_EXPORTED', json.dumps({k: v for k, v in report.items() if k not in ('joints', 'bodies', 'actuators')}), flush=True)
        print('ACTUATORS', json.dumps(actuators[:3]), flush=True)
        env.close()


if __name__ == '__main__':
    main()
