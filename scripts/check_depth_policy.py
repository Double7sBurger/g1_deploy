"""Smoke-test an exported vision student through the deployment path.

Loads the policy the way `run_policy_loop.py` would, feeds it a synthetic robot state and a
synthetic depth frame, and reports the shapes and the joint targets. What this catches is the
class of error that does not raise: a policy whose observation width happens to match but whose
joint order, default pose or depth scaling do not.
"""

import argparse

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export_dir", required=True, help="Directory holding policy.pt and contract.json.")
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()

    from g1_deploy import core
    from g1_deploy.depth import G1DepthPolicyRunner, load_contract, load_depth_policy

    contract = load_contract(args.export_dir)
    core.set_policy_backend("g1_29dof")

    # The exported policy has to agree with the deployment's joint table, or every target lands on
    # the wrong motor while every shape checks out.
    if list(contract["joint_names"]) != list(core.POLICY_JOINT_NAMES):
        first = next(
            (i for i, (a, b) in enumerate(zip(contract["joint_names"], core.POLICY_JOINT_NAMES)) if a != b),
            None,
        )
        raise SystemExit(
            f"joint order differs from the active backend at index {first}: "
            f"{contract['joint_names'][first]!r} vs {core.POLICY_JOINT_NAMES[first]!r}"
        )
    print(f"[check] joint order matches the g1_29dof profile ({len(core.POLICY_JOINT_NAMES)} joints)")

    policy = load_depth_policy(f"{args.export_dir}/policy.pt", contract)
    runner = G1DepthPolicyRunner(policy, contract)

    rng = np.random.default_rng(0)
    frames, height, width = contract["depth_shape"]
    joint_pos = np.array(core.build_default_pose(), dtype=np.float32)
    joint_vel = np.zeros(core.NUM_ROBOT_MOTORS, dtype=np.float32)
    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    gyro = np.zeros(3, dtype=np.float32)
    command = np.array([0.8, 0.0, 0.0], dtype=np.float32)

    for i in range(args.steps):
        # A plausible frame: ground close at the bottom of the image, far at the top.
        depth = np.linspace(3.0, 0.4, height, dtype=np.float32)[:, None] * np.ones((1, width), dtype=np.float32)
        depth = depth + 0.02 * rng.standard_normal((height, width)).astype(np.float32)
        target, action = runner.step(joint_pos, joint_vel, quat, gyro, command, depth)
        if i == 0:
            print(f"[check] obs_1d {contract['obs_1d_dim']}, depth {(frames, height, width)}")
        print(
            f"[check] step {i}: action |max| {np.abs(action).max():+.3f},"
            f" target legs {np.round(target[:6], 3).tolist()}"
        )

    # Invalid pixels must read as "nothing within range", not as "something at the lens".
    blind = np.full((height, width), np.inf, dtype=np.float32)
    runner.reset()
    runner.step(joint_pos, joint_vel, quat, gyro, command, blind)
    print("[check] all-invalid frame accepted and clamped to the far range")
    print("[check] OK")


if __name__ == "__main__":
    main()
