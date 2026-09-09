"""The depth half of a vision student's observation, and the policy that reads both halves.

`core.py` already assembles the 690-value proprioception contract. A distillation student trained
with a chest camera reads that *and* a stack of depth frames, and `rsl_rl` keeps them as separate
observation groups: the 1D group goes through the model's normalizer, the 4D group through a
convolutional encoder, and the two latents are concatenated before the MLP. The exported policy
therefore takes two arguments rather than one, which is why `core.load_policy` cannot load it.

Everything here mirrors what training did, and the values are read from the contract JSON that
`scripts/export_depth_student.py` writes beside the policy rather than transcribed. The three that
are easy to get wrong and produce a plausible-looking image either way:

* **Invalid pixels become the far range, not zero.** A ray that hits nothing comes back as `+inf`
  from the renderer and as 0 from a RealSense. Mapping either to 0 tells the policy something is
  touching the lens; training mapped them to 3 m.
* **Frames stack oldest-first along the channel axis.** Verified against
  `isaaclab.utils.buffers.CircularBuffer`, which is what fed the term during training.
* **The scale is 1/3 m applied after clipping to [0, 3].** The network sees [0, 1].
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_CONTRACT_CACHE: dict[str, dict] = {}


def load_contract(path: str | Path) -> dict:
    """Read the contract JSON written next to an exported policy.

    Args:
        path: Path to ``contract.json``, or to the directory containing it.

    Returns:
        The parsed contract.
    """
    p = Path(path)
    if p.is_dir():
        p = p / "contract.json"
    key = str(p.resolve())
    if key not in _CONTRACT_CACHE:
        with open(p) as handle:
            _CONTRACT_CACHE[key] = json.load(handle)
    return _CONTRACT_CACHE[key]


class DepthStack:
    """The last ``frames`` depth images, preprocessed exactly as training preprocessed them.

    Args:
        contract: Parsed contract from :func:`load_contract`.
    """

    def __init__(self, contract: dict) -> None:
        shape = contract["depth_shape"]
        if shape is None:
            raise ValueError("this contract has no depth group; use core.load_policy instead")
        self.frames, self.height, self.width = int(shape[0]), int(shape[1]), int(shape[2])
        self.max_range = float(contract["depth_max_range_m"])
        self._stack: np.ndarray | None = None

    def reset(self) -> None:
        """Drop the history.

        Carrying frames across a reset shows the policy terrain from the previous episode, which is
        a discontinuity it never saw in training.
        """
        self._stack = None

    def append(self, frame_m: np.ndarray) -> np.ndarray:
        """Add one depth frame and return the stacked observation.

        Args:
            frame_m: Depth in metres, shape ``(height, width)``. Values that are non-finite or
                non-positive are treated as "no return" and clamped to the far range, which is what
                both the renderer's ``+inf`` and a RealSense's 0 actually mean.

        Returns:
            Stacked depth of shape ``(1, frames, height, width)``, float32, scaled into ``[0, 1]``.

        Raises:
            ValueError: If the frame is not the resolution the policy was trained on. Resampling
                belongs to the camera driver, where the intrinsics are known; silently resizing here
                would change the field of view without changing the numbers.
        """
        frame = np.asarray(frame_m, dtype=np.float32)
        if frame.shape != (self.height, self.width):
            raise ValueError(f"depth frame is {frame.shape}, expected {(self.height, self.width)}")

        frame = np.where(np.isfinite(frame) & (frame > 0.0), frame, self.max_range)
        frame = np.clip(frame, 0.0, self.max_range) / self.max_range

        if self._stack is None:
            # Isaac Lab's circular buffer starts full of copies of the first frame, so a fresh
            # policy sees a constant history rather than an empty one.
            self._stack = np.repeat(frame[None, :, :], self.frames, axis=0)
        else:
            self._stack = np.concatenate([self._stack[1:], frame[None, :, :]], axis=0)
        return self._stack[None, :, :, :].astype(np.float32)


def load_depth_policy(path: str, contract: dict):
    """Load an exported vision student and wrap it as a numpy callable.

    Args:
        path: TorchScript file written by ``scripts/export_depth_student.py``.
        contract: Parsed contract for the same export.

    Returns:
        ``run(obs_1d, depth) -> action``, both inputs batched, action shaped ``(action_dim,)``.

    Raises:
        ValueError: If the policy's widths disagree with the contract, which usually means the
            policy and the contract came from different runs.
    """
    import torch

    module = torch.jit.load(path)
    module.eval()

    obs_dim = int(contract["obs_1d_dim"])
    frames, height, width = (int(v) for v in contract["depth_shape"])
    with torch.inference_mode():
        probe = module(torch.zeros(1, obs_dim), [torch.zeros(1, frames, height, width)])
    if probe.shape[-1] != int(contract["action_dim"]):
        raise ValueError(
            f"policy maps {obs_dim} observations to {probe.shape[-1]} actions,"
            f" expected {contract['action_dim']}"
        )

    def run(obs_1d: np.ndarray, depth: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            out = module(
                torch.from_numpy(np.ascontiguousarray(obs_1d, dtype=np.float32)),
                [torch.from_numpy(np.ascontiguousarray(depth, dtype=np.float32))],
            )
        return out.numpy()

    return run


class G1DepthPolicyRunner:
    """`core.G1PolicyRunner` for a student that also reads a chest depth camera.

    The proprioception half is delegated to the existing runner, so the term-major history layout,
    the robot-to-policy joint mapping and the ``default + scale * action`` convention are shared
    rather than reimplemented. What is added is the second observation group and a policy call that
    takes both.

    Args:
        policy: Callable from :func:`load_depth_policy`.
        contract: Parsed contract for the same export.
        num_robot_motors: Motors on ``rt/lowcmd``.
    """

    def __init__(self, policy, contract: dict, *, num_robot_motors: int | None = None) -> None:
        from g1_deploy import core

        self._contract = contract
        self._depth = DepthStack(contract)
        self._core = core.G1PolicyRunner(
            policy=lambda obs: policy(obs, self._depth_obs),
            **({} if num_robot_motors is None else {"num_robot_motors": num_robot_motors}),
        )
        self._depth_obs: np.ndarray | None = None

    def reset(self) -> None:
        """Clear both histories."""
        self._core.reset()
        self._depth.reset()

    def step(
        self,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        quat_wxyz: np.ndarray,
        gyro: np.ndarray,
        command: np.ndarray,
        depth_m: np.ndarray,
        *,
        action_limit: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run one control step with a fresh depth frame.

        Args:
            joint_pos: See :meth:`g1_deploy.core.G1PolicyRunner.observe`.
            joint_vel: See :meth:`g1_deploy.core.G1PolicyRunner.observe`.
            quat_wxyz: See :meth:`g1_deploy.core.G1PolicyRunner.observe`.
            gyro: See :meth:`g1_deploy.core.G1PolicyRunner.observe`.
            command: See :meth:`g1_deploy.core.G1PolicyRunner.observe`.
            depth_m: Depth image in metres at the policy's resolution, shape ``(height, width)``.
            action_limit: Symmetric clamp on the raw policy output, or ``None``.

        Returns:
            ``(joint_target, action)`` as :meth:`g1_deploy.core.G1PolicyRunner.step` returns them.
        """
        # Staged before the core step because the policy callable reads it during that call. The
        # camera runs at the control rate in training (``sim.render_interval == decimation``), so
        # one frame per step is the contract, not a convenience.
        self._depth_obs = self._depth.append(depth_m)
        return self._core.step(
            joint_pos, joint_vel, quat_wxyz, gyro, command, action_limit=action_limit
        )
