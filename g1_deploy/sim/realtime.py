"""Physics worker for the free-running simulator; graphics use separate snapshots."""

from __future__ import annotations

import threading
import time

import mujoco

from g1_deploy.timing import PeriodicDeadline


class RealtimePhysics:
    """Own the live MjData on a worker, leaving GL on the caller's thread for macOS.

    The renderer and viewer must use their own model/data. ``copy_state`` is their
    only access to the live state, and never holds the physics lock while rendering.
    """

    def __init__(self, env, default_pose, *, hoist_s=0.0, reset_on_fall=False, fall_height=0.2):
        self.env = env
        self.default_pose = default_pose
        self.hoist_steps = int(round(hoist_s / env.sim_dt))
        self.reset_on_fall = reset_on_fall
        self.fall_height = fall_height
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._error = None
        self._thread = threading.Thread(target=self._run, name="g1_physics", daemon=True)
        self.started_at = None
        self.finished_at = None
        self.steps = 0
        self.pacer = None

    def start(self) -> None:
        self._thread.start()

    def copy_state(self, model, data) -> None:
        """Copy a coherent snapshot into an independent, matching model/data pair."""
        with self._lock:
            mujoco.mj_copyData(data, model, self.env.data)

    def check(self) -> None:
        if self._error is not None:
            raise RuntimeError("simulator physics worker failed") from self._error

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            raise RuntimeError("physics worker did not stop within 5s")

    def _run(self) -> None:
        try:
            waiting = PeriodicDeadline(self.env.sim_dt)
            while not self._stop.is_set() and not self.env.bridge.cmd_received:
                self.env.publish_only()
                waiting.wait()
            if self._stop.is_set():
                return
            print(f"[ok] controller connected at t={self.env.data.time:.2f}s, physics running")
            hoist_z = float(self.env.data.qpos[2])
            if self.hoist_steps:
                print(f"[..] hoist holding the pelvis at {hoist_z:.3f} m for"
                      f" {self.hoist_steps * self.env.sim_dt:.1f}s, then releasing")
            step = 0
            self.started_at = time.monotonic()
            self.pacer = PeriodicDeadline(self.env.sim_dt)
            while not self._stop.is_set():
                with self._lock:
                    if step < self.hoist_steps:
                        self.env.apply_hoist(hoist_z)
                    elif step == self.hoist_steps and self.hoist_steps:
                        self.env.release_hoist()
                        print(f"[..] hoist released at t={self.env.data.time:.2f}s")
                    self.env.sim_step()
                    step += 1
                    self.steps += 1
                    if self.reset_on_fall and self.env.data.qpos[2] < self.fall_height:
                        print(f"[warn] fell at t={self.env.data.time:.2f}s, resetting")
                        self.env.reset(self.default_pose)
                        step = 0
                self.pacer.wait()
        except BaseException as exc:
            self._error = exc
        finally:
            self.finished_at = time.monotonic()
