# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded, asynchronous recordings of DDS traffic and policy decisions."""

from __future__ import annotations

import atexit
import hashlib
import io
import json
import queue
import sqlite3
import sys
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np


class DeploymentRecorder:
    """Keep disk I/O and serialization off the DDS and control threads.

    DDS samples must not be mutated after submission. Arrays are copied on submission,
    because policy buffers can be reused. Full queues drop records instead of delaying
    control; counters and writer errors are saved separately from the recorded data.
    """

    def __init__(self, directory: str | Path, metadata: dict, *, queue_size: int = 4096):
        if queue_size < 1:
            raise ValueError("queue_size must be positive")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.path = self.directory / "recording.sqlite3"
        self.metadata = dict(metadata, schema_version=1, wall_time_ns=time.time_ns(),
                             monotonic_ns=time.monotonic_ns(), clean_shutdown=False)
        self._queue = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._closed = False
        self.error = None
        self.dropped = Counter()
        self.written = Counter()
        self.step_index = -1
        self._save_status()
        self._thread = threading.Thread(target=self._write, name="deployment-recorder", daemon=True)
        self._thread.start()
        if not self._ready.wait(5) or self.error:
            self._stop.set()
            raise OSError(f"cannot start recording: {self.error or 'writer startup timeout'}")
        atexit.register(self.close)

    def _save_status(self):
        status = dict(self.metadata, written=dict(self.written), dropped=dict(self.dropped), error=self.error)
        temporary = self.directory / "metadata.tmp"
        temporary.write_text(json.dumps(status, indent=2, allow_nan=False) + "\n")
        temporary.replace(self.directory / "metadata.json")

    def _put(self, kind: str, encoding: str, payload, timestamp_ns: int | None = None):
        if self._closed:
            return
        if self.error:
            self.dropped[kind] += 1
            return
        try:
            self._queue.put_nowait((kind, time.monotonic_ns() if timestamp_ns is None else timestamp_ns,
                                    encoding, payload))
        except queue.Full:
            self.dropped[kind] += 1

    def message(self, kind: str, message, timestamp_ns: int):
        """Retain the complete DDS message, including unrecognized/reserved fields."""
        self._put(kind, "cdr", message, timestamp_ns)

    def event(self, name: str, *, timestamp_ns: int | None = None, **values):
        self._put("event", "json", dict(name=name, **values), timestamp_ns)

    def arrays(self, kind: str, timestamp_ns: int | None = None, **values):
        self._put(kind, "npz", {key: np.asarray(value).copy() for key, value in values.items()}, timestamp_ns)

    def wrap_policy(self, policy, *, kind="policy"):
        """Capture the exact model inputs and output before any action clipping."""
        def run(*inputs):
            snapshots = {f"input_{i}": np.asarray(value).copy() for i, value in enumerate(inputs)}
            started = time.monotonic_ns()
            output = policy(*inputs)
            finished = time.monotonic_ns()
            self._put(kind, "npz", dict(step=self.step_index, inference_ns=finished - started,
                                           action_raw=np.asarray(output).copy(), **snapshots), started)
            return output
        return run

    def _write(self):
        connection = None
        try:
            connection = sqlite3.connect(self.path)
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute("CREATE TABLE records (id INTEGER PRIMARY KEY, kind TEXT NOT NULL, "
                               "monotonic_ns INTEGER NOT NULL, encoding TEXT NOT NULL, payload BLOB NOT NULL)")
            connection.commit()
            self._ready.set()
            last_commit = time.monotonic()
            pending = Counter()
            while not self._stop.is_set() or not self._queue.empty():
                try:
                    kind, timestamp, encoding, payload = self._queue.get(timeout=0.1)
                except queue.Empty:
                    pass
                else:
                    if encoding == "cdr":
                        payload = payload.serialize()
                    elif encoding == "json":
                        payload = json.dumps(payload).encode()
                    else:
                        buffer = io.BytesIO()
                        np.savez(buffer, **payload)
                        payload = buffer.getvalue()
                    connection.execute("INSERT INTO records(kind, monotonic_ns, encoding, payload) VALUES (?, ?, ?, ?)",
                                       (kind, timestamp, encoding, payload))
                    pending[kind] += 1
                if time.monotonic() - last_commit >= 1:
                    connection.commit()
                    self.written.update(pending)
                    pending.clear()
                    self._save_status()
                    last_commit = time.monotonic()
            connection.commit()
            self.written.update(pending)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            print(f"[recording FAILED] {self.error}; control continues without a complete recording", file=sys.stderr)
        finally:
            self._ready.set()
            if connection is not None:
                connection.close()

    def close(self):
        """Drain after control has stopped; preserve an explicit incomplete-run status."""
        if self._closed:
            return
        self._closed = True
        self.metadata["stopped_monotonic_ns"] = time.monotonic_ns()
        self._stop.set()
        self._thread.join(timeout=10)
        if self._thread.is_alive():
            print("[recording FAILED] writer did not stop within 10 seconds", file=sys.stderr)
            return
        self.metadata["clean_shutdown"] = self.error is None
        self.metadata["finished_wall_time_ns"] = time.time_ns()
        self.metadata["pending_records"] = self._queue.qsize()
        try:
            self._save_status()
        except OSError as exc:
            print(f"[recording FAILED] could not save final status: {exc}", file=sys.stderr)
        print(f"[recording] {self.directory}; written={dict(self.written)}, dropped={dict(self.dropped)}")
        atexit.unregister(self.close)


def create_recorder(args, contract, core, *, tracking=None, standing=None) -> DeploymentRecorder:
    """Save the policy identity and interpretation of each recorded array before control starts."""
    import importlib.metadata
    import shutil
    import subprocess
    from g1_deploy.hardware import MOTOR_NAMES
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import IMUState_, LowCmd_, LowState_, MotorCmd_, MotorState_

    leapp_path = getattr(args, "leapp", None)
    source = Path(leapp_path or getattr(args, "tracking", None) or args.policy or args.depth).resolve()
    policy = source / "policy.pt" if source.is_dir() else source
    directory = Path(args.log_dir or "recordings") / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + source.stem)
    repo = Path(__file__).resolve().parents[1]
    commit = diff = ""
    if shutil.which("git"):
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
        diff = subprocess.run(["git", "diff", "HEAD"], cwd=repo, capture_output=True, text=True).stdout
    versions = {}
    for package in ("numpy", "torch", "cyclonedds", "unitree_sdk2py") + (
        ("leapp", "onnxruntime") if leapp_path else ()
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    # JSON has no infinity: preserve unbounded duration as a string.
    arguments = {key: (str(value) if isinstance(value, float) and not np.isfinite(value) else value)
                 for key, value in vars(args).items()}
    metadata = dict(arguments=arguments, policy_path=str(policy),
                    policy_sha256=hashlib.sha256(policy.read_bytes()).hexdigest(),
                    contract=contract, git_commit=commit, versions=versions,
                    source_sha256={str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
                                   for folder in ("g1_deploy", "scripts") for path in (repo / folder).rglob("*.py")},
                    dds_schema={cls.__name__: {key: str(value) for key, value in cls.__annotations__.items()}
                                for cls in (LowState_, LowCmd_, MotorState_, MotorCmd_, IMUState_)},
                    control_dt=core.CONTROL_DT, motor_names=list(MOTOR_NAMES),
                    policy_joint_names=list(core.POLICY_JOINT_NAMES),
                    policy_to_robot=core.POLICY_TO_ROBOT.tolist(), action_scale=np.asarray(core.ACTION_SCALE).tolist(),
                    default_joint_pos=core.DEFAULT_JOINT_POS.tolist(),
                    units={"joint_position": "rad", "joint_velocity": "rad/s", "torque_estimate": "Nm",
                           "gyro": "rad/s", "accelerometer": "m/s^2", "quaternion": "wxyz"},
                    note="DDS receipt/command-write times use host monotonic clock; robot tick is a separate clock. "
                         "No measured base position or linear velocity is available in LowState. "
                         "CDR blobs preserve all 35 motor slots; the deployment commands the first 29.")
    if tracking is not None:
        metadata.update(policy_joint_names=tracking.joint_names,
                        policy_to_robot=tracking.policy_to_robot.tolist(),
                        action_scale=tracking.action_scale.tolist(),
                        default_joint_pos=tracking.default_joint_pos.tolist(),
                        tracking=dict(motion_sha256=contract["motion_sha256"],
                                      duration=tracking.duration, command="reference_joint_pos_and_vel"))
    if standing is not None:
        metadata["standing_policy"] = dict(
            directory=str(standing.directory.resolve()), contract=standing.contract,
            reference_time_s=0.0, transition_s=args.stand_transition_s,
            policy_stream="standing_policy", engagement_s=0.3,
        )
    if leapp_path:
        import yaml

        description = yaml.safe_load(source.read_text())
        artifacts = [source] + [source.parent / model["parameters"]["model_path"]
                                for model in description["models"].values()]
        initial = description["pipeline"].get("initial_values")
        if initial:
            artifacts.append(source.parent / initial)
        metadata["leapp"] = dict(
            description=description,
            artifact_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in artifacts},
            recording_kind="leapp_policy", output="joint_target in radians, not raw actions",
        )
    recorder = DeploymentRecorder(directory, metadata)
    try:
        (directory / "deployment.diff").write_text(diff)
    except OSError:
        recorder.close()
        raise
    print(f"[recording] {directory}")
    return recorder
