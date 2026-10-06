# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Offline diagnostics from recorded measurements, without connecting to DDS."""

from __future__ import annotations

import io
import json
import sqlite3
from pathlib import Path

import numpy as np


def records(directory, kind, *, include_depth=False):
    """Iterate committed records; an interrupted run can still be inspected."""
    path = Path(directory).resolve() / "recording.sqlite3"
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    try:
        for timestamp, encoding, payload in connection.execute(
            "SELECT monotonic_ns, encoding, payload FROM records WHERE kind=? ORDER BY monotonic_ns, id", (kind,)
        ):
            if encoding == "npz":
                with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
                    # Depth stacks remain in the raw archive; aggregate diagnostics do not need them.
                    value = {key: archive[key] for key in archive.files if include_depth or key != "input_1"}
            elif encoding == "json":
                value = json.loads(payload)
            else:
                value = payload
            yield timestamp, value
    finally:
        connection.close()


def percentiles(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not values.size:
        return None
    return dict(zip(("p50", "p95", "p99", "max"), np.percentile(values, [50, 95, 99, 100]).tolist()))


def tick_intervals(ticks):
    """Unwrap the unsigned millisecond counter; resets remain negative intervals."""
    return ((np.diff(np.asarray(ticks, dtype=np.int64)) + 2**31) % 2**32 - 2**31) * 0.001


def quaternion_rpy(quaternion):
    """Derive roll/pitch/yaw from wxyz; invalid orientations remain missing data."""
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    normalized = np.full_like(quaternion, np.nan)
    np.divide(quaternion, norm, out=normalized, where=np.isfinite(norm) & (norm > 1e-9))
    w, x, y, z = normalized.T
    return np.stack((np.arctan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y)),
                     np.arcsin(np.clip(2 * (w*y - z*x), -1, 1)),
                     np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))), axis=-1)


def vibration_spectrum(ticks, velocity, cutoff_hz=5.0):
    """Welch spectra on regular, gap-free robot-time windows; never interpolate gaps.

    The cutoff selects a diagnostic band, not a pass/fail threshold. Gait harmonics,
    sensor noise and genuine oscillations may all contribute to this band.
    """
    dt = tick_intervals(ticks)
    positive = dt[dt > 0]
    if not positive.size:
        return None
    period = float(np.median(positive))
    size = min(256, len(velocity))
    if size < 32 or 0.5 / period <= cutoff_hz:
        return None
    window = np.hanning(size)
    power = []
    for start in range(0, len(velocity) - size + 1, size // 2):
        segment = velocity[start:start + size]
        intervals = dt[start:start + size - 1]
        if not np.all(np.abs(intervals - period) <= period * 0.05) or not np.isfinite(segment).all():
            continue
        centered = segment - segment.mean(axis=0)
        spectrum = np.abs(np.fft.rfft(centered * window[:, None], axis=0)) ** 2
        spectrum *= period / np.sum(window**2)
        spectrum[1:-1 if size % 2 == 0 else None] *= 2
        power.append(spectrum)
    if not power:
        return None
    density = np.mean(power, axis=0)
    frequency = np.fft.rfftfreq(size, period)
    band = frequency >= cutoff_hz
    return dict(frequency=frequency, density=density, windows=len(power), cutoff_hz=cutoff_hz,
                band_rms=np.sqrt(density[band].sum(axis=0) / (size * period)),
                peak_hz=frequency[band][np.argmax(density[band], axis=0)])


def analyze(directory):
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_

    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata["schema_version"] != 1:
        raise ValueError(f"unsupported recording schema: {metadata['schema_version']}")
    motor_fields = ("q", "dq", "ddq", "tau_est", "temperature", "vol", "motorstate")
    imu_fields = ("quaternion", "gyroscope", "accelerometer", "rpy")
    samples = {key: [] for key in ("timestamp", "tick", "mode_machine", *motor_fields, *imu_fields)}
    for stamp, payload in records(directory, "lowstate"):
        state = LowState_.deserialize(payload)
        samples["timestamp"].append(stamp)
        samples["tick"].append(state.tick)
        samples["mode_machine"].append(state.mode_machine)
        for field in motor_fields:
            dtype = np.uint32 if field == "motorstate" else np.float32
            samples[field].append(np.asarray([getattr(m, field) for m in state.motor_state], dtype=dtype))
        for field in imu_fields:
            samples[field].append(np.asarray(getattr(state.imu_state, field), dtype=np.float32))
    if len(samples["timestamp"]) < 2:
        raise ValueError("recording needs at least two lowstate samples")
    origin = metadata["monotonic_ns"]
    t = (np.asarray(samples.pop("timestamp"), dtype=np.int64) - origin) * 1e-9
    arrays = {key: np.asarray(value) for key, value in samples.items()}
    arrays["t"] = t
    arrays["orientation_rpy"] = quaternion_rpy(arrays["quaternion"])
    ticks = arrays["tick"]
    del samples
    events = [(stamp, value) for stamp, value in records(directory, "event")]
    starts = [stamp for stamp, event in events if event["name"] == "policy_start"]
    stops = [stamp for stamp, event in events if event["name"] == "policy_stop"]
    lo = (starts[0] - origin) * 1e-9 if starts else t[0]
    hi = (stops[-1] - origin) * 1e-9 if stops else t[-1]
    active = (t >= lo) & (t <= hi)
    summary = dict(samples=len(t), recorded_seconds=float(t[-1] - t[0]),
                   analysis_phase="policy" if starts else "all (no policy_start event)",
                   clean_shutdown=metadata["clean_shutdown"], dropped=metadata.get("dropped", {}),
                   writer_error=metadata.get("error"),
                   receipt_interval_ms=percentiles(np.diff(t) * 1000),
                   robot_tick_interval_ms=percentiles(tick_intervals(ticks) * 1000),
                   nonfinite_state_values=int(sum(np.count_nonzero(~np.isfinite(arrays[key]))
                                                for key in ("q", "dq", "tau_est", "quaternion", "gyroscope"))),
                   measured_base_linear_velocity_available=False,
                   events=[dict(time_s=(stamp - origin) * 1e-9, **event) for stamp, event in events])
    dt = tick_intervals(ticks[active])
    positive = dt[dt > 0]
    period = float(np.median(positive)) if len(positive) else float("nan")
    valid = (dt > 0) & (dt <= 1.5 * period)
    summary["robot_tick_gaps"] = int(np.count_nonzero(dt > 1.5 * period))
    summary["duplicate_or_reset_ticks"] = int(np.count_nonzero(dt <= 0))
    summary["joint_velocity_change_rate_rad_s2_p99"] = (
        np.nanpercentile(np.abs(np.diff(arrays["dq"][active], axis=0)[valid] / dt[valid, None]), 99, axis=0).tolist()
        if valid.any() else None)
    spectrum = vibration_spectrum(ticks[active], arrays["dq"][active])
    if spectrum is not None:
        summary["vibration_band"] = dict(cutoff_hz=spectrum["cutoff_hz"], windows=spectrum["windows"],
                                         velocity_rms_rad_s=spectrum["band_rms"].tolist(),
                                         peak_frequency_hz=spectrum["peak_hz"].tolist())
        arrays.update(spectrum_frequency=spectrum["frequency"], spectrum_density=spectrum["density"])
    commands = [(stamp, LowCmd_.deserialize(payload)) for stamp, payload in records(directory, "lowcmd")]
    if commands:
        arrays["target_t"] = (np.asarray([s[0] for s in commands], dtype=np.int64) - origin) * 1e-9
        for field in ("q", "dq", "kp", "kd", "tau"):
            arrays["target_" + field] = np.asarray([[getattr(m, field) for m in s.motor_cmd] for _, s in commands])
    steps = list(records(directory, "step"))
    if steps:
        arrays["control_t"] = (np.asarray([s[0] for s in steps], dtype=np.int64) - origin) * 1e-9
        arrays["command"] = np.stack([s[1]["command"] for s in steps])
        arrays["heading_target"] = np.asarray([s[1]["heading"] for s in steps])
        quat = np.stack([s[1]["quat"] for s in steps])
        arrays["yaw"] = quaternion_rpy(quat)[:, 2]
        arrays["heading_error"] = (arrays["heading_target"] - arrays["yaw"] + np.pi) % (2 * np.pi) - np.pi
        summary["heading_error_deg"] = percentiles(np.abs(np.rad2deg(arrays["heading_error"])))
        summary["control_interval_ms"] = percentiles(np.diff(arrays["control_t"]) * 1000)
        # State age is evaluated at policy-input capture, not at the start of the loop.
        arrays["state_age_ms"] = np.asarray([(s[1]["captured_ns"] - s[1]["state_received_ns"]) * 1e-6 for s in steps])
        summary["state_age_ms"] = percentiles(arrays["state_age_ms"])
        arrays["control_interval_ms"] = np.r_[np.nan, np.diff(arrays["control_t"]) * 1000]
        arrays["control_step"] = np.asarray([value["step"] for _, value in steps])
    policy_samples = list(records(directory, "policy"))
    inference = [value["inference_ns"] * 1e-6 for _, value in policy_samples]
    summary["inference_ms"] = percentiles(inference)
    if policy_samples:
        arrays["policy_t"] = (np.asarray([stamp for stamp, _ in policy_samples], dtype=np.int64) - origin) * 1e-9
        arrays["action_raw"] = np.stack([value["action_raw"].reshape(-1) for _, value in policy_samples])
        arrays["observation"] = np.stack([value["input_0"].reshape(-1) for _, value in policy_samples])
        arrays["inference_ms"] = np.asarray(inference)
        arrays["policy_step"] = np.asarray([value["step"] for _, value in policy_samples])
        summary["action_step_change_abs"] = percentiles(np.abs(np.diff(arrays["action_raw"], axis=0)))
    writes = list(records(directory, "write"))
    summary["failed_command_writes"] = sum(not bool(value["success"]) for _, value in writes)
    summary["command_write_ms"] = percentiles([(value["finished_ns"] - stamp) * 1e-6 for stamp, value in writes])
    if writes:
        arrays["write_t"] = (np.asarray([stamp for stamp, _ in writes], dtype=np.int64) - origin) * 1e-9
        arrays["write_ms"] = np.asarray([(value["finished_ns"] - stamp) * 1e-6 for stamp, value in writes])
        arrays["write_success"] = np.asarray([bool(value["success"]) for _, value in writes])
    depth = list(records(directory, "depth_age"))
    if depth:
        arrays["depth_t"] = (np.asarray([stamp for stamp, _ in depth], dtype=np.int64) - origin) * 1e-9
        arrays["depth_age_ms"] = np.asarray([value["age"] * 1000 for _, value in depth])
    arrays["receipt_interval_ms"] = np.r_[np.nan, np.diff(t) * 1000]
    arrays["robot_interval_ms"] = np.r_[np.nan, tick_intervals(ticks) * 1000]
    summary["expected_control_ms"] = metadata["control_dt"] * 1000
    return metadata, summary, arrays


def save_report(directory, output=None, *, plots=True):
    """Write portable arrays, metrics and diagnostic plots beside the raw recording."""
    from g1_deploy.bundle import finite_json, write_bundle

    metadata, summary, data = analyze(directory)
    output = Path(output or Path(directory) / "analysis")
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(finite_json(summary), indent=2, allow_nan=False) + "\n")
    np.savez_compressed(output / "states.npz", **data)
    write_bundle(output / (Path(directory).resolve().name + ".deploynose"), metadata, summary, data)
    if not plots:
        return output, summary
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    axes[0].plot(data["t"], data["gyroscope"], linewidth=0.6)
    axes[0].set_ylabel("IMU gyro [rad/s]")
    axes[0].legend(["x", "y", "z"])
    axes[1].plot(data["t"], np.rad2deg(data["orientation_rpy"][:, :2]), linewidth=0.6)
    axes[1].set_ylabel("IMU quaternion\nroll/pitch [deg]")
    axes[1].legend(["roll", "pitch"])
    if "control_t" in data:
        t = data["control_t"]
        axes[2].plot(t, np.rad2deg(data["heading_error"]))
        axes[2].set_ylabel("Heading error [deg]")
        axes[3].plot(t[1:], np.diff(t) * 1000, label="control period")
        axes[3].plot(t, data["state_age_ms"], label="state age")
        axes[3].axhline(metadata["control_dt"] * 1000, linestyle="--", color="black", label="nominal period")
        axes[3].legend()
    axes[3].set_ylabel("Timing [ms]")
    axes[3].set_xlabel("Host monotonic time since recording start [s]")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "overview.png", dpi=140)
    plt.close(fig)
    names = metadata.get("motor_names", [f"motor_{i}" for i in range(29)])
    fig, axes = plt.subplots(len(names), 2, figsize=(17, 2 * len(names)), squeeze=False)
    for i, name in enumerate(names):
        axes[i, 0].plot(data["t"], data["q"][:, i], linewidth=0.6, label="measured")
        if "target_t" in data:
            axes[i, 0].step(data["target_t"], data["target_q"][:, i], where="post", linewidth=0.6,
                            label="commanded target (write attempt)")
        axes[i, 0].set_ylabel(name + "\n[rad]", fontsize=8)
        axes[i, 1].plot(data["t"], data["dq"][:, i], linewidth=0.6)
        axes[i, 1].set_ylabel("velocity [rad/s]")
        for axis in axes[i]:
            axis.grid(alpha=0.2)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(output / "joints.png", dpi=100)
    plt.close(fig)
    if "spectrum_frequency" in data:
        fig, axis = plt.subplots(figsize=(12, 6))
        for i, name in enumerate(names):
            axis.semilogy(data["spectrum_frequency"], np.maximum(data["spectrum_density"][:, i], 1e-20), label=name)
        axis.set(xlabel="Frequency [Hz]", ylabel="Joint velocity PSD [(rad/s)^2/Hz]",
                 title="Policy phase, regular robot-time windows only")
        axis.legend(fontsize=6, ncol=3)
        fig.tight_layout()
        fig.savefig(output / "spectrum.png", dpi=140)
        plt.close(fig)
    return output, summary
