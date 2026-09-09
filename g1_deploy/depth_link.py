# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Carry depth frames from the robot's PC2 to whatever runs the control loop.

The camera is wired to PC2 and ``pyrealsense2`` has no macOS build, so the frame has to cross a
network before the policy can read it. What crosses is deliberately the *smallest possible* thing:
the 64x38 frame the policy actually consumes, already cropped and downsampled, 4.9 kB at 50 Hz.
Sending the native 848x480 instead would be 100x the bandwidth and would put the crop -- which
depends on the camera's own intrinsics -- on the machine that cannot query them.

**Why UDP and not TCP.** A control loop wants the newest frame, not every frame. TCP would hold a
retransmitted stale frame in front of a fresh one; UDP lets a lost frame simply be a lost frame,
which :class:`DepthReceiver` reports and the loop can decide about. One frame is one datagram; at
4880 bytes IP fragments it into four, and on a quiet direct gigabit link losing one is rare -- but
it is *possible*, which is why the receiver counts drops instead of assuming there are none.

**The crop is not optional and not cosmetic.** Measured on this robot the D435i's depth stream is
89.6 x 58.7 degrees at 848x480, while the policy was trained on 87.0 x 58.8. Resizing the full frame
would squeeze 89.6 degrees of world into an image that means 87.0 -- every array is still 38x64, so
nothing downstream can notice, and the policy silently reads terrain that is 3% narrower than it
believes. :func:`crop_to_fov` removes the difference before any resampling.
"""

from __future__ import annotations

import math
import socket
import struct
import threading
import time

import numpy as np

WIRE_MAGIC = 0x47314450
"""``G1DP``. Guards against a stale publisher or an unrelated service on the same port."""

HEADER = struct.Struct("<IIQHH")
"""``magic, seq, capture_time_ns, height, width``, then ``height * width`` uint16 millimetres.

Millimetres rather than float32 metres for two reasons, one of which is a hard limit. The RealSense
already produces ``z16`` with a 0.001 m scale, so uint16 mm *is* the native representation and
float32 would be inventing precision the sensor does not have -- a D435i's depth error at 2 m is
around 2 cm, twenty times the quantisation. And macOS caps a UDP datagram at 9216 bytes
(``net.inet.udp.maxdgram``): a 64x38 float32 frame is 9748 and fails to send, while uint16 is 4880.
"""

DEFAULT_PORT = 5601
"""UDP port for the depth stream. Nothing on the G1 uses it; DDS lives on 7400+."""

INVALID = 0.0
"""What a pixel with no return is sent as.

The RealSense already reports 0 for these and Isaac Lab reports ``+inf``;
:meth:`~g1_deploy.depth.DepthStack.append` maps both to the far range. Keeping the sentinel on the
wire rather than substituting the far range here means the receiver can still tell "nothing there"
from "something at exactly 3 m", which matters when diagnosing a camera that has gone blind.
"""


def crop_to_fov(width: int, height: int, hfov_deg: float, vfov_deg: float,
                target_hfov_deg: float, target_vfov_deg: float) -> tuple[int, int, int, int]:
    """Centre-crop box that turns a camera's field of view into the trained one.

    Args:
        width: Native frame width [px].
        height: Native frame height [px].
        hfov_deg: The camera's real horizontal field of view, from its own intrinsics.
        vfov_deg: The camera's real vertical field of view.
        target_hfov_deg: Horizontal field of view the policy was trained on.
        target_vfov_deg: Vertical field of view the policy was trained on.

    Returns:
        ``(x0, y0, w, h)``. An axis whose native field of view is already narrower than the target
        is left alone -- cropping cannot widen it, and the alternative (padding with invented
        pixels) would be worse than the small mismatch.
    """
    fx = width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    fy = height / (2.0 * math.tan(math.radians(vfov_deg) / 2.0))
    want_w = min(width, int(round(2.0 * fx * math.tan(math.radians(target_hfov_deg) / 2.0))))
    want_h = min(height, int(round(2.0 * fy * math.tan(math.radians(target_vfov_deg) / 2.0))))
    return (width - want_w) // 2, (height - want_h) // 2, want_w, want_h


def downsample_masked(frame_m: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Area-average a depth image, ignoring pixels that have no return.

    Args:
        frame_m: Depth in metres; ``<= 0`` or non-finite means no return.
        out_h: Output height.
        out_w: Output width.

    Returns:
        ``(out_h, out_w)`` float32 metres, with :data:`INVALID` where an entire output cell had no
        valid input pixel.

    .. attention::
        Averaging is a **divergence from training**, and a deliberate one. Isaac Lab rendered 64x38
        directly, one ray per pixel, so a cell straddling a depth edge came back as either the near
        surface or the far one. Averaging 12.7x12.7 real pixels across that same edge returns a
        depth that is at neither -- a surface that does not exist. The alternative, point-sampling
        one pixel in 161, throws away the sensor's redundancy and keeps all of its noise. Averaging
        is chosen because a real D435i frame is 11% invalid and visibly noisy where Isaac Lab's is
        neither, so the noise is the larger error; the phantom-edge artefact is the price and is
        worth re-examining if the policy behaves oddly near obstacle edges.

        Only valid pixels contribute, so a cell that is half no-return reports the mean of the half
        that returned rather than being dragged toward the sentinel.
    """
    frame = np.asarray(frame_m, dtype=np.float32)
    h, w = frame.shape
    valid = np.isfinite(frame) & (frame > 0.0)
    values = np.where(valid, frame, 0.0).astype(np.float64)
    counts = valid.astype(np.float64)

    # reduceat with computed edges gives variable-width bins, which is what a non-integer ratio
    # (810/64 = 12.66) needs; a reshape-based block mean would silently require an integer one.
    y_edges = (np.arange(out_h) * h // out_h).astype(np.intp)
    x_edges = (np.arange(out_w) * w // out_w).astype(np.intp)
    values = np.add.reduceat(np.add.reduceat(values, y_edges, axis=0), x_edges, axis=1)
    counts = np.add.reduceat(np.add.reduceat(counts, y_edges, axis=0), x_edges, axis=1)

    out = np.divide(values, counts, out=np.full_like(values, INVALID), where=counts > 0)
    return out.astype(np.float32)


def pack_frame(seq: int, frame_m: np.ndarray, capture_time_ns: int | None = None) -> bytes:
    """Serialise one frame for the wire.

    Args:
        seq: Monotonically increasing frame counter.
        frame_m: ``(height, width)`` float32 metres.
        capture_time_ns: When the camera produced it; ``None`` stamps now.

    Returns:
        Header followed by a row-major uint16 payload in millimetres.
    """
    frame = np.asarray(frame_m, dtype=np.float32)
    mm = np.clip(np.rint(frame * 1000.0), 0.0, 65535.0).astype(np.uint16)
    stamp = time.time_ns() if capture_time_ns is None else int(capture_time_ns)
    return HEADER.pack(WIRE_MAGIC, seq & 0xFFFFFFFF, stamp, frame.shape[0], frame.shape[1]) + mm.tobytes()


def unpack_frame(payload: bytes) -> tuple[int, int, np.ndarray]:
    """Parse a datagram produced by :func:`pack_frame`.

    Args:
        payload: One complete datagram.

    Returns:
        ``(seq, capture_time_ns, frame_m)``.

    Raises:
        ValueError: On a bad magic or a truncated payload -- either means the datagram did not come
            from a matching publisher, and guessing at it would hand the policy garbage.
    """
    if len(payload) < HEADER.size:
        raise ValueError(f"datagram is {len(payload)} bytes, shorter than the header")
    magic, seq, stamp, height, width = HEADER.unpack_from(payload)
    if magic != WIRE_MAGIC:
        raise ValueError(f"bad magic 0x{magic:08x}; something else is on this port")
    expect = HEADER.size + height * width * 2
    if len(payload) != expect:
        raise ValueError(f"datagram is {len(payload)} bytes, expected {expect} for {height}x{width}")
    mm = np.frombuffer(payload, dtype=np.uint16, count=height * width, offset=HEADER.size)
    return seq, stamp, (mm.reshape(height, width).astype(np.float32) / 1000.0)


class DepthReceiver:
    """Hold the newest depth frame from the network, and enough bookkeeping to distrust it.

    A control loop that simply reads "the latest frame" cannot tell a fresh one from one the
    publisher stopped sending four seconds ago -- both are just an array. This tracks the sequence
    numbers and the arrival time so the loop can refuse to run on a stale image.

    Args:
        expect_shape: ``(height, width)`` the policy needs; a frame of any other shape is rejected
            rather than resized, because resizing here would change the field of view.
        port: UDP port to listen on.
        bind: Interface address to bind; ``""`` is every interface.
    """

    def __init__(self, expect_shape: tuple[int, int], port: int = DEFAULT_PORT, bind: str = "") -> None:
        self.expect_shape = (int(expect_shape[0]), int(expect_shape[1]))
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._recv_monotonic = 0.0
        self._seq = -1
        self.received = 0
        """Datagrams accepted."""
        self.dropped = 0
        """Frames the publisher sent that never arrived, inferred from sequence gaps."""
        self.rejected = 0
        """Datagrams discarded as malformed or the wrong shape."""

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        self._sock.bind((bind, port))
        self._sock.settimeout(0.2)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="depth_rx", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                payload, _ = self._sock.recvfrom(1 << 16)
            except (TimeoutError, socket.timeout):
                continue
            except OSError:
                return
            try:
                seq, _stamp, frame = unpack_frame(payload)
            except ValueError:
                self.rejected += 1
                continue
            if frame.shape != self.expect_shape:
                self.rejected += 1
                continue
            with self._lock:
                if self._seq >= 0:
                    gap = (seq - self._seq) & 0xFFFFFFFF
                    if 0 < gap < 1000:
                        self.dropped += gap - 1
                self._seq = seq
                self._frame = frame
                self._recv_monotonic = time.monotonic()
                self.received += 1

    def latest(self) -> tuple[np.ndarray | None, float]:
        """Newest frame and how long ago it arrived.

        Returns:
            ``(frame_m, age_s)``; ``(None, inf)`` before the first frame.
        """
        with self._lock:
            if self._frame is None:
                return None, float("inf")
            return self._frame, time.monotonic() - self._recv_monotonic

    def wait_for_frame(self, timeout: float = 10.0) -> np.ndarray:
        """Block until a frame arrives.

        Args:
            timeout: Give up after this long [s].

        Returns:
            The first frame received.

        Raises:
            TimeoutError: If nothing arrives -- usually the publisher is not running, is aimed at a
                different address, or a firewall is dropping the datagrams.
        """
        deadline = time.monotonic() + timeout
        while True:
            frame, _ = self.latest()
            if frame is not None:
                return frame
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"no depth frame on UDP port {self._sock.getsockname()[1]} within {timeout:.0f}s"
                    f" ({self.rejected} datagrams rejected)"
                )
            time.sleep(0.01)

    def close(self) -> None:
        """Stop the receiver thread and release the socket."""
        self._stop.set()
        self._sock.close()
        self._thread.join(timeout=1.0)
