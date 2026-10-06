# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Portable, versioned numeric data for the offline deploynose viewer."""

from __future__ import annotations

import gzip
import json
import struct
from pathlib import Path

import numpy as np

MAGIC = b"DNOSE001"
HEADER = struct.Struct("<8sII")


def finite_json(value):
    """Keep missing telemetry explicit in JSON instead of emitting nonstandard NaN."""
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_bundle(path, metadata, summary, arrays):
    """Write aligned little-endian arrays in a gzip container, without pickle or resampling."""
    path = Path(path)
    encoded = {}
    descriptors = {}
    offset = 0
    for name, value in arrays.items():
        array = np.asarray(value)
        if array.dtype.kind not in "biuf" or not 1 <= array.ndim <= 3:
            raise ValueError(f"unsupported array {name}: {array.dtype}, {array.shape}")
        dtype = "<f8" if array.dtype.itemsize > 4 else "<f4"
        if array.dtype.kind == "u" and array.dtype.itemsize <= 4:
            dtype = "<u4"
        elif array.dtype.kind == "b":
            dtype = "|u1"
        elif array.dtype.kind == "i" and array.dtype.itemsize <= 4:
            dtype = "<i4"
        if array.dtype.kind in "iu" and array.dtype.itemsize > 4:
            if np.any(array > 2**53) or (array.dtype.kind == "i" and np.any(array < -(2**53))):
                raise ValueError(f"integer precision would be lost in {name}")
        array = np.ascontiguousarray(array, dtype=dtype)
        offset = (offset + 7) // 8 * 8
        descriptors[name] = dict(dtype=array.dtype.str, shape=list(array.shape), offset=offset, nbytes=array.nbytes)
        encoded[name] = array
        offset += array.nbytes
    manifest = dict(format="deploynose", version=1, metadata=metadata, summary=summary,
                    arrays=descriptors, payload_bytes=offset)
    header = json.dumps(finite_json(manifest), ensure_ascii=False, allow_nan=False,
                        separators=(",", ":")).encode("utf-8")
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with gzip.open(temporary, "wb", compresslevel=3) as output:
            output.write(HEADER.pack(MAGIC, len(header), 0))
            output.write(header)
            output.write(b"\0" * (-len(header) % 8))
            position = 0
            for name, array in encoded.items():
                padding = descriptors[name]["offset"] - position
                output.write(b"\0" * padding)
                if array.nbytes:
                    output.write(memoryview(array).cast("B"))
                position = descriptors[name]["offset"] + array.nbytes
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path
