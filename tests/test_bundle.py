import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from g1_deploy.bundle import HEADER, MAGIC, write_bundle


class BundleTests(unittest.TestCase):
    def test_preserves_shapes_precision_missing_values_and_integer_flags(self):
        arrays = {
            "t": np.array([0.123456789, 0.123456799], dtype=np.float64),
            "q": np.array([[0, np.nan], [np.inf, -1]], dtype=np.float32),
            "flags": np.array([2**32 - 1, 0], dtype=np.uint32),
            "signed": np.array([2**31 - 1, -(2**31)], dtype=np.int32),
            "empty": np.empty((0, 3), dtype=np.float32),
            "tick": np.array([2**32 - 1, 0], dtype=np.int64),
            "valid": np.array([False, True]),
        }
        with tempfile.TemporaryDirectory() as root:
            path = write_bundle(Path(root) / "trial.deploynose", {"label": "试验"}, {"missing": float("nan")}, arrays)
            data = gzip.decompress(path.read_bytes())
        magic, length, reserved = HEADER.unpack_from(data)
        self.assertEqual((magic, reserved), (MAGIC, 0))
        manifest = json.loads(data[16:16 + length])
        self.assertEqual(manifest["metadata"]["label"], "试验")
        self.assertIsNone(manifest["summary"]["missing"])
        base = 16 + (length + 7) // 8 * 8
        self.assertEqual(len(data), base + manifest["payload_bytes"])
        for name, original in arrays.items():
            spec = manifest["arrays"][name]
            self.assertEqual(spec["offset"] % 8, 0)
            restored = np.frombuffer(data, dtype=spec["dtype"], count=original.size,
                                     offset=base + spec["offset"]).reshape(spec["shape"])
            np.testing.assert_array_equal(restored, original)

    def test_rejects_objects_and_unsafe_integer_conversion(self):
        with tempfile.TemporaryDirectory() as root:
            for array in (np.array([object()]), np.array([2**60], dtype=np.int64)):
                with self.subTest(dtype=array.dtype), self.assertRaises(ValueError):
                    write_bundle(Path(root) / "bad.deploynose", {}, {}, {"bad": array})

    def test_failed_export_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "run.deploynose"
            path.write_bytes(b"original")
            with patch("g1_deploy.bundle.gzip.open", side_effect=OSError("storage unavailable")):
                with self.assertRaises(OSError):
                    write_bundle(path, {}, {}, {"t": np.arange(2)})
            self.assertEqual(path.read_bytes(), b"original")
            self.assertFalse(path.with_suffix(".deploynose.tmp").exists())

    def test_analysis_exports_bundle_without_matplotlib(self):
        from g1_deploy.analysis import save_report
        data = {"t": np.arange(2.0), "q": np.zeros((2, 29))}
        with tempfile.TemporaryDirectory() as root:
            run = Path(root) / "test_run"
            with patch("g1_deploy.analysis.analyze", return_value=({}, {"samples": 2, "policy_window_s": None}, data)):
                output, _ = save_report(run, plots=False)
            self.assertTrue((output / "test_run.deploynose").is_file())
            self.assertEqual(list(output.glob("*.png")), [])


if __name__ == "__main__":
    unittest.main()
