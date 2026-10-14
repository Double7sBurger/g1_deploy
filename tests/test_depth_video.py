import importlib.util
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts.view_depth import VideoRecorder, colourise, run_window


@unittest.skipUnless(importlib.util.find_spec("cv2"), "requires the video extra")
class DepthVideoTests(unittest.TestCase):
    def read_video(self, path):
        import cv2

        capture = cv2.VideoCapture(str(path))
        frames = []
        try:
            self.assertTrue(capture.isOpened())
            fps = capture.get(cv2.CAP_PROP_FPS)
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(frame)
        finally:
            capture.release()
        return fps, frames

    def test_gray_video_preserves_elapsed_time_and_refuses_overwrite(self):
        near = colourise(np.full((38, 64), 0.3), 3.0, gray=True)
        far = colourise(np.full((38, 64), 2.7), 3.0, gray=True)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "depth.mp4"
            recorder = VideoRecorder(str(path), 64, 38, 20, 2)
            try:
                with patch("scripts.view_depth.time.monotonic", side_effect=[50.0, 50.11, 50.26]):
                    recorder.write(near)
                    recorder.write(far)
                    recorder.write(near)
            finally:
                recorder.close()
            recorder.close()
            fps, frames = self.read_video(path)
            self.assertAlmostEqual(fps, 20)
            self.assertEqual(len(frames), 6)
            self.assertEqual(frames[0].shape, (76, 128, 3))
            for index, expected in enumerate([229, 229, 25, 25, 25, 229]):
                self.assertAlmostEqual(float(frames[index].mean()), expected, delta=10)
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                VideoRecorder(str(path), 64, 38, 20, 2)
            self.assertEqual(path.read_bytes(), before)

    def test_ctrl_c_finalizes_a_playable_video(self):
        code = """
import os, signal, sys, threading
from unittest.mock import patch
import cv2
import numpy as np
from scripts import view_depth

class Receiver:
    received = 1
    dropped = 0
    def __init__(self, *args, **kwargs):
        pass
    def wait_for_frame(self, timeout):
        threading.Timer(0.6, lambda: os.kill(os.getpid(), signal.SIGINT)).start()
    def latest(self):
        self.received += 1
        return np.full((38, 64), 0.5 + (self.received % 10) / 10), 0.0
    def close(self):
        print('receiver closed')

sys.argv = ['view_depth', '--record', sys.argv[1], '--gray', '--hz', '20', '--zoom', '2']
with patch.object(view_depth, 'DepthReceiver', Receiver):
    raise SystemExit(view_depth.main())
"""
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "interrupt.mp4"
            result = subprocess.run([sys.executable, "-B", "-c", code, str(path)],
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("[saved]", result.stdout)
            self.assertIn("receiver closed", result.stdout)
            fps, frames = self.read_video(path)
            self.assertAlmostEqual(fps, 20)
            self.assertGreaterEqual(len(frames), 5)


class WindowStopTests(unittest.TestCase):
    def test_ctrl_c_exits_tk_loop_and_restores_signal_handler(self):
        class Widget:
            def __init__(self, *args, **kwargs):
                pass
            def pack(self, **kwargs):
                pass
            def configure(self, **kwargs):
                pass

        class Window:
            quit_called = False
            destroyed = False
            def title(self, text):
                pass
            def after(self, delay, callback):
                self.callback = callback
            def mainloop(self):
                signal.raise_signal(signal.SIGINT)
                self.callback()
            def quit(self):
                self.quit_called = True
            def destroy(self):
                self.destroyed = True

        root = Window()
        receiver = SimpleNamespace(latest=lambda: (np.ones((38, 64)), 0.0), received=1, dropped=0)
        args = SimpleNamespace(port=5602, max_range=3.0, gray=True, zoom=2, hz=20)
        original_handler = signal.getsignal(signal.SIGINT)
        with patch("tkinter.Tk", return_value=root), patch("tkinter.Label", Widget), \
                patch("PIL.ImageTk.PhotoImage", return_value=object()):
            self.assertEqual(run_window(receiver, args), 0)
        self.assertTrue(root.quit_called)
        self.assertTrue(root.destroyed)
        self.assertIs(signal.getsignal(signal.SIGINT), original_handler)


if __name__ == "__main__":
    unittest.main()
