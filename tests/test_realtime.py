import copy
import threading
import unittest
from types import SimpleNamespace

import mujoco
import numpy as np

from g1_deploy.sim.realtime import RealtimePhysics


class FakeEnv:
    def __init__(self, connected=True):
        self.model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body><freejoint/><geom size=".1"/></body></worldbody></mujoco>'
        )
        self.data = mujoco.MjData(self.model)
        self.sim_dt = 0.002
        self.bridge = SimpleNamespace(cmd_received=connected)
        self.published = threading.Event()
        self.stepped = threading.Event()
        self.fail = False

    def publish_only(self):
        self.published.set()

    def sim_step(self):
        if self.fail:
            raise ValueError('test physics failure')
        self.data.time += self.sim_dt
        self.data.qpos[0] = self.data.time
        self.stepped.set()


class RealtimePhysicsTests(unittest.TestCase):
    def test_physics_continues_while_graphics_holds_snapshot(self):
        env = FakeEnv()
        model = copy.copy(env.model)
        data = mujoco.MjData(model)
        worker = RealtimePhysics(env, np.zeros(1))
        worker.start()
        try:
            self.assertTrue(env.stepped.wait(1))
            worker.copy_state(model, data)
            before = data.time
            snapshot = data.qpos.copy()
            env.stepped.clear()
            # Graphics deliberately does not copy/sync/render another frame here.
            self.assertTrue(env.stepped.wait(1))
            self.assertEqual(data.time, before)
            np.testing.assert_array_equal(data.qpos, snapshot)
            worker.copy_state(model, data)
            self.assertGreater(data.time, before)
            worker.check()
        finally:
            worker.stop()

    def test_waiting_for_controller_does_not_step_physics(self):
        env = FakeEnv(connected=False)
        worker = RealtimePhysics(env, np.zeros(1))
        worker.start()
        try:
            self.assertTrue(env.published.wait(1))
            self.assertEqual(env.data.time, 0)
            self.assertEqual(worker.steps, 0)
        finally:
            worker.stop()

    def test_worker_error_reaches_caller(self):
        env = FakeEnv()
        env.fail = True
        worker = RealtimePhysics(env, np.zeros(1))
        worker.start()
        worker._thread.join(timeout=1)
        try:
            with self.assertRaisesRegex(RuntimeError, 'physics worker failed') as caught:
                worker.check()
            self.assertIsInstance(caught.exception.__cause__, ValueError)
        finally:
            worker.stop()
