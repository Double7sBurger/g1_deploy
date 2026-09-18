import unittest

from g1_deploy.timing import PeriodicDeadline


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []
        self.oversleep = 0.0

    def __call__(self):
        return self.now

    def sleep(self, delay):
        self.sleeps.append(delay)
        self.now += delay + self.oversleep


class PeriodicDeadlineTests(unittest.TestCase):
    def make_pacer(self, period=0.02):
        clock = FakeClock()
        return clock, PeriodicDeadline(period, clock=clock, sleep=clock.sleep)

    def test_normal_work_keeps_frequency_without_adding_compute_time(self):
        clock, pacer = self.make_pacer()
        for _ in range(100):
            clock.now += 0.003
            pacer.wait()
        self.assertAlmostEqual(clock.now, 2.0)
        self.assertEqual(pacer.overruns, 0)

    def test_long_stall_does_not_replay_old_periods(self):
        for period in (0.002, 0.02):
            with self.subTest(period=period):
                clock, pacer = self.make_pacer(period)
                clock.now = 0.4
                pacer.wait()
                self.assertEqual(pacer.overruns, 1)
                for _ in range(10):
                    clock.now += period / 10
                    pacer.wait()
                    self.assertAlmostEqual(clock.sleeps[-1], 0.9 * period)
                self.assertAlmostEqual(clock.now, 0.4 + 10 * period)

    def test_late_wakeup_does_not_create_catchup_debt(self):
        clock, pacer = self.make_pacer()
        clock.oversleep = 0.4
        pacer.wait()
        self.assertEqual(pacer.overruns, 1)
        clock.oversleep = 0
        pacer.wait()
        self.assertAlmostEqual(clock.sleeps[-1], 0.02)

    def test_small_sleep_jitter_does_not_accumulate_drift(self):
        clock, pacer = self.make_pacer()
        clock.oversleep = 0.0001
        for _ in range(100):
            pacer.wait()
        self.assertAlmostEqual(clock.now, 2.0001)
        self.assertEqual(pacer.overruns, 0)

    def test_invalid_periods(self):
        for period in (0, -1, float('nan'), float('inf')):
            with self.subTest(period=period), self.assertRaises(ValueError):
                PeriodicDeadline(period)
