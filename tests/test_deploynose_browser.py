"""Browser interaction checks; skipped when Playwright or a local Chromium is absent."""

import base64
import csv
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from g1_deploy.bundle import write_bundle


class DeploynoseBrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise unittest.SkipTest("install Playwright to run browser tests")
        executable = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE") or shutil.which("google-chrome") or shutil.which("chromium")
        if not executable:
            raise unittest.SkipTest("a local Chromium browser is required")
        cls.playwright = sync_playwright().start()
        cls.browser = cls.playwright.chromium.launch(executable_path=executable, headless=True,
                                                     args=["--no-sandbox"])
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        t = np.arange(1000) * 0.002
        q = np.sin(t[:, None] * 4 + np.arange(29)[None, :]).astype(np.float32)
        q[345, 3] = 8.125
        cls.data = dict(t=t, q=q, dq=q * 2, tau_est=q * 3, robot_interval_ms=np.r_[np.nan, np.full(999, 2)],
                        control_t=t[::10], command=np.zeros((100, 3)), heading_error=np.zeros(100),
                        orientation_rpy=np.zeros((1000, 3)), control_interval_ms=np.r_[np.nan, np.full(99, 20)],
                        state_age_ms=np.ones(100), target_t=t[::10], target_q=q[::10])
        cls.metadata = dict(motor_names=[f"joint_{i}" for i in range(29)],
                            arguments={"policy": "browser_test", "sim": "sync"})
        cls.summary = dict(recorded_seconds=t[-1], clean_shutdown=True, dropped={},
                           events=[{"time_s": 0.1, "name": "policy_start"}, {"time_s": 0.8, "name": "marker"}])
        cls.bundle = write_bundle(cls.root / "test.deploynose", cls.metadata, cls.summary, cls.data)
        cls.url = (Path(__file__).resolve().parents[1] / "g1_deploy/deploynose/index.html").as_uri()

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()
        cls.temp.cleanup()

    def setUp(self):
        self.page = self.browser.new_page(viewport={"width": 1440, "height": 1000}, accept_downloads=True)
        self.errors, self.network = [], []
        self.page.on("pageerror", lambda error: self.errors.append(str(error)))
        self.page.on("request", lambda request: self.network.append(request.url) if request.url.startswith("http") else None)
        self.page.goto(self.url)

    def tearDown(self):
        self.page.close()
        self.assertEqual(self.errors, [])
        self.assertEqual(self.network, [])

    def load(self, path=None):
        self.page.locator("#file-input").set_input_files(str(path or self.bundle))
        self.page.locator("#loading").wait_for(state="hidden")

    def test_drop_opens_data_and_all_views_handle_missing_optional_signals(self):
        encoded = base64.b64encode(self.bundle.read_bytes()).decode()
        self.page.evaluate("""encoded => {
            const bytes = Uint8Array.from(atob(encoded), c => c.charCodeAt(0));
            const transfer = new DataTransfer();
            transfer.items.add(new File([bytes], 'dragged.deploynose'));
            window.dispatchEvent(new DragEvent('drop', {dataTransfer: transfer, bubbles: true}));
        }""", encoded)
        self.page.locator("#workspace").wait_for(state="visible")
        self.assertIn("dragged.deploynose", self.page.locator("#run-title").inner_text())
        for view in ("joints", "imu", "timing", "policy", "events"):
            self.page.locator(f'button[data-view="{view}"]').click()
        self.assertIn("marker", self.page.locator(".events-table").inner_text())
        self.page.get_by_role("button", name="0.800", exact=True).click()
        self.assertEqual(self.page.locator("#page-title").inner_text(), "部署总览")

    def test_zoom_comparison_and_csv_keep_exact_samples(self):
        self.load()
        self.page.locator('button[data-view="joints"]').click()
        self.page.locator("#compare-select").select_option("9")
        self.assertIn("joint_9", self.page.locator(".legend").first.inner_text())
        self.page.locator("#range-start").fill("0.4")
        self.page.locator("#range-end").fill("0.9")
        self.page.locator("#apply-range").click()
        with self.page.expect_download() as download:
            self.page.get_by_role("button", name="关节位置与目标 导出 CSV").click()
        rows = list(csv.DictReader(Path(download.value.path()).read_text(encoding="utf-8-sig").splitlines()))
        measured = [row for row in rows if row["series"] == "joint_3 / 反馈"]
        expected = self.data["q"][(self.data["t"] >= 0.4) & (self.data["t"] <= 0.9), 3]
        np.testing.assert_array_equal([float(row["value"]) for row in measured], expected)
        self.assertEqual(max(float(row["value"]) for row in measured), 8.125)
        with self.page.expect_download() as download:
            self.page.get_by_role("button", name="关节位置与目标 导出 PNG").click()
        self.assertEqual(Path(download.value.path()).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        self.page.locator("#reset-range").click()
        before = float(self.page.locator("#range-end").input_value())
        plot = self.page.locator(".overlay").first
        plot.hover()
        self.page.mouse.wheel(0, -120)
        self.page.wait_for_timeout(100)
        self.assertLess(float(self.page.locator("#range-end").input_value()), before)
        self.assertTrue(self.page.locator(".chart-tooltip").first.is_visible())

    def test_bad_file_keeps_previous_run_and_reports_error(self):
        self.load()
        title = self.page.locator("#run-title").inner_text()
        broken = self.root / "broken.deploynose"
        broken.write_bytes(self.bundle.read_bytes()[:-10])
        self.load(broken)
        self.assertTrue(self.page.locator("#message").is_visible())
        self.assertEqual(self.page.locator("#run-title").inner_text(), title)
        mismatch = write_bundle(self.root / "mismatch.deploynose", self.metadata, self.summary,
                                dict(self.data, q=self.data["q"][:-1]))
        self.load(mismatch)
        self.assertIn("不匹配", self.page.locator("#message").inner_text())

    def test_untrusted_labels_are_text_and_incomplete_run_is_marked(self):
        name = '<img src=x onerror="window.injected=true">'
        metadata = dict(self.metadata, motor_names=[name] * 29)
        summary = dict(self.summary, clean_shutdown=False, dropped={"lowstate": 7})
        path = write_bundle(self.root / "untrusted.deploynose", metadata, summary, self.data)
        self.load(path)
        self.assertEqual(self.page.locator("#integrity-badge").inner_text(), "记录存在缺口")
        self.assertIn("7", self.page.locator("#notices").inner_text())
        self.page.locator('button[data-view="joints"]').click()
        self.assertFalse(self.page.evaluate("Boolean(window.injected)"))
        self.assertEqual(self.page.locator("img").count(), 0)

    def test_pixel_reduction_preserves_a_one_sample_spike(self):
        visible = self.page.evaluate("""() => {
            const parent = document.createElement('div'); document.body.append(parent);
            parent.style.width = '400px';
            const x = Float64Array.from({length: 20000}, (_, i) => i / 1000);
            const y = new Float64Array(20000); y[12345] = 100;
            const chart = new DeploynoseCharts.TraceChart(parent, {title: 'spike', unit: '', series: [{x, y, name:'spike'}]},
                {range:[0,20], events:[], cursor:null, hover() {}, setRange() {}, reset() {}});
            chart.draw();
            const px = Math.round(chart.x(12.345)), py = Math.round(chart.y(100));
            const pixels = chart.canvas.getContext('2d').getImageData(px-2,py-2,5,5).data;
            let green = 0;
            for(let i=0;i<pixels.length;i+=4) if(pixels[i+1]>pixels[i]*1.3 && pixels[i+1]>pixels[i+2]) green++;
            chart.destroy(); parent.remove(); return green;
        }""")
        self.assertGreater(visible, 0)

    def test_small_screen_has_no_horizontal_overflow(self):
        self.page.set_viewport_size({"width": 390, "height": 844})
        self.load()
        self.page.locator('button[data-view="joints"]').click()
        self.assertLessEqual(self.page.evaluate("document.documentElement.scrollWidth"), 391)


if __name__ == "__main__":
    unittest.main()
