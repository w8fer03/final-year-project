"""Regression checks for updating charts in a long-running Streamlit process."""
from pathlib import Path
import unittest
from unittest.mock import patch

from dashboard import charts, runtime


class ChartRefreshTests(unittest.TestCase):
    def tearDown(self):
        runtime.refresh_charts()

    def test_missing_callable_in_cached_module_is_repaired(self):
        current = runtime.refresh_charts()
        del current.rating_imbalance
        restored = runtime.refresh_charts()
        self.assertIs(restored, current)
        self.assertIs(restored, charts)
        self.assertTrue(callable(restored.rating_imbalance))

    def test_unchanged_source_does_not_reload_each_rerun(self):
        runtime.refresh_charts()
        original = charts.rating_imbalance
        for _ in range(3):
            self.assertIs(runtime.refresh_charts().rating_imbalance, original)

    def test_changed_source_refreshes_even_without_file_metadata_changes(self):
        runtime.refresh_charts()
        original_source = runtime._CHARTS_PATH.read_bytes()
        changed_source = original_source.replace(b'POSITIVE = "#0E9279"', b'POSITIVE = "#123456"')
        self.assertNotEqual(original_source, changed_source)
        self.assertEqual(len(original_source), len(changed_source))
        original_function = charts.rating_imbalance
        with patch("dashboard.runtime._read_source", return_value=changed_source):
            refreshed = runtime.refresh_charts()
            self.assertIs(refreshed, charts)
            self.assertEqual(refreshed.POSITIVE, "#123456")
            self.assertIsNot(refreshed.rating_imbalance, original_function)
        runtime.refresh_charts()
        self.assertEqual(charts.POSITIVE, "#0E9279")

    def test_first_call_repairs_existing_module_from_before_refresh_support(self):
        runtime.refresh_charts()
        del charts._sentilokal_source_sha256
        original_function = charts.rating_bars
        self.assertIs(runtime.refresh_charts(), charts)
        self.assertIsNot(charts.rating_bars, original_function)

    def test_rejects_a_module_outside_the_local_application(self):
        runtime.refresh_charts()
        with patch.object(charts, "__file__", str(Path("/tmp/unrelated-charts.py"))):
            with self.assertRaisesRegex(ImportError, "local application"):
                runtime.refresh_charts()


if __name__ == "__main__":
    unittest.main()
