"""Run the app in an isolated code-only deployment, then provision read-only data.

No private CSVs are copied into the code checkout. A separate temporary volume
simulates owner-provisioned storage; no remote dataset is downloaded or uploaded.
"""
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
CLEANED = ROOT / 'data' / 'dataset_preprocessed_v3_clean.csv'


class FreshDeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.checkout = Path(self.temp.name) / 'code-only-checkout'
        self.checkout.mkdir()
        for name in ['app.py', 'dashboard', 'assets', '.streamlit', 'scripts']:
            source = ROOT / name
            target = self.checkout / name
            if source.is_dir():
                shutil.copytree(source, target, ignore=shutil.ignore_patterns('__pycache__', 'secrets.toml'))
            else:
                shutil.copy2(source, target)
        self.env = dict(os.environ)
        for key in ['SENTILOKAL_RESEARCH_DATA_PATH', 'SENTILOKAL_RESEARCH_DATA_SHA256', 'SHOPEE_DATA_PATH', 'SHOPEE_RESULTS_DIR']:
            self.env.pop(key, None)

    def tearDown(self):
        self.temp.cleanup()

    def run_app(self):
        code = '''import json
from streamlit.testing.v1 import AppTest
app = AppTest.from_file('app.py', default_timeout=30).run()
metrics = {item.label:item.value for block in app.get('flex_container') if block.key=='section_overview' for item in block.metric}
print('DEPLOYMENT_RESULT=' + json.dumps({'errors':[item.value for item in app.exception], 'metrics':metrics, 'warnings':[item.value for item in app.warning], 'uploaders':len(app.get('file_uploader')), 'charts':len(app.get('plotly_chart')), 'setup':any('Research dataset setup' in item.label for item in app.expander)}))
'''
        result = subprocess.run([sys.executable, '-c', code], cwd=self.checkout, env=self.env, capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        payload = next(line.removeprefix('DEPLOYMENT_RESULT=') for line in result.stdout.splitlines() if line.startswith('DEPLOYMENT_RESULT='))
        return json.loads(payload)

    def test_code_only_deployment_has_explicit_setup_state_not_fake_results(self):
        self.assertFalse((self.checkout / 'data').exists())
        result = self.run_app()
        self.assertEqual(result['errors'], [])
        self.assertEqual(result['metrics'], {})
        self.assertEqual(result['charts'], 0)
        self.assertEqual(result['uploaders'], 0)
        self.assertTrue(result['setup'])
        self.assertTrue(any('official research dataset is not available' in message for message in result['warnings']))
        check = subprocess.run([sys.executable, 'scripts/check_research_data.py'], cwd=self.checkout, env=self.env, capture_output=True, text=True, timeout=30)
        self.assertEqual(check.returncode, 1)
        self.assertIn('setup required', check.stderr)

    @unittest.skipUnless(CLEANED.exists(), 'Private approved dataset needed for real deployment integration')
    def test_current_branch_data_asset_loads_automatically_after_fresh_deployment(self):
        # The owner has now committed this exact approved file to the branch.
        # Mirror the present repository layout, without alternate dataset copies.
        (self.checkout / 'data').mkdir()
        target = self.checkout / 'data' / CLEANED.name
        shutil.copy2(CLEANED, target)
        before = sha256(target.read_bytes()).hexdigest()
        result = self.run_app()
        self.assertEqual(result['errors'], [])
        self.assertFalse(result['setup'])
        self.assertEqual(result['metrics']['Total reviews'], '19,295')
        self.assertEqual(result['metrics']['Products'], '5')
        self.assertEqual(result['charts'], 7)
        self.assertEqual(result['uploaders'], 0)
        self.assertEqual(sha256(target.read_bytes()).hexdigest(), before)

    @unittest.skipUnless(CLEANED.exists(), 'Private approved dataset needed for real deployment integration')
    def test_code_only_deployment_loads_actual_data_from_readonly_volume(self):
        volume = Path(self.temp.name) / 'readonly-private-volume'
        volume.mkdir()
        path = volume / CLEANED.name
        shutil.copy2(CLEANED, path)
        path.chmod(0o444)
        before = sha256(path.read_bytes()).hexdigest()
        self.env['SENTILOKAL_RESEARCH_DATA_PATH'] = str(path)
        result = self.run_app()
        self.assertEqual(result['errors'], [])
        self.assertEqual(result['metrics']['Total reviews'], '19,295')
        self.assertEqual(result['metrics']['Products'], '5')
        self.assertEqual(result['charts'], 7)
        self.assertEqual(result['uploaders'], 0)
        self.assertFalse(result['setup'])
        self.assertFalse((self.checkout / 'data').exists())
        self.assertEqual(sha256(path.read_bytes()).hexdigest(), before)
        self.assertEqual(path.stat().st_mode & 0o222, 0)
        check = subprocess.run([sys.executable, 'scripts/check_research_data.py'], cwd=self.checkout, env=self.env, capture_output=True, text=True, timeout=30)
        self.assertEqual(check.returncode, 0, check.stderr)
        self.assertIn('Approved research data ready', check.stdout)
        self.assertIn('14 Jul 2020', check.stdout)


if __name__ == '__main__':
    unittest.main()
