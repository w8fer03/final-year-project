"""Approved-data provisioning checks with local temporary read-only fixtures."""
from hashlib import sha256
from pathlib import Path
import tempfile
import unittest

from dashboard.research import APPROVED_SHA256, DEFAULT_FILENAME, ResearchDataError, read_research_source


class ResearchSourceTests(unittest.TestCase):
    def test_missing_code_only_deployment_explains_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ResearchDataError, 'project owner must complete'):
                read_research_source(Path(directory), {})

    def test_ignores_other_csvs_in_data_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'data').mkdir()
            (root / 'data' / 'dataset v3 labelled 2.csv').write_text('review\ngood\n')
            with self.assertRaises(ResearchDataError):
                read_research_source(root, {})

    def test_approved_readonly_server_path_and_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'private-volume.csv'
            content = b'review,stars\ngood,5\n'
            source.write_bytes(content)
            source.chmod(0o444)
            config = {'SENTILOKAL_RESEARCH_DATA_PATH': str(source), 'SENTILOKAL_RESEARCH_DATA_SHA256': sha256(content).hexdigest()}
            data = read_research_source(root, config)
            self.assertEqual(data.content, content)
            self.assertEqual(source.read_bytes(), content)
            self.assertEqual(source.stat().st_mode & 0o222, 0)

    def test_mismatched_or_invalid_checksum_never_silently_falls_back(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'other.csv'
            source.write_text('review\nwrong data\n')
            with self.assertRaisesRegex(ResearchDataError, 'does not match'):
                read_research_source(root, {'SENTILOKAL_RESEARCH_DATA_PATH': str(source)})
            with self.assertRaisesRegex(ResearchDataError, 'checksum configuration is invalid'):
                read_research_source(root, {'SENTILOKAL_RESEARCH_DATA_SHA256': 'invalid'})

    def test_default_approved_file_in_current_workspace_is_read_only(self):
        root = Path(__file__).resolve().parents[1]
        path = root / 'data' / DEFAULT_FILENAME
        if not path.exists():
            self.skipTest('Private approved dataset not provisioned')
        before = path.read_bytes()
        result = read_research_source(root, {})
        self.assertEqual(result.path, path)
        self.assertEqual(result.sha256, APPROVED_SHA256)
        self.assertEqual(path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
