"""Upload-page isolation using temporary fixtures, never model findings."""
from hashlib import sha256
from pathlib import Path
import io
import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
CLEANED = ROOT / 'data' / 'dataset_preprocessed_v3_clean.csv'
C17 = 'Baseus C17 Type-C Earphones'
RATED = b'product_name,review,stars,date\nEarphone A,bagus sound,5,2024-01-01\nEarphone A,rosak,1,2024-02-01\nEarphone B,biasa,3,\nEarphone B,unrated,,\n'


def widget(elements, label):
    return next(item for item in elements if item.label == label)


def section(app, name):
    return next(block for kind in ['flex_container', 'vertical'] for block in app.get(kind) if block.key == f'section_{name}')


class FileFixture(io.BytesIO):
    def __init__(self, content, name='temporary-test.csv'):
        super().__init__(content)
        self.name = name


class UploadPageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {'SHOPEE_RESULTS_DIR': self.temp.name,
                                          'SENTILOKAL_RESEARCH_DATA_PATH': str(CLEANED),
                                          'SENTILOKAL_RESEARCH_DATA_SHA256': sha256(CLEANED.read_bytes()).hexdigest() if CLEANED.exists() else '0' * 64})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def app(self, upload=None):
        app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30)
        app.session_state['application_module'] = 'Upload & Analyze'
        with patch('streamlit.file_uploader', return_value=upload):
            app.run()
        self.assertNoException(app)
        return app

    def assertNoException(self, app):
        self.assertEqual([item.value for item in app.exception], [])

    def metrics(self, app):
        return {item.label: item.value for item in app.metric}

    def test_empty_upload_page_is_separate_and_does_not_require_research_data(self):
        with patch.dict(os.environ, {'SENTILOKAL_RESEARCH_DATA_PATH': '/tmp/nonexistent-research-for-upload-test.csv'}):
            app = self.app()
        self.assertEqual([item.value for item in app.title], ['Upload & Analyze'])
        self.assertFalse(any(block.key == 'section_overview' for block in app.get('flex_container')))
        self.assertEqual(len(app.metric), 0)
        self.assertTrue(any('Your analysis starts here' in item.value for item in app.markdown))
        self.assertEqual(len(app.error), 0)

    def test_rated_upload_counts_all_rows_but_binary_shares_exclude_three_stars(self):
        app = self.app(FileFixture(RATED))
        self.assertEqual(self.metrics(app), {'Total uploaded reviews': '4', 'Uploaded products': '2',
                         'Positive uploaded reviews': '1', 'Negative uploaded reviews': '1',
                         'Positive share · binary ratings': '50.00%', 'Negative share · binary ratings': '50.00%'})
        self.assertEqual(len(app.get('plotly_chart')), 4)
        self.assertEqual(len(app.dataframe[0].value), 4)
        self.assertEqual(app.dataframe[0].value['Rating-derived sentiment'].dropna().tolist(), ['Positive', 'Negative'])
        self.assertTrue(any('2024' in item.value for item in app.caption))
        widget(app.text_input, 'Search uploaded reviews').set_value('BaGuS').run()
        self.assertNoException(app)
        self.assertEqual(app.dataframe[0].value['Original review'].tolist(), ['bagus sound'])

    def test_text_only_upload_never_generates_sentiment_or_metrics(self):
        app = self.app(FileFixture(b'review\nbagus\nrosak\n'))
        self.assertEqual(self.metrics(app)['Total uploaded reviews'], '2')
        self.assertEqual(self.metrics(app)['Positive uploaded reviews'], '—')
        self.assertEqual(self.metrics(app)['Negative uploaded reviews'], '—')
        self.assertEqual(self.metrics(app)['Reviews with original comments'], '2')
        self.assertEqual(self.metrics(app)['Median review length · words'], '1.0')
        self.assertEqual(len(app.get('plotly_chart')), 0)
        self.assertTrue(app.dataframe[0].value['Rating-derived sentiment'].isna().all())
        self.assertTrue(any('No sentiment classifications have been generated' in item.value for item in app.info))

    def test_clear_removes_upload_and_restores_empty_state(self):
        fixture = FileFixture(RATED)
        def upload_for_generation(*args, **kwargs):
            return fixture if kwargs['key'] == 'upload_file_0' else None
        with patch('streamlit.file_uploader', side_effect=upload_for_generation):
            app = self.app(fixture)
            widget(app.button, 'Clear Uploaded Dataset').click().run()
        self.assertNoException(app)
        self.assertEqual(len(app.metric), 0)
        self.assertNotIn('user_upload_dataset', app.session_state)
        self.assertNotIn('user_upload_filename', app.session_state)
        self.assertNotIn('user_upload_digest', app.session_state)
        self.assertTrue(any('Your analysis starts here' in item.value for item in app.markdown))

    def test_both_modules_show_pending_prediction_without_loading_a_model(self):
        with patch('dashboard.inference.load_trained_model') as loader:
            app = self.app()
            self.assertTrue(any('Model prediction feature coming after training' in item.value for item in app.info))
            self.assertEqual(widget(app.selectbox, 'Application module').options, ['Baseus Research Dashboard', 'Upload & Analyze'])
            widget(app.selectbox, 'Application module').set_value('Baseus Research Dashboard').run()
            self.assertNoException(app)
            self.assertTrue(any('Model prediction feature coming after training' in item.value for item in app.info))
            loader.assert_not_called()

    def test_invalid_csv_shows_helpful_message(self):
        app = self.app(FileFixture(b'customer,rating\nA,5\n'))
        self.assertTrue(any('review text column is required' in item.value for item in app.error))
        self.assertEqual(len(app.metric), 0)
        self.assertIsNone(app.session_state['user_upload_dataset'])

    def test_uploads_are_not_shared_between_sessions(self):
        first = self.app(FileFixture(RATED))
        second = self.app()
        self.assertEqual(self.metrics(first)['Total uploaded reviews'], '4')
        self.assertEqual(len(second.metric), 0)
        self.assertNotIn('user_upload_dataset', second.session_state)

    @unittest.skipUnless(CLEANED.exists(), 'Private approved research CSV not provisioned')
    def test_upload_clear_and_navigation_never_replace_research_or_its_filters(self):
        before = sha256(CLEANED.read_bytes()).hexdigest()
        app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
        self.assertEqual(len(app.get('file_uploader')), 0)
        initial = {item.label: item.value for item in section(app, 'overview').metric}
        self.assertEqual(initial['Total reviews'], '19,295')
        self.assertEqual(initial['Products'], '5')
        widget(app.multiselect, 'Products').set_value([C17]).run()
        widget(app.text_input, 'Search original reviews').set_value('BaGuS').run()
        research_metrics = {item.label: item.value for item in section(app, 'overview').metric}
        fixture = FileFixture(RATED)
        def upload_for_generation(*args, **kwargs):
            return fixture if kwargs['key'] == 'upload_file_0' else None
        with patch('streamlit.file_uploader', side_effect=upload_for_generation):
            widget(app.selectbox, 'Application module').set_value('Upload & Analyze').run()
            self.assertNoException(app)
            self.assertEqual(self.metrics(app)['Total uploaded reviews'], '4')
            widget(app.multiselect, 'Uploaded products').set_value(['Earphone B']).run()
            self.assertEqual(self.metrics(app)['Total uploaded reviews'], '2')
            widget(app.selectbox, 'Application module').set_value('Baseus Research Dashboard').run()
            self.assertNoException(app)
            self.assertEqual({item.label: item.value for item in section(app, 'overview').metric}, research_metrics)
            self.assertEqual(widget(app.multiselect, 'Products').value, [C17])
            self.assertEqual(widget(app.text_input, 'Search original reviews').value, 'BaGuS')
            self.assertEqual(len(app.get('file_uploader')), 0)
            widget(app.selectbox, 'Application module').set_value('Upload & Analyze').run()
            self.assertEqual(widget(app.multiselect, 'Uploaded products').value, ['Earphone B'])
            widget(app.button, 'Clear Uploaded Dataset').click().run()
            self.assertEqual(len(app.metric), 0)
            widget(app.selectbox, 'Application module').set_value('Baseus Research Dashboard').run()
        self.assertNoException(app)
        self.assertEqual({item.label: item.value for item in section(app, 'overview').metric}, research_metrics)
        widget(app.button, 'Reset filters').click().run()
        self.assertEqual({item.label: item.value for item in section(app, 'overview').metric}, initial)
        self.assertEqual(sha256(CLEANED.read_bytes()).hexdigest(), before)


if __name__ == '__main__':
    unittest.main()
