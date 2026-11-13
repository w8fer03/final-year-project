"""Final evaluation UI and explicit upload inference (doubles, not FYP findings)."""
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from dashboard.prediction_backend import PredictionOutput
from dashboard.version_b import EVALUATIONS, ROOT, deployment_config


def widget(elements, label):
    return next(item for item in elements if item.label == label)


class UploadedFile(io.BytesIO):
    name = 'temporary-adapter-test.csv'


class FinalInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.environment = patch.dict(os.environ, {'SENTILOKAL_EVALUATION_DIR':str(EVALUATIONS),
                                                  'SENTILOKAL_RESEARCH_PROFILE':'legacy', 'SHOPEE_RESULTS_DIR':self.temp.name})
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.temp.cleanup()

    def check(self, app):
        self.assertEqual([item.value for item in app.exception], [])

    def upload_app(self, content):
        app = AppTest.from_file(str(ROOT/'app.py'), default_timeout=30)
        app.session_state.application_module = 'Upload & Analyze'
        with patch('streamlit.file_uploader',return_value=UploadedFile(content)):
            app.run()
        self.check(app)
        return app

    def test_actual_final_exports_display_eight_rows_and_selected_recorded_matrix(self):
        app = AppTest.from_file(str(ROOT/'app.py'), default_timeout=30).run()
        self.check(app)
        tables = [item.value for item in app.dataframe if 'Macro F1' in item.value]
        self.assertEqual([len(frame) for frame in tables], [4,4])
        self.assertEqual(set(tables[0]['Test split']), {'80:20'})
        self.assertEqual(set(tables[1]['Test split']), {'70:30'})
        widget(app.selectbox,'Evaluation test split').set_value('70_30').run()
        widget(app.selectbox,'Evaluation architecture').set_value('finetuned_hybrid').run()
        self.check(app)
        metrics = {item.label:item.value for item in app.metric}
        self.assertEqual(metrics['Macro F1 · primary'],'0.71930')
        self.assertEqual(widget(app.selectbox,'Recorded comparison metric').value,'macro_f1')
        specifications = [json.loads(chart.proto.spec) for chart in app.get('plotly_chart')]
        matrix = next(spec for spec in specifications if spec['data'][0]['type']=='heatmap')['data'][0]
        original = pd.read_csv(EVALUATIONS/'day3_all_models_confusion.csv')
        row = original[(original.architecture=='finetuned_hybrid') & (original.split=='70_30')].iloc[0]
        self.assertEqual(matrix['z'], [[int(row.actual_negative_pred_negative),int(row.actual_negative_pred_positive)],
                                       [int(row.actual_positive_pred_negative),int(row.actual_positive_pred_positive)]])

    def test_final_report_never_loads_or_recalculates_legacy_test_evaluations(self):
        (Path(self.temp.name)/'evaluation_legacy.json').write_text('{}')
        with patch('dashboard.artifacts.load_evaluation',side_effect=AssertionError('Held-out re-evaluation is forbidden')) as load:
            app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
        self.check(app)
        load.assert_not_called()
        self.assertTrue(any(item.label=='Macro F1 · primary' for item in app.metric))

    def test_no_inference_occurs_until_requested_and_model_switch_invalidates_outputs(self):
        with patch('dashboard.prediction_controls.readiness',return_value=(True,'TEST DOUBLE — not real model validation')):
            app = self.upload_app(b'review,stars\nsame,1\nbad,5\nsame,1\n')
            with patch('dashboard.prediction_backend.get_predictor') as loader:
                widget(app.selectbox,'Prediction model').set_value('tfidf_xgb').run()
                loader.assert_not_called()
                model = Mock()
                model.predict.return_value = PredictionOutput(np.array([1,0,1]),np.array([.1,.8,.1]),'tfidf_xgb',.28)
                loader.return_value = model
                widget(app.button,'Analyze Reviews').click().run()
                self.check(app)
                model.predict.assert_called_once_with(['same','bad','same'])
                table = next(item.value for item in app.dataframe if 'Original review' in item.value)
                self.assertEqual(table['Original review'].tolist(),['same','bad','same'])
                self.assertEqual(table['Rating-derived sentiment'].tolist(),['Negative','Positive','Negative'])
                self.assertEqual(table['Model-predicted sentiment'].tolist(),['Positive','Negative','Positive'])
                widget(app.selectbox,'Model prediction filter').set_value('Predicted negative').run()
                table = next(item.value for item in app.dataframe if 'Original review' in item.value)
                self.assertEqual(table['Original review'].tolist(),['bad'])
                widget(app.selectbox,'Prediction model').set_value('frozen_hybrid').run()
                self.check(app)
                self.assertNotIn('user_upload_predictions',app.session_state)

    def test_text_only_batch_export_has_actual_labels_without_rating_labels(self):
        with patch('dashboard.prediction_controls.readiness',return_value=(True,'TEST DOUBLE — not real model validation')):
            app = self.upload_app(b'review\ngood\nbad\ngood\n')
            widget(app.selectbox,'Prediction model').set_value('tfidf_xgb').run()
            model = Mock()
            model.predict.return_value = PredictionOutput(np.array([1,0,1]),np.array([.1,.8,.1]),'tfidf_xgb',.28)
            import streamlit as st
            with patch('dashboard.prediction_backend.get_predictor',return_value=model), patch('streamlit.download_button',wraps=st.download_button) as download:
                widget(app.button,'Analyze Reviews').click().run()
                call = next(call for call in download.call_args_list if call.kwargs.get('key')=='upload_export')
                exported = pd.read_csv(io.BytesIO(call.args[1]))
            self.check(app)
            self.assertEqual(exported.review_text.tolist(),['good','bad','good'])
            self.assertEqual(exported.predicted_label.tolist(),[1,0,1])
            self.assertTrue(exported.sentiment.isna().all())
            self.assertNotIn('confidence',exported)
            self.assertNotIn('source_row_id',exported)
            widget(app.text_input,'Search uploaded reviews').set_value('bad').run()
            self.check(app)
            table = next(item.value for item in app.dataframe if 'Original review' in item.value)
            self.assertEqual(table['Model-predicted sentiment'].tolist(),['Negative'])

    def test_unavailable_models_leave_upload_exploration_usable(self):
        with patch.dict(os.environ,{'SENTILOKAL_MODEL_DIR':self.temp.name}):
            app = self.upload_app(b'review\noriginal text\n')
            self.assertTrue(widget(app.button,'Analyze Reviews').disabled)
            self.assertTrue(widget(app.button,'Predict this review').disabled)
            self.assertEqual(app.dataframe[0].value['Original review'].tolist(),['original text'])

    @unittest.skipUnless((ROOT/'models/version_b/version_B_model_ready.csv').exists(),'Private Version B source not provisioned')
    def test_explicit_version_b_profile_loads_five_products_18705_rows_without_upload(self):
        with patch.dict(os.environ, {'SENTILOKAL_RESEARCH_PROFILE':'version_b','SENTILOKAL_MODEL_DIR':str(ROOT/'models/version_b')}):
            app = AppTest.from_file(str(ROOT/'app.py'), default_timeout=30).run()
        self.check(app)
        metrics = {item.label:item.value for item in app.metric}
        self.assertEqual(metrics['Total reviews'],'18,705')
        self.assertEqual(metrics['Positive reviews'],'18,423')
        self.assertEqual(metrics['Products'],'5')
        self.assertEqual(len(app.get('file_uploader')),0)
        self.assertEqual(widget(app.radio,'Navigation').options,['Home','Dashboard Overview','Product Sentiment Analysis','Customer Review Explorer','Model Performance'])


if __name__ == '__main__':
    unittest.main()
