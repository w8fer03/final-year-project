"""Small inference controls added to the existing upload interface, session-only."""
from __future__ import annotations

from hashlib import sha256

import pandas as pd
import streamlit as st

from dashboard.prediction_backend import (ARCHITECTURES, _cache_identity, get_predictor,
                                           predict_review_frame, readiness)
from dashboard.version_b import RESOURCES, VersionBError, deployment_config, model_directory

PREDICTION_KEYS = ('user_upload_predictions','user_upload_prediction_identity','user_upload_prediction_error')


def clear_batch_predictions():
    for key in PREDICTION_KEYS:
        st.session_state.pop(key,None)
    st.session_state.pop('user_upload_single_result',None)
    st.session_state.pop('user_upload_single_identity',None)


def render_model_controls():
    try:
        config = deployment_config()
        options = list(ARCHITECTURES)
        default = options.index(config['dashboard_default_architecture'])
    except VersionBError as exc:
        st.warning(str(exc))
        return None, False
    with st.container(border=True,key='panel_upload_model'):
        st.subheader('Trained sentiment model')
        architecture = st.selectbox('Prediction model',options,index=default,format_func=ARCHITECTURES.get,key='upload_model',on_change=clear_batch_predictions)
        available, message = readiness(architecture)
        if available:
            st.success(message)
        else:
            st.warning(message)
        st.caption('Four deployed 80:20 configurations. Default and decision thresholds are fixed by the supplied training-only CV deployment configuration. Held-out scores do not select or tune this model.')
        st.caption(f"Decision: Negative (0) when the negative-class model score is at least {config['models'][architecture]['threshold']!r}; otherwise Positive (1). Scores are uncalibrated, not confidence percentages.")
        with st.expander('Model artifact availability'):
            for key,name in ARCHITECTURES.items():
                ready, detail = readiness(key)
                st.write(f"**{name}** · {'Available locally' if ready else 'Unavailable'}")
                st.caption(detail)
    with st.expander('Analyze one review'):
        text = st.text_area('Original review text',key='upload_single_text',placeholder='Type an original Malay, English, or mixed-language review…')
        artifact_identity = _cache_identity(model_directory(),RESOURCES,architecture)
        identity = sha256((architecture+'\0'+text+'\0'+artifact_identity).encode()).hexdigest()
        if st.session_state.get('user_upload_single_identity') != identity:
            st.session_state.pop('user_upload_single_result',None)
        clicked = st.button('Predict this review',key='upload_predict_single',disabled=not available)
        if clicked and not text.strip():
            st.info('Enter an original review before predicting its sentiment.')
        if clicked and text.strip():
            st.session_state.pop('user_upload_single_result',None)
            try:
                with st.spinner('Analyzing the original review…'):
                    result = predict_review_frame(pd.DataFrame({'review_text':[text]}),architecture)
                st.session_state.user_upload_single_result = result
                st.session_state.user_upload_single_identity = identity
            except Exception as exc:
                st.error(str(exc) if isinstance(exc,VersionBError) else 'The trained model could not run. Check local artifacts and the documented inference dependency versions.')
        result = st.session_state.get('user_upload_single_result')
        if result is not None:
            st.dataframe(result[['review_text','predicted_label','predicted_sentiment','negative_score']],hide_index=True,use_container_width=True)
            st.caption('The score is P(Negative) from the model, without a calibration claim.')
            st.download_button('Download single-review prediction',result.to_csv(index=False).encode('utf-8-sig'),'single_review_prediction.csv','text/csv',key='upload_single_download')
    return architecture, available


def render_batch_action(records, architecture, available):
    if architecture is None:
        return
    identity = (st.session_state.get('user_upload_digest'), architecture, _cache_identity(model_directory(),RESOURCES,architecture))
    if st.session_state.get('user_upload_prediction_identity') != identity:
        for key in PREDICTION_KEYS:
            st.session_state.pop(key,None)
    if st.button('Analyze Reviews',key='upload_analyze_reviews',type='primary',disabled=not available):
        for key in PREDICTION_KEYS:
            st.session_state.pop(key,None)
        try:
            with st.spinner('Running the saved model on all original uploaded reviews…'):
                result = predict_review_frame(records,architecture)
            # One-to-one output of this exact input, not a cross-dataset positional join.
            result.index = records.index
            st.session_state.user_upload_predictions = result
            st.session_state.user_upload_prediction_identity = identity
        except Exception as exc:
            st.session_state.user_upload_prediction_error = str(exc) if isinstance(exc,VersionBError) else 'The saved model could not run. Check artifacts and inference dependency versions; no partial predictions were accepted.'
    if st.session_state.get('user_upload_prediction_error'):
        st.error(st.session_state.user_upload_prediction_error)
    st.caption('Analyze Reviews uses every original row in this CSV, including duplicates and rows without ratings. No balancing, star-rating features, or model fitting is performed. Search and display filters are applied afterward.')


def prediction_summary(selected, charts):
    predictions = st.session_state.get('user_upload_predictions')
    if predictions is None:
        return
    selected = predictions.loc[selected.index]
    with st.container(border=True,key='panel_upload_prediction_summary'):
        st.subheader('Model-predicted sentiment')
        left, right = st.columns(2)
        left.metric('Predicted positive reviews',f"{selected.predicted_label.eq(1).sum():,}")
        right.metric('Predicted negative reviews',f"{selected.predicted_label.eq(0).sum():,}")
        st.caption('Saved-model outputs for the current product/date/rating-label selection. These predictions are separate from rating-derived sentiment. Negative scores are uncalibrated.')
        if len(selected):
            st.plotly_chart(charts.sentiment_donut(pd.DataFrame({'sentiment':selected.predicted_sentiment})),use_container_width=True,key='upload_prediction_distribution')
