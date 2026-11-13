"""Existing Model Performance section extended with recorded Version B exports."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from dashboard.runtime import refresh_charts
from dashboard.final_evaluations import METRICS, FinalEvaluationReport
from dashboard.version_b import ARCHITECTURES


def render_final_performance(report: FinalEvaluationReport):
    charts = refresh_charts()
    st.success('Eight recorded held-out configurations · Version B CLDR experiments')
    st.caption('Primary thesis measures: macro F1 and minority-class negative F1. These are the supplied final exports; no held-out inference, metric re-evaluation, tuning, or model selection is performed here.')
    columns = {'architecture':'Architecture','split':'Test split','method':'Training method','threshold':'Negative threshold','test_count':'Held-out reviews', **METRICS}
    for split, label in [('80_20','80:20'),('70_30','70:30')]:
        with st.container(border=True, key=f'panel_final_table_{split}'):
            st.markdown(f'#### {label} recorded results')
            group = report.group(split)
            public = group[list(columns)].copy()
            public['architecture'] = public.architecture.map(ARCHITECTURES)
            public['split'] = label
            st.dataframe(public.rename(columns=columns), hide_index=True, use_container_width=True,
                         column_config={name:st.column_config.NumberColumn(format='%.5f') for name in METRICS.values()})
    st.caption('The two splits contain different test reviews and are separate comparison groups. Within-split support counts are checked; per-review held-out membership lists were not supplied and are not inferred.')
    left, right = st.columns(2)
    split = left.selectbox('Evaluation test split', ['80_20','70_30'], format_func=lambda value:value.replace('_',':'), key='final_eval_split')
    group = report.group(split)
    architecture = right.selectbox('Evaluation architecture', list(ARCHITECTURES), format_func=ARCHITECTURES.get, key='final_eval_architecture')
    row = group[group.architecture.eq(architecture)].iloc[0]
    st.write(f"**{ARCHITECTURES[architecture]} · {split.replace('_',':')}** · {int(row.test_count):,} held-out reviews · {int(row.actual_negative):,} actual negative reviews")
    st.caption(f"Recorded training method: {row.method} · Decision threshold on negative-class score: {row.threshold!r}")
    entries = [('Macro F1 · primary','macro_f1'),('Negative F1 · minority','negative_f1'),('Positive F1','positive_f1'),('Weighted F1','weighted_f1'),
               ('Accuracy','accuracy'),('Negative precision','negative_precision'),('Negative recall','negative_recall'),('Negative PR-AUC','negative_PR_AUC')]
    for start in (0,4):
        for column, (label,key) in zip(st.columns(4),entries[start:start+4]):
            column.metric(label,f'{row[key]:.5f}')
    left, right = st.columns(2)
    with left:
        st.markdown('#### Within-split model comparison')
        metric = st.selectbox('Recorded comparison metric',list(METRICS),format_func=METRICS.get,key='final_eval_metric')
        ordered = group.set_index('architecture').loc[list(ARCHITECTURES)]
        names = ['Frozen hybrid','TF-IDF + XGBoost','Fine-tuned DistilBERT','Fine-tuned hybrid']
        figure = go.Figure(go.Bar(x=names,y=ordered[metric],marker_color='#D9A900',customdata=[ARCHITECTURES[key] for key in ordered.index],
                                 hovertemplate='%{customdata}<br>'+METRICS[metric]+': %{y:.5f}<extra></extra>'))
        figure.update_layout(height=355,template='plotly_white',font=dict(family='Poppins, sans-serif',size=13),
                             paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',margin=dict(l=10,r=10,t=20,b=50),
                             yaxis_title=METRICS[metric],yaxis_range=[0,1.05])
        st.plotly_chart(figure,use_container_width=True,key='final_comparison',config={'displaylogo':False})
    with right:
        st.markdown('#### Recorded confusion matrix')
        st.plotly_chart(charts.confusion_heatmap(row.confusion_matrix),use_container_width=True,key='final_confusion',config={'displaylogo':False})
        st.caption('Rows: actual Negative / Positive. Columns: predicted Negative / Positive. Counts are taken directly from the confusion CSV.')
    with st.expander('Read these final results'):
        st.write('Macro F1 averages the two class F1 scores equally. Negative F1 focuses on the severely underrepresented class. Weighted F1 and accuracy can remain high despite minority-class errors. Negative PR-AUC is imported as recorded; it cannot be recovered from a confusion matrix.')
        st.caption('The deployment default is recorded in deployment_config.json from training-only CV. No test-score winner is selected automatically. Prediction scores are uncalibrated and are not confidence percentages.')
    left, right = st.columns(2)
    left.download_button('Download recorded test metrics',report.metrics_bytes,'day3_all_models_test_metrics.csv','text/csv',key='final_metrics_download')
    right.download_button('Download recorded confusion counts',report.confusion_bytes,'day3_all_models_confusion.csv','text/csv',key='final_confusion_download')
