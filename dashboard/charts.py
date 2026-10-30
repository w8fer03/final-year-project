"""Interactive charts computed only from the supplied reviews or validated results."""
from __future__ import annotations

from collections.abc import Iterable
from html import escape
from textwrap import wrap

import numpy as np
import pandas as pd
import plotly.graph_objects as go

POSITIVE = "#0E9279"
NEGATIVE = "#DC6277"
ACCENT = "#D9A900"
COLORS = {"Positive": POSITIVE, "Negative": NEGATIVE}
SENTIMENTS = ("Positive", "Negative")


def _theme(fig: go.Figure, *, height: int = 355) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        font=dict(family="Inter, Arial, sans-serif", size=12, color="#68665D"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=12, r=20, t=46, b=18),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.06, x=0, title=None,
            font=dict(size=12, color="#57554D"), itemsizing="constant",
        ),
        hoverlabel=dict(
            bgcolor="white", bordercolor="#E9E3D0",
            font=dict(family="Inter, Arial, sans-serif", size=13, color="#292820"),
        ),
        hovermode=fig.layout.hovermode or "closest",
        colorway=[POSITIVE, NEGATIVE, ACCENT],
        barcornerradius=7,
    )
    fig.update_xaxes(
        showgrid=False, zeroline=False, automargin=True,
        title_font=dict(size=12, color="#7D796B"), tickfont=dict(size=11),
        ticks="", showline=False,
    )
    fig.update_yaxes(
        gridcolor="#F1EEE3", gridwidth=1, zeroline=False, automargin=True,
        title_font=dict(size=12, color="#7D796B"), tickfont=dict(size=11),
        ticks="", showline=False,
    )
    return fig


def sentiment_donut(df: pd.DataFrame) -> go.Figure:
    counts = df["sentiment"].value_counts().reindex(SENTIMENTS, fill_value=0)
    fig = go.Figure(go.Pie(
        labels=list(SENTIMENTS), values=counts.values,
        hole=.76, sort=False, direction="clockwise",
        marker=dict(colors=[POSITIVE, NEGATIVE], line=dict(color="white", width=4)),
        textinfo="percent", textposition="outside", textfont=dict(size=13),
        hovertemplate="%{label}<br>%{value:,} reviews · %{percent}<extra></extra>",
    ))
    fig.add_annotation(x=.5, y=.5, text=f"<b>{int(counts.sum()):,}</b><br><span style='font-size:12px;color:#7D796B'>reviews</span>", showarrow=False, font=dict(size=28, color="#292820"))
    return _theme(fig)


def rating_bars(df: pd.DataFrame) -> go.Figure:
    ratings = [1, 2, 4, 5]
    counts = df["rating"].value_counts().reindex(ratings, fill_value=0)
    fig = go.Figure(go.Bar(
        x=[f"{r} ★" for r in ratings], y=counts.values,
        marker_color=[NEGATIVE, NEGATIVE, POSITIVE, POSITIVE],
        text=counts.values, texttemplate="%{text:,}", textposition="outside",
        hovertemplate="%{x}<br>%{y:,} reviews<extra></extra>",
    ))
    fig.update_layout(showlegend=False, bargap=.45, yaxis_title="Number of reviews", xaxis_title="Original star rating")
    return _theme(fig)


def trend_chart(df: pd.DataFrame, frequency: str = "MS", percentage: bool = False) -> go.Figure:
    """Aggregate dated reviews, filling zero-count periods without invented dates.

    The bounds are the first/last observed review periods. Percentage denominators
    are all eligible reviews in each period; periods with no reviews remain gaps.
    """
    dated = df.loc[df["review_date"].notna() & df["sentiment"].isin(SENTIMENTS), ["review_date", "sentiment"]]
    fig = go.Figure()
    if dated.empty:
        return _theme(fig)
    counts = (
        dated.assign(count=1)
        .groupby([pd.Grouper(key="review_date", freq=frequency), "sentiment"], observed=True)["count"]
        .sum().unstack(fill_value=0).reindex(columns=SENTIMENTS, fill_value=0)
    )
    complete_dates = pd.date_range(counts.index.min(), counts.index.max(), freq=frequency)
    counts = counts.reindex(complete_dates, fill_value=0)
    values = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0) * 100 if percentage else counts
    for sentiment in SENTIMENTS:
        fig.add_trace(go.Scatter(
            x=values.index, y=values[sentiment], name=sentiment,
            mode="lines+markers", line=dict(color=COLORS[sentiment], width=3),
            marker=dict(size=6, line=dict(color="white", width=1.5)), connectgaps=False,
            customdata=counts[sentiment].to_numpy(),
            hovertemplate=("%{x|%d %b %Y}<br>" + sentiment +
                           (": %{y:.1f}%<br>%{customdata:,} reviews" if percentage else ": %{y:,} reviews") + "<extra></extra>"),
        ))
    fig.update_layout(
        hovermode="x unified", xaxis_title="Review period",
        yaxis_title="Share of reviews (%)" if percentage else "Number of reviews",
    )
    if percentage:
        fig.update_yaxes(range=[0, 100], ticksuffix="%")
    else:
        fig.update_yaxes(rangemode="tozero")
    return _theme(fig, height=365)


def product_comparison(summary: pd.DataFrame) -> go.Figure:
    """Compare named product groups with review counts and sentiment shares.

    ``summary`` is already aggregated by product_name by the data layer.
    """
    fig = go.Figure()
    names = summary["product_name"].astype(str).tolist()
    for sentiment in SENTIMENTS:
        prefix = sentiment.lower()
        custom = np.array(list(zip(
            [escape(name) for name in names],
            summary[f"{prefix}_percentage"],
            summary["total_reviews"],
        )), dtype=object) if len(summary) else np.empty((0, 3))
        fig.add_trace(go.Bar(
            x=summary[f"{prefix}_reviews"], y=names, orientation="h",
            name=sentiment, marker_color=COLORS[sentiment], customdata=custom,
            hovertemplate=("%{customdata[0]}<br>" + sentiment +
                           ": %{x:,} reviews (%{customdata[1]:.1f}%)"
                           "<br>Product total: %{customdata[2]:,}"
                           "<extra></extra>"),
        ))
    fig.update_layout(
        barmode="stack", bargap=.42,
        xaxis_title="Number of reviews", yaxis_title="Product",
    )
    _theme(fig, height=max(380, len(summary) * 70 + 75))
    fig.update_xaxes(showgrid=True, gridcolor="#F1EEE3", rangemode="tozero")
    fig.update_yaxes(
        type="category", showgrid=False, autorange="reversed",
        categoryorder="array", categoryarray=names,
        tickmode="array", tickvals=names,
        ticktext=["<br>".join(escape(line) for line in wrap(name, width=29)) for name in names],
    )
    return fig


def confusion_heatmap(matrix: list[list[int]]) -> go.Figure:
    names = [["True negative", "False positive"], ["False negative", "True positive"]]
    fig = go.Figure(go.Heatmap(
        z=matrix, x=["Negative (0)", "Positive (1)"], y=["Negative (0)", "Positive (1)"],
        colorscale=[[0, "#FFFBE5"], [0.5, "#F0CD4E"], [1, "#76600D"]], customdata=names,
        text=matrix, texttemplate="%{text:,}", textfont=dict(size=23),
        hovertemplate="%{customdata}<br>Actual: %{y}<br>Predicted: %{x}<br>%{z:,} reviews<extra></extra>",
        colorbar=dict(title="Count", thickness=12), xgap=5, ygap=5,
    ))
    fig.update_layout(xaxis_title="Predicted Sentiment", yaxis_title="Actual Sentiment")
    fig.update_yaxes(autorange="reversed")
    return _theme(fig, height=365)


def metric_comparison(evaluations: Iterable, metric: str = "f1") -> go.Figure:
    rows = list(evaluations)
    if metric not in {"accuracy", "precision", "recall", "f1"}:
        raise ValueError("Metric must be accuracy, precision, recall, or f1.")
    title = "F1-score" if metric == "f1" else metric.title()
    scores = [row.metrics[metric] for row in rows]
    fig = go.Figure(go.Bar(
        x=[row.model_name for row in rows], y=scores, marker_color=ACCENT,
        text=scores, texttemplate="%{text:.3f}", textposition="outside",
        hovertemplate="%{x}<br>" + title + ": %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(yaxis_title=title, xaxis_title="Model", showlegend=False, bargap=.5)
    fig.update_yaxes(range=[0, 1.08], tickformat=".0%")
    return _theme(fig)
