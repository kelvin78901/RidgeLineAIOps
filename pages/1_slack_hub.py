"""Slack Intelligence Hub — channel health dashboard."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.utils import (  # noqa: E402
    DATA_DIR, RESULTS_DIR, apply_page_chrome, empty_state, health_pill,
    load_cached, make_trend, plotly_layout, status_tag, tag,
)

apply_page_chrome(
    "Slack Intelligence Hub",
    breadcrumb="Module 1 · Channel health",
    subtitle="Real Reddit messages (Google GoEmotions) partitioned into 50 "
             "simulated client channels. Each channel scored by an LLM on "
             "satisfaction, urgency, churn signal and tone trajectory.",
)

manifest = load_cached(DATA_DIR / "channel_manifest.json")
results = load_cached(RESULTS_DIR / "channel_health.json")

if not manifest:
    st.markdown(empty_state(
        "Channel manifest missing.",
        'python -c "from core.slack_analyzer import build_manifest; build_manifest()"'
    ), unsafe_allow_html=True)
    st.stop()

if not results or not results.get("channels"):
    st.markdown(empty_state(
        "LLM scoring not run yet.",
        'python -c "from core.slack_analyzer import batch_analyze; batch_analyze()"'
    ), unsafe_allow_html=True)
    st.stop()

df = pd.DataFrame(results["channels"])

# ---------------- KPI strip ----------------

red = int((df["health"] == "red").sum())
yellow = int((df["health"] == "yellow").sum())
green = int((df["health"] == "green").sum())
likely = int((df["churn_signal"] == "likely").sum())

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Red", red)
c2.metric("Amber", yellow)
c3.metric("Green", green)
c4.metric("Likely-churn signal", likely)
c5.metric("Avg satisfaction", f"{df['satisfaction'].mean():.2f} / 5")

st.divider()

# ---------------- Filters ----------------

with st.container():
    fc1, fc2, fc3, fc4 = st.columns([1, 1, 1, 2])
    health_pick = fc1.multiselect("Health", ["red", "yellow", "green"],
                                  default=["red", "yellow", "green"])
    churn_pick = fc2.multiselect("Churn signal",
                                 ["likely", "possible", "none"],
                                 default=["likely", "possible", "none"])
    sat_min = fc3.slider("Min satisfaction", 1, 5, 1)
    search = fc4.text_input("Search channel / summary", "")

mask = df["health"].isin(health_pick) & df["churn_signal"].isin(churn_pick) & (df["satisfaction"] >= sat_min)
if search:
    mask &= (df["channel_id"].str.contains(search, case=False, na=False) |
             df["summary"].str.contains(search, case=False, na=False))
filtered = df[mask].copy()

# ---------------- Charts ----------------

st.markdown('<div class="rg-section-title">Distribution</div>',
            unsafe_allow_html=True)

chart_c1, chart_c2, chart_c3 = st.columns([1, 1, 1])

with chart_c1:
    counts = (filtered["health"].value_counts()
              .reindex(["green", "yellow", "red"]).fillna(0).reset_index())
    counts.columns = ["health", "count"]
    fig = px.bar(
        counts, x="health", y="count", color="health", text="count",
        color_discrete_map={"green": "#059669", "yellow": "#ea580c", "red": "#dc2626"},
        category_orders={"health": ["green", "yellow", "red"]},
        title="Health bands",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(**plotly_layout(height=260))
    st.plotly_chart(fig, width="stretch")

with chart_c2:
    fig = px.scatter(
        filtered, x="satisfaction", y="urgency",
        color="health", size="unanswered_count", hover_data=["channel_id", "client_name"],
        color_discrete_map={"green": "#059669", "yellow": "#ea580c", "red": "#dc2626"},
        title="Satisfaction vs urgency (bubble = unanswered)",
    )
    fig.update_layout(**plotly_layout(height=260, showlegend=True))
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                  xanchor="right", x=1))
    st.plotly_chart(fig, width="stretch")

with chart_c3:
    trend = make_trend("slack-red-page", base=red or 4, drift=0.05, noise=2.0)
    tdf = pd.DataFrame(trend)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tdf["date"], y=tdf["value"], mode="lines+markers",
        line=dict(color="#dc2626", width=2),
        marker=dict(size=5, color="#dc2626"),
        fill="tozeroy", fillcolor="rgba(220,38,38,0.13)",
    ))
    fig.update_layout(**plotly_layout(height=260),
                      title="Red channel count · last 14d")
    st.plotly_chart(fig, width="stretch")

st.divider()

# ---------------- Red channel cards ----------------

red_df = filtered[filtered["health"] == "red"].sort_values("satisfaction")
if not red_df.empty:
    st.markdown('<div class="rg-section-title">Needs attention now</div>',
                unsafe_allow_html=True)
    for _, row in red_df.iterrows():
        churn_var = ("red" if row["churn_signal"] == "likely"
                     else "amber" if row["churn_signal"] == "possible" else "slate")
        tone_var = {"deteriorating": "red", "stable": "slate",
                    "improving": "green"}.get(row["tone_trajectory"], "slate")
        gt = row.get("ground_truth_sentiment", 0.0)
        sat = row["satisfaction"]
        urg = row["urgency"]
        churn_signal = row["churn_signal"]
        tone = row["tone_trajectory"]
        channel_id = row["channel_id"]
        client_name = row["client_name"]
        summary = row["summary"]
        tags_html = (
            health_pill(row["health"])
            + tag(f"sat {sat}/5", "blue")
            + tag(f"urgency {urg}/5", "orange")
            + tag(f"churn: {churn_signal}", churn_var)
            + tag(f"tone: {tone}", tone_var)
            + tag(f"GT sentiment {gt:+.2f}", "slate")
        )
        st.markdown(
            f'<div class="rg-card">'
            f'<h4>{channel_id} &nbsp; <span class="rg-muted" style="font-weight:400">'
            f'· {client_name}</span></h4>'
            f'<div class="rg-row" style="margin-bottom:8px">{tags_html}</div>'
            f'<div class="rg-muted">{summary}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

st.divider()

# ---------------- Detail table ----------------

st.markdown('<div class="rg-section-title">All channels</div>', unsafe_allow_html=True)
display = filtered[
    ["channel_id", "client_name", "health", "satisfaction", "urgency",
     "churn_signal", "tone_trajectory", "unanswered_count",
     "ground_truth_sentiment", "summary"]
].rename(columns={
    "channel_id": "Channel", "client_name": "Client", "health": "Health",
    "satisfaction": "Sat", "urgency": "Urg", "churn_signal": "Churn",
    "tone_trajectory": "Tone", "unanswered_count": "Unanswered",
    "ground_truth_sentiment": "GT sentiment", "summary": "Summary",
})
st.dataframe(display.sort_values("Health"), width="stretch",
             hide_index=True, height=480)
