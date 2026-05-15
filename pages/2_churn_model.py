"""Churn Risk Model — XGBoost + SHAP explainability."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import shap
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.utils import (  # noqa: E402
    MODEL_DIR, RESULTS_DIR, apply_page_chrome, empty_state, health_pill,
    load_cached, make_trend, plotly_layout, tag,
)

apply_page_chrome(
    "Churn Risk Model",
    breadcrumb="Module 2 · Predictive analytics",
    subtitle="XGBoost trained on 7,043 IBM Telco customers, augmented with "
             "four Ridgeline signals — Slack sentiment, QA trend, SLA "
             "breaches, CSAT change. SHAP explains per-client risk drivers.",
)


@st.cache_data
def _load():
    summary = load_cached(RESULTS_DIR / "churn_predictions.json")
    shap_records = load_cached(MODEL_DIR / "shap_values.json")
    p = MODEL_DIR / "test_set.parquet"
    ts = pd.read_parquet(p) if p.exists() else None
    return summary, shap_records, ts


summary, shap_records, test_set = _load()
if summary is None or test_set is None or shap_records is None:
    st.markdown(empty_state(
        "Model artifacts not found.",
        'python -c "from core.churn_trainer import train; train()"'
    ), unsafe_allow_html=True)
    st.stop()

risk_scores = test_set["_risk_score"].to_numpy()
high_risk_n = int((risk_scores >= 70).sum())
med_risk_n = int(((risk_scores >= 40) & (risk_scores < 70)).sum())
low_risk_n = int((risk_scores < 40).sum())

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Model AUC", f"{summary['auc']:.3f}")
c2.metric("Customers scored", f"{summary['n_test']:,}")
c3.metric("High risk", high_risk_n, f"{high_risk_n / summary['n_test']:.0%}")
c4.metric("Medium risk", med_risk_n, f"{med_risk_n / summary['n_test']:.0%}")
c5.metric("Average score", f"{risk_scores.mean():.1f}")

st.divider()

tab_overview, tab_drilldown, tab_drivers, tab_table = st.tabs(
    ["Overview", "Client drill-down", "Global drivers", "Top-N table"]
)

# ---------------- Overview tab ----------------

with tab_overview:
    o1, o2 = st.columns([2, 1])

    with o1:
        st.markdown('<div class="rg-section-title">Risk score distribution</div>',
                    unsafe_allow_html=True)
        bins = np.arange(0, 105, 5)
        hist, edges = np.histogram(risk_scores, bins=bins)
        df_h = pd.DataFrame({"bin": [f"{int(e)}-{int(e+5)}" for e in edges[:-1]],
                             "count": hist})
        colors = ["#059669"] * 8 + ["#ea580c"] * 6 + ["#dc2626"] * 6
        fig = px.bar(df_h, x="bin", y="count")
        fig.update_traces(marker_color=colors)
        fig.update_layout(**plotly_layout(height=300))
        fig.update_xaxes(title="Risk score (0-100)")
        fig.update_yaxes(title="Customers")
        st.plotly_chart(fig, width="stretch")

    with o2:
        st.markdown('<div class="rg-section-title">Volume trend · last 14d</div>',
                    unsafe_allow_html=True)
        trend = make_trend("churn-vol", base=high_risk_n, drift=0.5, noise=8.0)
        tdf = pd.DataFrame(trend)
        fig = px.area(tdf, x="date", y="value")
        fig.update_traces(line_color="#dc2626", fillcolor="rgba(220,38,38,0.13)")
        fig.update_layout(**plotly_layout(height=300))
        fig.update_yaxes(title="High-risk count")
        fig.update_xaxes(title="")
        st.plotly_chart(fig, width="stretch")

    st.markdown('<div class="rg-section-title">Population segmentation</div>',
                unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    s1.markdown(f'<div class="rg-card"><h4>High risk</h4>'
                f'<div style="font-size:30px;font-weight:700;color:#dc2626">{high_risk_n}</div>'
                f'<div class="rg-muted">score ≥ 70</div></div>',
                unsafe_allow_html=True)
    s2.markdown(f'<div class="rg-card"><h4>Medium risk</h4>'
                f'<div style="font-size:30px;font-weight:700;color:#ea580c">{med_risk_n}</div>'
                f'<div class="rg-muted">40 ≤ score &lt; 70</div></div>',
                unsafe_allow_html=True)
    s3.markdown(f'<div class="rg-card"><h4>Low risk</h4>'
                f'<div style="font-size:30px;font-weight:700;color:#059669">{low_risk_n}</div>'
                f'<div class="rg-muted">score &lt; 40</div></div>',
                unsafe_allow_html=True)

# ---------------- Drill-down tab ----------------

with tab_drilldown:
    client_ids = test_set["_client_id"].tolist()
    sorted_ids = test_set.sort_values("_risk_score", ascending=False)["_client_id"].tolist()

    d1, d2 = st.columns([1, 1])
    with d1:
        sort_mode = st.radio("Sort", ["Highest risk first", "Alphabetical"],
                             horizontal=True)
        listing = sorted_ids if sort_mode == "Highest risk first" else client_ids
        chosen = st.selectbox("Client", listing)
    with d2:
        threshold = st.slider("High-risk threshold", 50, 95, 70,
                              help="Affects the band colour but not the score.")

    row = test_set[test_set["_client_id"] == chosen].iloc[0]
    risk = float(row["_risk_score"])
    actually_churned = bool(row["_churn_actual"])
    band = "red" if risk >= threshold else "amber" if risk >= 40 else "green"

    score_color = {"red": "#dc2626", "amber": "#ea580c", "green": "#059669"}[band]
    st.markdown(
        f'<div class="rg-card">'
        f'<div class="rg-row" style="justify-content:space-between">'
        f'<div>'
        f'<h4 style="margin-bottom:6px">{chosen}</h4>'
        f'<div class="rg-row">'
        f'{health_pill(band)}'
        f'{tag("CHURNED" if actually_churned else "ACTIVE", "red" if actually_churned else "green")}'
        f"</div></div>"
        f'<div style="text-align:right;min-width:120px">'
        f'<div style="font-size:42px;font-weight:700;color:{score_color}">{risk:.0f}'
        f'<span style="font-size:16px;color:#64748b">/100</span></div>'
        f'<div class="rg-muted">risk score</div>'
        f"</div></div></div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="rg-section-title">Why this client is at risk</div>',
                unsafe_allow_html=True)
    rec = shap_records[chosen]
    features = rec["feature_names"]
    shap_vals = np.array(rec["values"])
    feature_vals = np.array(rec["feature_values"])
    base_value = rec["base_value"]

    drivers = pd.DataFrame({
        "feature": features, "value": feature_vals, "shap": shap_vals,
    })
    drivers["abs"] = drivers["shap"].abs()
    drivers = drivers.sort_values("abs", ascending=False).head(12)
    drivers["impact"] = np.where(drivers["shap"] > 0, "increases risk", "lowers risk")
    drivers = drivers.drop(columns=["abs"])

    bar_fig = px.bar(
        drivers.iloc[::-1], x="shap", y="feature", orientation="h",
        color="impact",
        color_discrete_map={"increases risk": "#dc2626", "lowers risk": "#059669"},
        text=drivers.iloc[::-1]["shap"].round(3),
        title="Top SHAP contributors (positive = pushes toward churn)",
    )
    bar_fig.update_layout(**plotly_layout(height=420, showlegend=True))
    bar_fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                      xanchor="right", x=1))
    bar_fig.update_xaxes(title="SHAP contribution")
    bar_fig.update_yaxes(title="")
    st.plotly_chart(bar_fig, width="stretch")

    st.markdown('<div class="rg-section-title">SHAP waterfall</div>',
                unsafe_allow_html=True)
    explanation = shap.Explanation(
        values=shap_vals, base_values=base_value,
        data=feature_vals, feature_names=features,
    )
    fig, _ = plt.subplots(figsize=(10, 5))
    shap.plots.waterfall(explanation, max_display=10, show=False)
    fig.patch.set_facecolor("#ffffff")
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

# ---------------- Drivers tab ----------------

with tab_drivers:
    st.markdown('<div class="rg-section-title">Global feature importance (SHAP summary)</div>',
                unsafe_allow_html=True)
    img = RESULTS_DIR / "shap_summary.png"
    if img.exists():
        st.image(str(img), width="stretch")
    else:
        st.info("Re-run training to generate the summary plot.")

# ---------------- Top-N table ----------------

with tab_table:
    n = st.slider("Show top N at-risk customers", 5, 50, 20)
    only_churn = st.checkbox("Show only customers that actually churned", value=False)
    top = pd.DataFrame(summary["top_50_at_risk"][:n])
    if only_churn:
        top = top[top["actually_churned"]]
    drivers_short = top["top_drivers"].apply(
        lambda lst: ", ".join(d["feature"] for d in lst[:3])
    )
    view = pd.DataFrame({
        "Rank": top["rank"],
        "Client": top["client_id"],
        "Risk score": top["risk_score"],
        "Churned (GT)": top["actually_churned"],
        "Top drivers": drivers_short,
    })
    st.dataframe(view, width="stretch", hide_index=True, height=520)
