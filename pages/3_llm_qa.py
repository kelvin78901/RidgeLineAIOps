"""LLM-as-Judge QA Evaluator — multi-rubric support-reply scoring."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.utils import (  # noqa: E402
    DATA_DIR, RESULTS_DIR, apply_page_chrome, empty_state, load_cached,
    plotly_layout, tag,
)

apply_page_chrome(
    "LLM-as-Judge QA Evaluator",
    breadcrumb="Module 3 · Quality assurance",
    subtitle="Every support reply is scored against multiple client brand "
             "rubrics. The same reply often scores very differently across "
             "brands — proving that one-size-fits-all QA fails for "
             "multi-client agencies.",
)

results = load_cached(RESULTS_DIR / "qa_results.json")
if not results or not results.get("evaluations"):
    st.markdown(empty_state(
        "QA results not found.",
        'python -c "from core.qa_evaluator import batch_evaluate; batch_evaluate()"'
    ), unsafe_allow_html=True)
    replies_path = DATA_DIR / "qa_replies.json"
    if replies_path.exists():
        st.markdown('<div class="rg-section-title">Replies queued for evaluation</div>',
                    unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(json.loads(replies_path.read_text())["replies"]),
                     width="stretch", hide_index=True, height=420)
    st.stop()

rows = []
for ev in results["evaluations"]:
    s = ev["scores"]
    rows.append({
        "reply_id": ev["reply_id"], "rubric": ev["rubric_id"],
        "client": ev["client_name"], "tier": ev["intended_tier"],
        "overall": s["overall"],
        "accuracy": s["accuracy"]["score"],
        "brand_voice": s["brand_voice"]["score"],
        "completeness": s["completeness"]["score"],
        "sla": s["sla"], "escalation": s["escalation"],
        "flag": bool(s["flag_for_review"]),
        "summary": s["summary"],
        "customer_message": ev["customer_message"],
        "agent_reply": ev["agent_reply"],
    })
df = pd.DataFrame(rows)

# ---------------- KPIs ----------------

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Replies", df["reply_id"].nunique())
c2.metric("Rubrics", df["rubric"].nunique())
c3.metric("Avg overall", f"{df['overall'].mean():.2f}")
c4.metric("Flagged", int(df["flag"].sum()))
c5.metric("SLA fails", int((df["sla"] == "fail").sum()))

st.divider()

# ---------------- Filters ----------------

with st.container():
    fc1, fc2, fc3, fc4 = st.columns([1, 1, 1, 2])
    rubric_pick = fc1.multiselect("Rubric", sorted(df["rubric"].unique()),
                                  default=sorted(df["rubric"].unique()))
    tier_pick = fc2.multiselect("Intended tier", sorted(df["tier"].unique()),
                                default=sorted(df["tier"].unique()))
    score_range = fc3.slider("Overall score", 0.0, 1.0, (0.0, 1.0), 0.05)
    search = fc4.text_input("Search reply / summary", "")

mask = (df["rubric"].isin(rubric_pick) & df["tier"].isin(tier_pick)
        & df["overall"].between(*score_range))
if search:
    mask &= (df["agent_reply"].str.contains(search, case=False, na=False)
             | df["summary"].str.contains(search, case=False, na=False)
             | df["customer_message"].str.contains(search, case=False, na=False))
filtered = df[mask].copy()

# ---------------- Charts ----------------

st.markdown('<div class="rg-section-title">Brand & dimension breakdown</div>',
            unsafe_allow_html=True)
cc1, cc2, cc3 = st.columns([1, 1, 1])

with cc1:
    by_brand = filtered.groupby("client", as_index=False)["overall"].mean()
    fig = px.bar(by_brand, x="client", y="overall", color="client", text="overall",
                 color_discrete_sequence=["#4f46e5", "#ea580c"],
                 title="Average score by brand")
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(**plotly_layout(height=260))
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(fig, width="stretch")

with cc2:
    fig = px.box(filtered, x="client", y="overall", color="client", points="all",
                 color_discrete_sequence=["#4f46e5", "#ea580c"],
                 title="Score distribution by brand")
    fig.update_layout(**plotly_layout(height=260))
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(fig, width="stretch")

with cc3:
    dims = filtered.groupby("rubric")[["accuracy", "brand_voice", "completeness"]].mean()
    dims_long = dims.reset_index().melt(id_vars="rubric", var_name="dim", value_name="score")
    fig = px.bar(dims_long, x="dim", y="score", color="rubric", barmode="group",
                 color_discrete_sequence=["#4f46e5", "#ea580c"],
                 title="Average dimension score (1-5)")
    fig.update_layout(**plotly_layout(height=260))
    fig.update_yaxes(range=[0, 5])
    st.plotly_chart(fig, width="stretch")

st.divider()

# ---------------- Divergence cards ----------------

st.markdown('<div class="rg-section-title">Largest brand-voice divergence</div>',
            unsafe_allow_html=True)
pivot = filtered.pivot_table(index="reply_id", columns="rubric",
                             values="overall", aggfunc="first").dropna()
if pivot.shape[1] >= 2:
    rubric_cols = list(pivot.columns)
    pivot["divergence"] = (pivot[rubric_cols[0]] - pivot[rubric_cols[1]]).abs()
    top = pivot.sort_values("divergence", ascending=False).head(3)
    for reply_id in top.index:
        rr = filtered[filtered["reply_id"] == reply_id].sort_values("rubric")
        if rr.empty:
            continue
        msg = rr.iloc[0]["customer_message"]
        reply_text = rr.iloc[0]["agent_reply"]
        tier = rr.iloc[0]["tier"]
        div = top.loc[reply_id, "divergence"]

        # Header card
        st.markdown(
            f'<div class="rg-card">'
            f'<div class="rg-spread">'
            f'<h4>{reply_id} <span class="rg-muted" style="font-weight:400">'
            f'· intended: {tier}</span></h4>'
            f'<div>{tag(f"divergence {div:.2f}", "orange")}</div></div>'
            f'<div class="rg-muted" style="margin:6px 0"><b>Customer:</b> {msg}</div>'
            f'<div style="background:#f7f8fb;border-left:3px solid #4f46e5;'
            f'padding:10px 12px;border-radius:0 8px 8px 0;margin:8px 0;'
            f'color:#0f172a;font-size:13px;line-height:1.5">'
            f'<b>Agent reply:</b> {reply_text}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

        # Per-rubric breakdown — use real Streamlit columns for clean wrapping
        sub_cols = st.columns(len(rr))
        for col, (_, r) in zip(sub_cols, rr.iterrows()):
            sla_var = "green" if r["sla"] == "pass" else "red"
            with col:
                st.markdown(
                    f'<div class="rg-card rg-card-tight">'
                    f'<div class="rg-muted" style="font-size:11px;text-transform:uppercase;'
                    f'letter-spacing:0.06em;font-weight:600">'
                    f'{r["client"]} · {r["rubric"]}</div>'
                    f'<div style="font-size:28px;font-weight:700;color:#0f172a;'
                    f'margin:4px 0">{r["overall"]:.2f}</div>'
                    f'<div class="rg-row" style="margin-top:4px">'
                    f'{tag(f"voice {r["brand_voice"]}/5", "blue")}'
                    f'{tag(f"acc {r["accuracy"]}/5", "violet")}'
                    f'{tag(f"sla:{r["sla"]}", sla_var)}'
                    f"</div>"
                    f'<div class="rg-muted" style="margin-top:8px;font-size:12px">'
                    f'{r["summary"]}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

st.divider()

# ---------------- Heatmap + detail ----------------

st.markdown('<div class="rg-section-title">Score matrix (reply × rubric)</div>',
            unsafe_allow_html=True)
heat = filtered.pivot_table(index="reply_id", columns="rubric",
                            values="overall", aggfunc="first")
if not heat.empty:
    fig = px.imshow(heat.values, x=heat.columns, y=heat.index,
                    color_continuous_scale="RdYlGn", zmin=0, zmax=1,
                    aspect="auto", text_auto=".2f")
    fig.update_layout(**plotly_layout(height=480))
    st.plotly_chart(fig, width="stretch")

st.markdown('<div class="rg-section-title">All evaluations</div>',
            unsafe_allow_html=True)
detail = filtered[
    ["reply_id", "client", "tier", "overall", "accuracy", "brand_voice",
     "completeness", "sla", "escalation", "flag", "summary"]
].sort_values(["reply_id", "client"])
st.dataframe(detail, width="stretch", hide_index=True, height=420)
