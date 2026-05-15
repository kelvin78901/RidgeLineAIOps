"""Ridgeline AI Ops — unified overview dashboard."""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from core.utils import (  # noqa: E402
    MODEL_DIR, RESULTS_DIR, active_model_id, active_provider,
    apply_page_chrome, health_pill, load_cached, make_trend,
    plotly_layout, status_tag, tag,
)

apply_page_chrome(
    "Operations Overview",
    breadcrumb="Ridgeline AI Ops · Command Center",
    subtitle="Unified client-health view across Slack signals, churn risk, "
             "QA scores and code security. Demo data drawn from open-source "
             "sources — Telco Churn, GoEmotions, intentionally vulnerable Python.",
)

# Provider badge
try:
    _provider, _model = active_provider(), active_model_id()
    st.markdown(
        f'<div class="rg-row" style="margin:8px 0 16px 0">'
        f'{tag("LIVE", "green")}'
        f'{tag(f"Provider · {_provider}", "blue")}'
        f'{tag(f"Model · {_model}", "violet")}'
        f'{tag(f"Last sync · {date.today().isoformat()}", "slate")}'
        f"</div>",
        unsafe_allow_html=True,
    )
except Exception as e:  # noqa: BLE001
    st.warning(f"LLM provider misconfigured: {e}")

# Caches
churn = load_cached(RESULTS_DIR / "churn_predictions.json")
qa = load_cached(RESULTS_DIR / "qa_results.json")
slack = load_cached(RESULTS_DIR / "channel_health.json")
security = load_cached(RESULTS_DIR / "security_report.json")

# ---------------- KPIs ----------------

slack_channels = (slack or {}).get("channels", [])
slack_red = sum(1 for c in slack_channels if c.get("health") == "red")
slack_amber = sum(1 for c in slack_channels if c.get("health") == "yellow")
slack_total = len(slack_channels)

churn_avg = churn_high = churn_n = None
if churn:
    test_path = MODEL_DIR / "test_set.parquet"
    if test_path.exists():
        ts = pd.read_parquet(test_path)
        churn_avg = float(ts["_risk_score"].mean())
        churn_high = int((ts["_risk_score"] >= 70).sum())
        churn_n = len(ts)

qa_avg = qa_flagged = qa_n = None
if qa and qa.get("evaluations"):
    overalls = [e["scores"]["overall"] for e in qa["evaluations"] if "scores" in e]
    qa_avg = sum(overalls) / len(overalls) if overalls else None
    qa_flagged = sum(1 for e in qa["evaluations"] if e["scores"].get("flag_for_review"))
    qa_n = len(qa["evaluations"])

sec_semgrep = sec_llm = sec_high = None
if security:
    t = security["totals"]
    sec_semgrep = t["semgrep"]
    sec_llm = t.get("llm", t.get("claude", 0))
    sec_high = t["semgrep_high"] + t.get("llm_high", t.get("claude_high", 0))

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Slack — Red channels",
    f"{slack_red}" if slack else "—",
    f"{slack_amber} amber · {slack_total} total" if slack else "no data",
)
c2.metric(
    "Churn — High risk clients",
    f"{churn_high}" if churn_high is not None else "—",
    f"avg score {churn_avg:.1f}/100" if churn_avg is not None else "no data",
)
c3.metric(
    "QA — Avg score",
    f"{qa_avg:.2f}" if qa_avg is not None else "—",
    f"{qa_flagged} flagged · {qa_n} evals" if qa_avg is not None else "no data",
)
c4.metric(
    "Security — Findings",
    f"{(sec_semgrep or 0) + (sec_llm or 0)}" if security else "—",
    f"{sec_high} high severity" if sec_high is not None else "no data",
)

st.divider()

# ---------------- Trend strip ----------------

st.markdown('<div class="rg-section-title">14-day trend (simulated rollup)</div>',
            unsafe_allow_html=True)

trend_red = make_trend("slack-red", base=slack_red or 4, drift=0.05, noise=2.0)
trend_churn = make_trend("churn-high", base=churn_high or 200, drift=0.4, noise=8.0)
trend_qa = make_trend("qa-avg", base=(qa_avg or 0.55) * 100, drift=-0.1, noise=2.5)
trend_sec = make_trend("sec-high", base=sec_high or 6, drift=-0.02, noise=1.0)

trend_dfs = [
    ("Slack red channels", trend_red, "#dc2626", "rgba(220,38,38,0.10)"),
    ("High-risk clients", trend_churn, "#4f46e5", "rgba(79,70,229,0.10)"),
    ("QA score (×100)", trend_qa, "#059669", "rgba(5,150,105,0.10)"),
    ("Security high-sev", trend_sec, "#ea580c", "rgba(234,88,12,0.10)"),
]

cols = st.columns(4)
for (title, series, color, fill), col in zip(trend_dfs, cols):
    df_t = pd.DataFrame(series)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_t["date"], y=df_t["value"], mode="lines",
        line=dict(color=color, width=2),
        fill="tozeroy", fillcolor=fill,
    ))
    fig.update_layout(**plotly_layout(height=140))
    fig.update_xaxes(showticklabels=False)
    col.markdown(f'<div class="rg-muted" style="font-size:12px;margin-bottom:-6px">'
                 f'{title}</div>', unsafe_allow_html=True)
    col.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

st.divider()

# ---------------- Filters + client board ----------------

left, right = st.columns([1, 3], gap="large")

with left:
    st.markdown('<div class="rg-section-title">Filters</div>', unsafe_allow_html=True)
    sources = st.multiselect(
        "Signal source",
        ["churn_model", "slack_hub", "qa_evaluator"],
        default=["churn_model", "slack_hub"],
    )
    band_filter = st.multiselect(
        "Risk band", ["red", "yellow", "green"], default=["red", "yellow"]
    )
    min_score = st.slider("Minimum score (0-100)", 0, 100, 40)
    search = st.text_input("Search client / channel", "")

# Build unified client list
rows = []
if "churn_model" in sources and churn:
    for c in churn.get("top_50_at_risk", []):
        rows.append({
            "client": c["client_id"], "source": "churn_model",
            "score": c["risk_score"],
            "band": "red" if c["risk_score"] >= 70 else "yellow" if c["risk_score"] >= 40 else "green",
            "note": f"Top driver: {c['top_drivers'][0]['feature']}" if c.get("top_drivers") else "",
            "extra_tag": ("CHURNED" if c.get("actually_churned") else "ACTIVE"),
            "extra_variant": "red" if c.get("actually_churned") else "blue",
        })
if "slack_hub" in sources and slack:
    for c in slack.get("channels", []):
        s = c.get("satisfaction", 3)
        score = max(0.0, min(100.0, (5 - s) * 20 + c.get("urgency", 3) * 4))
        rows.append({
            "client": c["channel_id"], "source": "slack_hub",
            "score": score, "band": c["health"],
            "note": c.get("summary", ""),
            "extra_tag": f"churn:{c.get('churn_signal','-')}",
            "extra_variant": "red" if c.get("churn_signal") == "likely" else "amber" if c.get("churn_signal") == "possible" else "slate",
        })
if "qa_evaluator" in sources and qa:
    for e in qa.get("evaluations", []):
        s = e["scores"]
        sc = (1 - s["overall"]) * 100
        rows.append({
            "client": e["client_name"], "source": "qa_evaluator",
            "score": sc,
            "band": "red" if sc >= 60 else "yellow" if sc >= 30 else "green",
            "note": s["summary"], "extra_tag": e["rubric_id"],
            "extra_variant": "violet",
        })

df = pd.DataFrame(rows)
if not df.empty:
    df = df[df["band"].isin(band_filter)]
    df = df[df["score"] >= min_score]
    if search:
        df = df[df["client"].str.contains(search, case=False, na=False) |
                df["note"].str.contains(search, case=False, na=False)]

with right:
    st.markdown('<div class="rg-section-title">Client / channel watchlist</div>',
                unsafe_allow_html=True)
    if df.empty:
        st.info("No items match the current filters.")
    else:
        df_view = df.sort_values("score", ascending=False).head(20)
        for _, r in df_view.iterrows():
            st.markdown(
                f'<div class="rg-card" style="padding:12px 16px">'
                f'<div class="rg-row" style="justify-content:space-between">'
                f'<div style="flex:1">'
                f'<b>{r["client"]}</b> &nbsp; '
                f'{tag(r["source"], "blue")} '
                f'{health_pill(r["band"])} '
                f'{tag(str(r["extra_tag"]), r["extra_variant"])}'
                f'<div class="rg-muted" style="margin-top:4px">{r["note"]}</div>'
                f"</div>"
                f'<div style="text-align:right;min-width:60px">'
                f'<div style="font-size:22px;font-weight:700;color:#0f172a">{r["score"]:.0f}</div>'
                f'<div class="rg-muted" style="font-size:11px">score</div>'
                f"</div></div></div>",
                unsafe_allow_html=True,
            )

st.divider()

# ---------------- Bottom: source breakdown + readiness ----------------

cb1, cb2 = st.columns([2, 1], gap="large")

with cb1:
    st.markdown('<div class="rg-section-title">Signals by source</div>',
                unsafe_allow_html=True)
    if not df.empty:
        bd = (df.groupby(["source", "band"]).size()
                .reset_index(name="count"))
        fig = px.bar(
            bd, x="source", y="count", color="band",
            color_discrete_map={"red": "#dc2626", "yellow": "#ea580c", "green": "#059669"},
            category_orders={"band": ["red", "yellow", "green"]},
            barmode="stack",
        )
        fig.update_layout(**plotly_layout(height=300, showlegend=True))
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom",
                                      y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Adjust filters to populate this chart.")

with cb2:
    st.markdown('<div class="rg-section-title">Module readiness</div>',
                unsafe_allow_html=True)
    agent_scenarios = load_cached(RESULTS_DIR / "agent_scenarios.json")
    modules = [
        ("Slack Hub", slack, "core.slack_analyzer.batch_analyze"),
        ("Churn Model", churn, "core.churn_trainer.train"),
        ("LLM QA", qa, "core.qa_evaluator.batch_evaluate"),
        ("Security", security, "core.security_scanner.scan_all"),
        ("AI Agent", agent_scenarios, "core.agent_simulator.precompute_scenarios"),
    ]
    for name, payload, fn in modules:
        ok = bool(payload)
        st.markdown(
            f'<div class="rg-card" style="padding:10px 14px">'
            f'<div class="rg-row" style="justify-content:space-between">'
            f'<div><b>{name}</b><div class="rg-muted" style="font-size:11px">'
            f'<code>{fn}()</code></div></div>'
            f'<div>{tag("READY", "green") if ok else tag("PENDING", "amber")}</div>'
            f"</div></div>",
            unsafe_allow_html=True,
        )

st.divider()
st.markdown(
    '<div class="rg-muted" style="font-size:12px;text-align:center">'
    "Built with Streamlit · XGBoost + SHAP · Anthropic Claude / DeepSeek · Semgrep. "
    "All data is open-source or synthetic — no Ridgeline customer data is used."
    "</div>",
    unsafe_allow_html=True,
)
