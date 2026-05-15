"""Security Scanner — two-layer findings (Semgrep + LLM)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.utils import (  # noqa: E402
    RESULTS_DIR, apply_page_chrome, empty_state, load_cached,
    plotly_layout, status_tag, tag,
)

apply_page_chrome(
    "Code Security Scanner",
    breadcrumb="Module 4 · AppSec",
    subtitle="Two-layer pipeline on typical \"vibe-coded\" Python. Semgrep "
             "finds known patterns (injection, secrets, debug mode); the "
             "LLM finds business-logic and auth issues static analysis misses.",
)

report = load_cached(RESULTS_DIR / "security_report.json")
if not report:
    st.markdown(empty_state(
        "Security scan not run yet.",
        'python -c "from core.security_scanner import scan_all; scan_all()"'
    ), unsafe_allow_html=True)
    st.stop()

totals = report["totals"]
llm_findings = report.get("llm_findings") or report.get("claude_findings") or []
semgrep_findings = report.get("semgrep_findings") or []

provider = report.get("provider") or "LLM"
model = report.get("model") or "—"

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Semgrep findings", totals["semgrep"])
c2.metric("Semgrep high-sev", totals["semgrep_high"])
c3.metric(f"{provider.title()} findings", totals.get("llm", totals.get("claude", 0)))
c4.metric(f"{provider.title()} high-sev",
          totals.get("llm_high", totals.get("claude_high", 0)))
c5.metric("Combined total",
          totals["semgrep"] + totals.get("llm", totals.get("claude", 0)))

st.markdown(
    f'<div class="rg-row" style="margin:8px 0 16px 0">'
    f'{tag("scan target: " + report["target"], "blue")}'
    f'{tag("semgrep: " + ("on" if report["semgrep_available"] else "off"), "slate")}'
    f'{tag(provider + " · " + model, "violet")}'
    f"</div>",
    unsafe_allow_html=True,
)

st.divider()

# ---------------- Filters ----------------

fc1, fc2 = st.columns([1, 3])
sev_filter = fc1.multiselect(
    "Severity",
    ["high", "medium", "low", "error", "warning", "info"],
    default=["high", "medium", "low", "error", "warning", "info"],
)
search = fc2.text_input("Search rule / issue text", "")


def _filter(findings, key_msg, key_sev):
    out = []
    for f in findings:
        sev = (f.get(key_sev, "") or "").lower()
        msg = f.get(key_msg, "") or ""
        if sev not in sev_filter:
            # also accept high-as-error mapping
            if not (sev == "" and "high" in sev_filter):
                continue
        if search and search.lower() not in msg.lower() and search.lower() not in str(f.get("rule", "")).lower() and search.lower() not in str(f.get("category", "")).lower():
            continue
        out.append(f)
    return out


sg = _filter(semgrep_findings, "message", "severity")
llm = _filter(llm_findings, "issue", "severity")

# ---------------- Severity distribution chart ----------------

st.markdown('<div class="rg-section-title">Severity distribution</div>',
            unsafe_allow_html=True)
mix_rows = []
for f in sg:
    mix_rows.append({"scanner": "Semgrep", "severity": (f.get("severity") or "info").upper()})
for f in llm:
    mix_rows.append({"scanner": provider.title(), "severity": (f.get("severity") or "info").upper()})
if mix_rows:
    mix_df = pd.DataFrame(mix_rows)
    counts = mix_df.groupby(["scanner", "severity"]).size().reset_index(name="count")
    fig = px.bar(counts, x="scanner", y="count", color="severity",
                 color_discrete_map={"HIGH": "#dc2626", "ERROR": "#dc2626",
                                     "MEDIUM": "#ea580c", "WARNING": "#ea580c",
                                     "LOW": "#059669", "INFO": "#94a3b8"},
                 barmode="stack", text="count")
    fig.update_layout(**plotly_layout(height=260, showlegend=True))
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                  xanchor="right", x=1))
    st.plotly_chart(fig, width="stretch")

st.divider()

# ---------------- Side-by-side scanner panels ----------------

ps, pl = st.columns(2)

with ps:
    st.markdown('<div class="rg-section-title">Semgrep · static analysis</div>',
                unsafe_allow_html=True)
    if not report.get("semgrep_available"):
        st.info("Semgrep not installed — `pip install semgrep`.")
    if sg:
        for f in sg:
            sev = (f.get("severity") or "info").lower()
            variant = "red" if sev in ("error", "high") else "amber" if sev in ("warning", "medium") else "slate"
            rule_short = (f.get("rule") or "").split(".")[-1]
            st.markdown(
                f'<div class="rg-card" style="padding:10px 14px">'
                f'<div class="rg-row" style="justify-content:space-between">'
                f'<div><b>{rule_short}</b> '
                f'<span class="rg-muted" style="font-size:12px">line {f.get("line", "?")}</span></div>'
                f'<div>{tag(sev.upper(), variant)}</div></div>'
                f'<div class="rg-muted" style="margin-top:6px;font-size:13px">{f.get("message", "")}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("No Semgrep findings matching the current filters.")

with pl:
    st.markdown(f'<div class="rg-section-title">{provider.title()} · semantic review</div>',
                unsafe_allow_html=True)
    err = report.get("llm_error") or report.get("claude_error")
    if err:
        st.error(f"Semantic review error: {err}")
    if llm:
        for f in llm:
            sev = (f.get("severity") or "info").lower()
            variant = "red" if sev == "high" else "amber" if sev == "medium" else "slate"
            cat = f.get("category", "")
            st.markdown(
                f'<div class="rg-card" style="padding:10px 14px">'
                f'<div class="rg-row" style="justify-content:space-between">'
                f'<div><b>{cat}</b> '
                f'<span class="rg-muted" style="font-size:12px">line {f.get("line", "?")}</span></div>'
                f'<div>{tag(sev.upper(), variant)}</div></div>'
                f'<div style="margin-top:6px;font-size:13px;color:#1e293b">{f.get("issue", "")}</div>'
                f'<div style="margin-top:6px;font-size:12px;color:#475569"><b>Fix:</b> {f.get("fix", "")}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
    elif not err:
        st.info(f"No {provider} findings matching the current filters.")

st.divider()

# ---------------- Source viewer ----------------

st.markdown(f'<div class="rg-section-title">Source · {report["target"]}</div>',
            unsafe_allow_html=True)
src = ROOT / report["target"]
if src.exists():
    code = src.read_text()
    st.code(code, language="python", line_numbers=True)
else:
    st.warning(f"Source file {src} not found.")
