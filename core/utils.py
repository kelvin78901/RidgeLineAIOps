"""Shared helpers: paths, caching, LLM client, theme, dashboard utilities."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MODEL_DIR = ROOT / "model"
RUBRICS_DIR = ROOT / "rubrics"
SAMPLE_CODE_DIR = ROOT / "sample-code"

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

MODEL_ID = None  # legacy alias — modules use active_model_id() instead

# Switcher catalog — surfaced in the UI dropdown. Order matters (default first).
AVAILABLE_MODELS = {
    "anthropic": [
        {"id": "claude-sonnet-4-6",        "label": "Claude Sonnet 4.6",  "tier": "balanced"},
        {"id": "claude-opus-4-7",          "label": "Claude Opus 4.7",    "tier": "max-quality"},
        {"id": "claude-haiku-4-5-20251001","label": "Claude Haiku 4.5",   "tier": "fastest"},
    ],
    "deepseek": [
        {"id": "deepseek-v4-flash",        "label": "DeepSeek V4 Flash",  "tier": "fastest"},
        {"id": "deepseek-chat",            "label": "DeepSeek Chat",      "tier": "balanced"},
        {"id": "deepseek-reasoner",        "label": "DeepSeek Reasoner",  "tier": "max-quality"},
    ],
}

PALETTE = {
    "bg": "#ffffff",
    "surface": "#f7f8fb",
    "card": "#ffffff",
    "border": "#e6e8ee",
    "border_strong": "#c8ccd6",
    "text": "#0f172a",
    "text_muted": "#475569",
    "text_faint": "#64748b",
    "primary": "#4f46e5",
    "primary_soft": "#eef2ff",
    "success": "#059669",
    "success_soft": "#d1fae5",
    "warning": "#ea580c",
    "warning_soft": "#ffedd5",
    "danger": "#dc2626",
    "danger_soft": "#fee2e2",
    "info": "#0284c7",
    "info_soft": "#e0f2fe",
}


def ensure_dirs() -> None:
    for d in (DATA_DIR, RESULTS_DIR, MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_cached(path: str | os.PathLike) -> Any | None:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return None
    with p.open() as f:
        return json.load(f)


def save_cached(path: str | os.PathLike, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump(obj, f, indent=2, default=str)


# ----------------------------- LLM providers -----------------------------

def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass


def active_provider() -> str:
    _load_env()
    return os.environ.get("LLM_PROVIDER", "anthropic").strip().lower()


def active_model_id() -> str:
    _load_env()
    provider = active_provider()
    if provider == "anthropic":
        return os.environ.get("ANTHROPIC_MODEL_ID", "").strip() or DEFAULT_ANTHROPIC_MODEL
    if provider == "deepseek":
        return os.environ.get("DEEPSEEK_MODEL_ID", "").strip() or DEFAULT_DEEPSEEK_MODEL
    raise RuntimeError(f"Unknown LLM_PROVIDER={provider!r}.")


def get_anthropic_client():
    _load_env()
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key or key == "sk-ant-your-key-here":
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Add a real key to .env, then re-run.")
    import anthropic
    return anthropic.Anthropic(api_key=key)


def get_deepseek_client():
    _load_env()
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key or key == "sk-deepseek-your-key-here":
        raise RuntimeError("DEEPSEEK_API_KEY is not set. Add a real key to .env, then re-run.")
    from openai import OpenAI
    return OpenAI(api_key=key, base_url=DEEPSEEK_BASE_URL)


def chat_with_llm(
    system: str, user: str, max_tokens: int = 600,
    provider: str | None = None, model: str | None = None,
) -> str:
    provider = (provider or active_provider()).lower()
    model = model or active_model_id()

    if provider == "anthropic":
        client = get_anthropic_client()
        resp = client.messages.create(
            model=model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text

    if provider == "deepseek":
        client = get_deepseek_client()
        resp = client.chat.completions.create(
            model=model, max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""

    raise RuntimeError(f"Unknown LLM provider={provider!r}.")


# ----------------------------- Theme + UI -----------------------------

THEME_CSS = """
<style>
    :root {
        --rg-bg:        #f7f8fb;
        --rg-card:      #ffffff;
        --rg-border:    #e6e8ee;
        --rg-border-2:  #d6d9e0;
        --rg-text:      #0f172a;
        --rg-text-2:    #475569;
        --rg-text-3:    #64748b;
        --rg-primary:   #4f46e5;
        --rg-shadow:    0 1px 2px rgba(15,23,42,0.04), 0 1px 1px rgba(15,23,42,0.02);
    }

    html, body, [class*="css"], .stApp, .stApp * {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text",
                     "Inter", "Segoe UI", "Helvetica Neue", sans-serif;
        font-feature-settings: "tnum" 1, "cv11" 1;
    }
    .stApp { background-color: var(--rg-bg); color: var(--rg-text); }

    /* Global wrap defaults — fixes long IDs / JSON overflow */
    .stApp, .stApp * {
        overflow-wrap: anywhere;
        word-break: normal;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid var(--rg-border);
    }
    section[data-testid="stSidebar"] *,
    section[data-testid="stSidebar"] a span { color: var(--rg-text) !important; }
    section[data-testid="stSidebar"] [aria-selected="true"] {
        background: #eef2ff !important; color: var(--rg-primary) !important;
        border-radius: 6px;
    }

    /* Typography */
    h1 { color: var(--rg-text) !important; font-weight: 700; letter-spacing: -0.02em;
         font-size: 22px; line-height: 1.25; }
    h2 { color: var(--rg-text) !important; font-weight: 600; font-size: 18px;
         line-height: 1.3; }
    h3 { color: var(--rg-text) !important; font-weight: 600; font-size: 16px;
         line-height: 1.35; }
    h4 { color: var(--rg-text) !important; font-weight: 600; font-size: 14px;
         line-height: 1.4; }
    p, li, span, label, div { color: var(--rg-text); line-height: 1.55; }
    code, pre, .language-python, .stCodeBlock {
        font-family: "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace !important;
        font-size: 12.5px;
    }

    /* KPI metric cards */
    div[data-testid="stMetric"] {
        background: var(--rg-card);
        border: 1px solid var(--rg-border);
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: var(--rg-shadow);
    }
    div[data-testid="stMetric"] label {
        color: var(--rg-text-3) !important;
        font-weight: 500; font-size: 12px;
        text-transform: uppercase; letter-spacing: 0.06em;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--rg-text) !important; font-weight: 700;
        font-size: 26px; line-height: 1.1; margin-top: 4px;
    }
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: var(--rg-text-3) !important; font-size: 12px; font-weight: 500;
    }

    /* Data tables */
    div[data-testid="stDataFrame"], div[data-testid="stTable"] {
        background: var(--rg-card);
        border: 1px solid var(--rg-border);
        border-radius: 8px;
    }
    .stDataFrame [role="columnheader"] {
        background:#f1f3f8 !important; color:#334155 !important;
        font-weight: 600 !important; text-transform: uppercase; font-size: 11px;
        letter-spacing: 0.04em;
    }

    /* Form controls */
    .stMultiSelect [data-baseweb="tag"] {
        background: #eef2ff !important; color: var(--rg-primary) !important;
        border-radius: 6px !important;
    }
    .stSlider [data-baseweb="slider"] > div > div { background: var(--rg-primary) !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px; border-bottom: 1px solid var(--rg-border);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: var(--rg-text-2) !important;
        border-radius: 6px 6px 0 0 !important;
        padding: 8px 14px !important;
    }
    .stTabs [aria-selected="true"] {
        color: var(--rg-primary) !important;
        background: #eef2ff !important;
        border-bottom: 2px solid var(--rg-primary) !important;
    }

    /* Dividers */
    hr { border-color: var(--rg-border) !important; margin: 18px 0 !important; }

    /* Custom cards */
    .rg-card {
        background: var(--rg-card);
        border: 1px solid var(--rg-border);
        border-radius: 10px;
        padding: 14px 18px;
        margin: 8px 0;
        box-shadow: var(--rg-shadow);
        overflow-wrap: anywhere;
    }
    .rg-card h4 { margin: 0 0 6px 0; font-size: 14px; }
    .rg-card.rg-card-tight { padding: 10px 14px; }
    .rg-muted { color: var(--rg-text-3); font-size: 13px; line-height: 1.5; }
    .rg-section-title {
        font-size: 11px; font-weight: 600; letter-spacing: 0.1em;
        text-transform: uppercase; color: var(--rg-text-3);
        margin: 18px 0 8px 0;
    }
    .rg-row {
        display:flex; align-items:center; gap: 6px; flex-wrap:wrap;
        line-height: 1.6;
    }
    .rg-spread { display:flex; align-items:flex-start; justify-content:space-between; gap:12px; }

    /* Tag / pill system — atomic, never break mid-pill */
    .tag {
        display: inline-flex; align-items:center;
        padding: 2px 9px; border-radius: 6px;
        font-size: 11px; font-weight: 600; letter-spacing: 0.02em;
        border: 1px solid transparent; line-height: 18px;
        white-space: nowrap;
        font-variant-numeric: tabular-nums;
    }
    .tag-red    { background:#fee2e2; color:#b91c1c; border-color:#fecaca; }
    .tag-amber  { background:#fef3c7; color:#a16207; border-color:#fde68a; }
    .tag-green  { background:#d1fae5; color:#047857; border-color:#a7f3d0; }
    .tag-blue   { background:#eef2ff; color:#4338ca; border-color:#e0e7ff; }
    .tag-violet { background:#f5f3ff; color:#6d28d9; border-color:#ede9fe; }
    .tag-slate  { background:#f1f5f9; color:#334155; border-color:#e2e8f0; }
    .tag-orange { background:#ffedd5; color:#9a3412; border-color:#fed7aa; }

    .dot { display:inline-block; width:7px; height:7px; border-radius:50%;
           margin-right:6px; vertical-align:middle; }
    .dot-red    { background:#dc2626; }
    .dot-amber  { background:#ea580c; }
    .dot-green  { background:#059669; }
    .dot-blue   { background:#4f46e5; }

    /* Banner */
    .rg-banner {
        background: #ffffff;
        border: 1px solid var(--rg-border);
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: var(--rg-shadow);
    }
    .rg-banner .crumb {
        font-size: 11px; color: var(--rg-text-3);
        letter-spacing: 0.1em; text-transform: uppercase; font-weight: 600;
    }
    .rg-banner h1 { margin: 4px 0 6px 0; font-size: 22px; }
    .rg-banner p { margin: 0; color: var(--rg-text-2); font-size: 13px; max-width: 920px; }

    /* Empty state */
    .rg-empty {
        background: var(--rg-card);
        border: 1px dashed var(--rg-border-2);
        border-radius: 10px;
        padding: 24px; color: var(--rg-text-3);
    }
    .rg-empty code {
        color: var(--rg-primary); background: #eef2ff;
        padding: 2px 6px; border-radius: 4px; border: 1px solid #e0e7ff;
        font-size: 12px; white-space: pre-wrap;
    }

    /* Chat bubbles */
    .rg-bubble {
        background: var(--rg-card);
        border: 1px solid var(--rg-border);
        border-radius: 10px;
        padding: 12px 14px;
        margin: 6px 0;
        box-shadow: var(--rg-shadow);
        overflow-wrap: anywhere;
    }
    .rg-bubble.customer {
        background: #eef2ff;
        border-color: #e0e7ff;
    }
    .rg-bubble .who {
        font-size: 11px; font-weight: 600; letter-spacing: 0.06em;
        text-transform: uppercase; color: var(--rg-text-3);
        margin-bottom: 4px;
    }
    .rg-bubble .body {
        color: var(--rg-text);
        white-space: pre-wrap;
        word-break: normal;
        overflow-wrap: anywhere;
        font-size: 13.5px; line-height: 1.55;
    }

    /* Streamlit code/json output — let it scroll horizontally instead of breaking */
    .stCodeBlock pre, .stCodeBlock code, pre code {
        white-space: pre !important;
        overflow-x: auto !important;
        word-break: normal !important;
        overflow-wrap: normal !important;
    }

    /* Buttons */
    .stButton button {
        border-radius: 8px;
        border: 1px solid var(--rg-border-2);
        font-weight: 500;
    }
    .stButton button[kind="primary"] {
        background: var(--rg-primary);
        border-color: var(--rg-primary);
    }

    /* Expander */
    .streamlit-expanderHeader, [data-testid="stExpander"] summary {
        font-weight: 500 !important; font-size: 13px !important;
    }

    /* Top padding */
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
</style>
"""


def apply_page_chrome(title: str, breadcrumb: str = "Ridgeline AI Ops",
                      subtitle: str | None = None) -> None:
    import streamlit as st
    st.set_page_config(page_title=title, layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    sub = (f'<p>{subtitle}</p>' if subtitle else "")
    st.markdown(
        f'<div class="rg-banner">'
        f'<div class="crumb">{breadcrumb}</div>'
        f'<h1>{title}</h1>{sub}'
        f"</div>",
        unsafe_allow_html=True,
    )


def empty_state(message: str, command: str | None = None) -> str:
    cmd = (f'<br><br>Run: <code>{command}</code>' if command else "")
    return f'<div class="rg-empty"><strong>{message}</strong>{cmd}</div>'


def tag(label: str, variant: str = "slate") -> str:
    return f'<span class="tag tag-{variant}">{label}</span>'


def status_tag(level: str) -> str:
    """level: red | yellow | green (case-insensitive) — also accepts 'amber'."""
    v = (level or "").lower()
    if v in ("red", "high", "critical", "fail"):
        return tag("HIGH", "red")
    if v in ("yellow", "amber", "medium", "warn"):
        return tag("MEDIUM", "amber")
    if v in ("green", "low", "ok", "pass", "none"):
        return tag("LOW", "green")
    return tag(v.upper() or "INFO", "slate")


def health_pill(health: str) -> str:
    h = (health or "").lower()
    mapping = {
        "red":    ("<span class='dot dot-red'></span>RED",    "red"),
        "yellow": ("<span class='dot dot-amber'></span>AMBER", "amber"),
        "amber":  ("<span class='dot dot-amber'></span>AMBER", "amber"),
        "green":  ("<span class='dot dot-green'></span>GREEN", "green"),
    }
    label, variant = mapping.get(h, (h.upper(), "slate"))
    return f'<span class="tag tag-{variant}">{label}</span>'


# ----------------------------- Synthetic dashboard data -----------------------------

def make_trend(seed: str, days: int = 14, base: float = 50.0,
               drift: float = 0.0, noise: float = 5.0) -> list[dict]:
    """Deterministic synthetic time-series for the last `days` days."""
    h = int(hashlib.sha1(seed.encode()).hexdigest()[:8], 16)
    out = []
    today = date.today()
    value = base
    for i in range(days):
        d = today - timedelta(days=days - 1 - i)
        wiggle = ((h >> i) & 0xFF) / 255.0 - 0.5
        value = max(0.0, base + drift * i + wiggle * noise * 2)
        out.append({"date": d.isoformat(), "value": round(value, 2)})
    return out


def plotly_layout(height: int = 280, showlegend: bool = False) -> dict:
    """Common layout used across pages."""
    return dict(
        height=height,
        showlegend=showlegend,
        margin=dict(l=10, r=10, t=24, b=10),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(color="#0f172a", size=12,
                  family="-apple-system, BlinkMacSystemFont, Inter, sans-serif"),
        xaxis=dict(gridcolor="#eef2f7", linecolor="#cbd5e1", zerolinecolor="#eef2f7"),
        yaxis=dict(gridcolor="#eef2f7", linecolor="#cbd5e1", zerolinecolor="#eef2f7"),
    )
