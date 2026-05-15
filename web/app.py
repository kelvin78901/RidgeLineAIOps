"""FastAPI dashboard for Ridgeline AI Ops.

Reads the same precomputed JSON caches the Streamlit version did:
    results/channel_health.json   — Slack Hub
    results/churn_predictions.json + model/test_set.parquet + model/shap_values.json
    results/qa_results.json
    results/security_report.json
    results/agent_scenarios.json + results/agent_archive.json
    data/mock_crm.json

Run:
    uvicorn web.app:app --host 127.0.0.1 --port 8000 --reload
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.utils import (  # noqa: E402
    AVAILABLE_MODELS, DATA_DIR, MODEL_DIR, RESULTS_DIR, SAMPLE_CODE_DIR,
    active_model_id, active_provider, load_cached, save_cached,
)

WEB_DIR = ROOT / "web"
templates = Jinja2Templates(directory=str(WEB_DIR / "templates"))

app = FastAPI(title="Ridgeline AI Ops")
app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")
# Serve generated images (SHAP summary) from the results/ folder
app.mount("/assets", StaticFiles(directory=str(RESULTS_DIR)), name="assets")

NAV = [
    {"href": "/",          "label": "Overview",      "key": "overview"},
    {"href": "/slack",     "label": "Slack Hub",     "key": "slack"},
    {"href": "/churn",     "label": "Churn Model",   "key": "churn"},
    {"href": "/qa",        "label": "LLM QA",        "key": "qa"},
    {"href": "/security",  "label": "Security",      "key": "security"},
    {"href": "/agent",     "label": "AI Agent",      "key": "agent"},
    {"href": "/assistant", "label": "Assistant",     "key": "assistant"},
]


def _ctx(request: Request, active: str, **extra: Any) -> dict[str, Any]:
    """Build the common template context."""
    try:
        provider, model = active_provider(), active_model_id()
    except Exception:  # noqa: BLE001
        provider, model = "unknown", "unknown"
    ctx = {
        "request": request,
        "nav": NAV,
        "active": active,
        "provider": provider,
        "model": model,
        "today": date.today().isoformat(),
    }
    ctx.update(extra)
    return ctx


def _band(score: float, hi: float = 70, mid: float = 40) -> str:
    if score >= hi:
        return "red"
    if score >= mid:
        return "amber"
    return "green"


# ----------------------------- Overview -----------------------------

@app.get("/", response_class=HTMLResponse)
async def overview(request: Request) -> HTMLResponse:
    slack = load_cached(RESULTS_DIR / "channel_health.json")
    churn = load_cached(RESULTS_DIR / "churn_predictions.json")
    qa = load_cached(RESULTS_DIR / "qa_results.json")
    security = load_cached(RESULTS_DIR / "security_report.json")
    agent = load_cached(RESULTS_DIR / "agent_scenarios.json")

    # Slack KPIs
    sl_channels = (slack or {}).get("channels", [])
    sl_red = sum(1 for c in sl_channels if c.get("health") == "red")
    sl_amber = sum(1 for c in sl_channels if c.get("health") == "yellow")
    sl_green = sum(1 for c in sl_channels if c.get("health") == "green")

    # Churn KPIs
    ch_avg = ch_high = ch_total = None
    ch_top = []
    if churn:
        import pandas as pd  # noqa: PLC0415
        path = MODEL_DIR / "test_set.parquet"
        if path.exists():
            ts = pd.read_parquet(path)
            ch_avg = float(ts["_risk_score"].mean())
            ch_high = int((ts["_risk_score"] >= 70).sum())
            ch_total = len(ts)
        ch_top = churn.get("top_50_at_risk", [])[:10]

    # QA KPIs
    qa_avg = qa_flagged = qa_total = None
    if qa and qa.get("evaluations"):
        evals = qa["evaluations"]
        overalls = [e["scores"]["overall"] for e in evals]
        qa_avg = sum(overalls) / len(overalls) if overalls else 0
        qa_flagged = sum(1 for e in evals if e["scores"].get("flag_for_review"))
        qa_total = len(evals)

    # Security KPIs
    sec_total = sec_high = None
    if security:
        t = security["totals"]
        sec_total = t["semgrep"] + (t.get("llm") or t.get("claude") or 0)
        sec_high = t["semgrep_high"] + (t.get("llm_high") or t.get("claude_high") or 0)

    # Agent KPIs
    ag_total = ag_resolved = ag_escalated = ag_waiting = None
    if agent:
        scen = agent["scenarios"]
        ag_total = len(scen)
        ag_resolved = sum(1 for s in scen if s["result"]["status"] == "resolved")
        ag_escalated = sum(1 for s in scen if s["result"]["status"] == "escalated")
        ag_waiting = sum(1 for s in scen if s["result"]["status"] == "ended_no_action")

    # Trend strip (synthetic, deterministic)
    from core.utils import make_trend  # noqa: PLC0415
    trends = {
        "slack":    make_trend("slack-red",  base=sl_red or 4,            drift=0.05,  noise=2.0),
        "churn":    make_trend("churn-high", base=ch_high or 200,         drift=0.4,   noise=8.0),
        "qa":       make_trend("qa-avg",     base=(qa_avg or 0.55) * 100, drift=-0.1,  noise=2.5),
        "security": make_trend("sec-high",   base=sec_high or 6,          drift=-0.02, noise=1.0),
    }

    # Unified client watchlist (top 25 across sources)
    watchlist = []
    for c in ch_top:
        watchlist.append({
            "client": c["client_id"], "source": "churn_model",
            "score": c["risk_score"], "band": _band(c["risk_score"]),
            "note": f"Top driver: {c['top_drivers'][0]['feature']}" if c.get("top_drivers") else "",
            "extra": "CHURNED" if c.get("actually_churned") else "ACTIVE",
            "extra_variant": "red" if c.get("actually_churned") else "blue",
        })
    for c in sl_channels[:20]:
        if c.get("health") not in ("red", "yellow"):
            continue
        sat = c.get("satisfaction", 3)
        score = max(0.0, min(100.0, (5 - sat) * 20 + c.get("urgency", 3) * 4))
        watchlist.append({
            "client": c["channel_id"], "source": "slack_hub",
            "score": score,
            "band": "red" if c["health"] == "red" else "amber",
            "note": c.get("summary", ""),
            "extra": f"churn:{c.get('churn_signal','-')}",
            "extra_variant": ("red" if c.get("churn_signal") == "likely"
                              else "amber" if c.get("churn_signal") == "possible"
                              else "slate"),
        })
    watchlist.sort(key=lambda r: r["score"], reverse=True)
    watchlist = watchlist[:25]

    readiness = [
        {"name": "Slack Hub",    "ready": bool(slack),    "cmd": "core.slack_analyzer.batch_analyze"},
        {"name": "Churn Model",  "ready": bool(churn),    "cmd": "core.churn_trainer.train"},
        {"name": "LLM QA",       "ready": bool(qa),       "cmd": "core.qa_evaluator.batch_evaluate"},
        {"name": "Security",     "ready": bool(security), "cmd": "core.security_scanner.scan_all"},
        {"name": "AI Agent",     "ready": bool(agent),    "cmd": "core.agent_simulator.precompute_scenarios"},
    ]

    return templates.TemplateResponse(request, "overview.html", _ctx(
        request, "overview",
        kpis={
            "slack_red": sl_red, "slack_amber": sl_amber, "slack_green": sl_green,
            "slack_total": len(sl_channels),
            "churn_high": ch_high, "churn_avg": ch_avg, "churn_total": ch_total,
            "qa_avg": qa_avg, "qa_flagged": qa_flagged, "qa_total": qa_total,
            "sec_total": sec_total, "sec_high": sec_high,
            "agent_total": ag_total, "agent_resolved": ag_resolved,
            "agent_escalated": ag_escalated, "agent_waiting": ag_waiting,
        },
        trends_json=json.dumps(trends),
        watchlist=watchlist,
        readiness=readiness,
    ))


# ----------------------------- Slack Hub -----------------------------

@app.get("/slack", response_class=HTMLResponse)
async def slack_page(request: Request) -> HTMLResponse:
    results = load_cached(RESULTS_DIR / "channel_health.json")
    manifest = load_cached(DATA_DIR / "channel_manifest.json")
    channels = (results or {}).get("channels", []) if results else []

    red = sum(1 for c in channels if c.get("health") == "red")
    amber = sum(1 for c in channels if c.get("health") == "yellow")
    green = sum(1 for c in channels if c.get("health") == "green")
    likely = sum(1 for c in channels if c.get("churn_signal") == "likely")
    avg_sat = (sum(c.get("satisfaction", 0) for c in channels) / len(channels)) if channels else 0

    return templates.TemplateResponse(request, "slack.html", _ctx(
        request, "slack",
        kpis={"red": red, "amber": amber, "green": green, "likely": likely,
              "avg_sat": avg_sat, "total": len(channels)},
        channels=channels,
        channels_json=json.dumps(channels),
        has_results=bool(results),
        has_manifest=bool(manifest),
    ))


# ----------------------------- Churn Model -----------------------------

@app.get("/churn", response_class=HTMLResponse)
async def churn_page(request: Request, client_id: str | None = None) -> HTMLResponse:
    summary = load_cached(RESULTS_DIR / "churn_predictions.json")
    shap_records = load_cached(MODEL_DIR / "shap_values.json")
    path = MODEL_DIR / "test_set.parquet"

    test_rows = []
    if path.exists():
        import pandas as pd  # noqa: PLC0415
        ts = pd.read_parquet(path)
        test_rows = ts[["_client_id", "_risk_score", "_churn_actual"]].to_dict("records")

    if not summary or not test_rows or not shap_records:
        return templates.TemplateResponse(request, "churn.html", _ctx(
            request, "churn", has_data=False,
        ))

    # Default to highest-risk client
    sorted_by_risk = sorted(test_rows, key=lambda r: -r["_risk_score"])
    chosen_id = client_id or sorted_by_risk[0]["_client_id"]
    chosen_row = next((r for r in test_rows if r["_client_id"] == chosen_id),
                      sorted_by_risk[0])
    chosen_id = chosen_row["_client_id"]
    rec = shap_records[chosen_id]
    drivers = sorted(
        [
            {"feature": f, "value": v, "shap": s}
            for f, v, s in zip(rec["feature_names"], rec["feature_values"], rec["values"])
        ],
        key=lambda d: abs(d["shap"]), reverse=True,
    )[:12]

    risk = chosen_row["_risk_score"]
    band = _band(risk)
    high_n = sum(1 for r in test_rows if r["_risk_score"] >= 70)
    med_n = sum(1 for r in test_rows if 40 <= r["_risk_score"] < 70)
    low_n = sum(1 for r in test_rows if r["_risk_score"] < 40)

    # Histogram bins
    import math  # noqa: PLC0415
    hist = [0] * 20  # 5-point bins 0-100
    for r in test_rows:
        idx = min(int(r["_risk_score"] // 5), 19)
        hist[idx] += 1

    return templates.TemplateResponse(request, "churn.html", _ctx(
        request, "churn", has_data=True,
        summary=summary,
        client_list=[r["_client_id"] for r in sorted_by_risk],
        chosen=chosen_id, chosen_risk=risk, chosen_band=band,
        chosen_churned=bool(chosen_row["_churn_actual"]),
        drivers=drivers,
        drivers_json=json.dumps(drivers),
        histogram=hist,
        bands={"high": high_n, "med": med_n, "low": low_n},
        top20=summary["top_50_at_risk"][:20],
    ))


# ----------------------------- LLM QA -----------------------------

@app.get("/qa", response_class=HTMLResponse)
async def qa_page(request: Request) -> HTMLResponse:
    results = load_cached(RESULTS_DIR / "qa_results.json")
    if not results or not results.get("evaluations"):
        return templates.TemplateResponse(request, "qa.html", _ctx(request, "qa", has_data=False))

    evals = []
    for ev in results["evaluations"]:
        s = ev["scores"]
        evals.append({
            "reply_id": ev["reply_id"],
            "rubric": ev["rubric_id"],
            "client": ev["client_name"],
            "tier": ev["intended_tier"],
            "overall": s["overall"],
            "accuracy": s["accuracy"]["score"],
            "brand_voice": s["brand_voice"]["score"],
            "completeness": s["completeness"]["score"],
            "sla": s["sla"],
            "escalation": s["escalation"],
            "flag": bool(s["flag_for_review"]),
            "summary": s["summary"],
            "customer_message": ev["customer_message"],
            "agent_reply": ev["agent_reply"],
        })

    # Compute divergence (per reply_id, abs diff between rubrics)
    reply_groups: dict[str, list[dict]] = {}
    for e in evals:
        reply_groups.setdefault(e["reply_id"], []).append(e)
    divergence_rows = []
    for reply_id, rows in reply_groups.items():
        if len(rows) >= 2:
            overalls = [r["overall"] for r in rows]
            divergence_rows.append({
                "reply_id": reply_id,
                "divergence": max(overalls) - min(overalls),
                "rows": sorted(rows, key=lambda r: r["rubric"]),
                "customer_message": rows[0]["customer_message"],
                "agent_reply": rows[0]["agent_reply"],
                "tier": rows[0]["tier"],
            })
    divergence_rows.sort(key=lambda r: -r["divergence"])

    return templates.TemplateResponse(request, "qa.html", _ctx(
        request, "qa", has_data=True,
        evals=evals,
        evals_json=json.dumps(evals),
        rubrics=sorted({e["rubric"] for e in evals}),
        tiers=sorted({e["tier"] for e in evals}),
        divergence=divergence_rows[:3],
        kpis={
            "replies": len(reply_groups),
            "rubrics": len({e["rubric"] for e in evals}),
            "avg": sum(e["overall"] for e in evals) / len(evals),
            "flagged": sum(1 for e in evals if e["flag"]),
            "sla_fail": sum(1 for e in evals if e["sla"] == "fail"),
        },
    ))


# ----------------------------- Security -----------------------------

@app.get("/security", response_class=HTMLResponse)
async def security_page(request: Request) -> HTMLResponse:
    report = load_cached(RESULTS_DIR / "security_report.json")
    if not report:
        return templates.TemplateResponse(request, "security.html", _ctx(request, "security", has_data=False))

    code_text = ""
    src = ROOT / report["target"]
    if src.exists():
        code_text = src.read_text()

    semgrep = report.get("semgrep_findings") or []
    llm = report.get("llm_findings") or report.get("claude_findings") or []

    for f in semgrep:
        f["rule_short"] = (f.get("rule") or "").split(".")[-1]
        f["sev_norm"] = (f.get("severity") or "info").lower()

    for f in llm:
        f["sev_norm"] = (f.get("severity") or "info").lower()

    return templates.TemplateResponse(request, "security.html", _ctx(
        request, "security", has_data=True,
        report=report,
        semgrep=semgrep,
        llm=llm,
        code_text=code_text,
        provider_label=(report.get("provider") or "LLM").title(),
        model_label=report.get("model") or "—",
    ))


# ----------------------------- AI Agent -----------------------------

@app.get("/agent", response_class=HTMLResponse)
async def agent_page(request: Request, scenario: str | None = None) -> HTMLResponse:
    scenarios = load_cached(RESULTS_DIR / "agent_scenarios.json")
    archive = load_cached(RESULTS_DIR / "agent_archive.json") or {"tickets": []}
    crm = load_cached(DATA_DIR / "mock_crm.json")
    if not scenarios:
        return templates.TemplateResponse(request, "agent.html", _ctx(
            request, "agent", has_data=False,
        ))

    scen_list = scenarios["scenarios"]
    active_scenario = next(
        (s for s in scen_list if s["id"] == scenario), scen_list[0],
    )

    total = len(scen_list)
    resolved = sum(1 for s in scen_list if s["result"]["status"] == "resolved")
    escalated = sum(1 for s in scen_list if s["result"]["status"] == "escalated")
    waiting = sum(1 for s in scen_list if s["result"]["status"] == "ended_no_action")
    avg_turns = sum(s["result"]["turns"] for s in scen_list) / total if total else 0

    from core.agent_simulator import TOOLS  # noqa: PLC0415

    return templates.TemplateResponse(request, "agent.html", _ctx(
        request, "agent", has_data=True,
        scenarios=scen_list,
        active_scenario=active_scenario,
        archive=archive,
        crm=crm,
        tools=TOOLS,
        kpis={
            "total": total, "resolved": resolved, "escalated": escalated,
            "waiting": waiting, "avg_turns": avg_turns,
            "auto_pct": resolved / total if total else 0,
        },
    ))


@app.post("/api/agent/run")
async def api_agent_run(message: str = Form(...)) -> JSONResponse:
    """Drive the agent live and return the structured transcript (non-streaming)."""
    if not message.strip():
        raise HTTPException(400, "message required")
    from core.agent_simulator import append_to_archive, run_agent  # noqa: PLC0415
    try:
        result = run_agent(message.strip())
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": str(e)}, status_code=500)
    ticket_id = append_to_archive(result)
    return JSONResponse({"ticket_id": ticket_id, "result": result})


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, default=str)}\n\n"


@app.get("/api/agent/run_stream")
async def api_agent_run_stream(message: str) -> StreamingResponse:
    """Server-Sent Events: stream each tool call as the agent runs.

    Events emitted in order:
      - customer  : initial customer message
      - agent     : agent text reply (one per turn)
      - tool      : a tool call + its result (one per call, with status OK/DENIED)
      - done      : final status, ticket id, archive/escalation record
      - error     : transport-level error
    """
    if not message.strip():
        raise HTTPException(400, "message required")

    async def gen():
        try:
            from core.agent_simulator import (  # noqa: PLC0415
                append_to_archive, stream_agent,
            )
            yield _sse("customer", {"text": message})
            final_result = None
            for ev in stream_agent(message.strip()):
                yield _sse(ev["type"], ev["payload"])
                if ev["type"] == "_final":
                    final_result = ev["payload"]
            if final_result:
                ticket_id = append_to_archive(final_result)
                yield _sse("done", {
                    "ticket_id": ticket_id,
                    "status": final_result["status"],
                    "turns": final_result["turns"],
                    "archive": final_result.get("archive"),
                    "escalation": final_result.get("escalation"),
                })
        except Exception as e:  # noqa: BLE001
            yield _sse("error", {"message": str(e)})

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ----------------------------- LLM Assistant -----------------------------

@app.get("/assistant", response_class=HTMLResponse)
async def assistant_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "assistant.html", _ctx(request, "assistant"))


def _assistant_context() -> str:
    """Compact summary of all current module data — fed as system context."""
    bits = []
    slack = load_cached(RESULTS_DIR / "channel_health.json")
    if slack:
        chs = slack.get("channels", [])
        bits.append("SLACK HUB:")
        bits.append(f"  {len(chs)} channels — "
                    f"{sum(1 for c in chs if c['health']=='red')} red, "
                    f"{sum(1 for c in chs if c['health']=='yellow')} amber, "
                    f"{sum(1 for c in chs if c['health']=='green')} green")
        for c in chs[:8]:
            bits.append(f"  - {c['channel_id']} ({c['health']}): {c.get('summary','')[:120]}")

    churn = load_cached(RESULTS_DIR / "churn_predictions.json")
    if churn:
        bits.append("\nCHURN MODEL:")
        bits.append(f"  AUC={churn['auc']:.3f}, n_test={churn['n_test']}, "
                    f"churn_rate={churn['churn_rate']:.1%}")
        bits.append("  Top-5 highest-risk clients:")
        for c in churn.get("top_50_at_risk", [])[:5]:
            top = c["top_drivers"][0]["feature"] if c.get("top_drivers") else "-"
            bits.append(f"  - {c['client_id']}: {c['risk_score']}/100, "
                        f"churned={c['actually_churned']}, top_driver={top}")

    qa = load_cached(RESULTS_DIR / "qa_results.json")
    if qa and qa.get("evaluations"):
        evals = qa["evaluations"]
        avg = sum(e["scores"]["overall"] for e in evals) / len(evals)
        flagged = sum(1 for e in evals if e["scores"].get("flag_for_review"))
        bits.append(f"\nLLM QA: {len(evals)} evaluations, avg overall={avg:.2f}, "
                    f"{flagged} flagged for human review")

    security = load_cached(RESULTS_DIR / "security_report.json")
    if security:
        t = security["totals"]
        llm_n = t.get("llm") or t.get("claude") or 0
        bits.append(f"\nSECURITY: Semgrep={t['semgrep']} findings "
                    f"({t['semgrep_high']} high-sev), "
                    f"LLM semantic review={llm_n} findings")

    agent = load_cached(RESULTS_DIR / "agent_scenarios.json")
    if agent:
        scen = agent["scenarios"]
        resolved = sum(1 for s in scen if s["result"]["status"] == "resolved")
        escalated = sum(1 for s in scen if s["result"]["status"] == "escalated")
        bits.append(f"\nAI AGENT: {len(scen)} preset scenarios, "
                    f"{resolved} auto-resolved, {escalated} escalated")

    return "\n".join(bits) or "(no module data cached yet)"


ASSISTANT_SYSTEM = """You are the Ridgeline AI Ops co-pilot — a concise analyst.

OUTPUT FORMAT — every reply MUST be valid GitHub-Flavored Markdown:
- Lead with one short paragraph (no greeting, no "Sure, …", no preamble).
- Use **bold** for key terms, `inline code` for IDs / column names / commands,
  and fenced ``` blocks for multi-line code.
- Use `-` bullet lists when enumerating 3+ items; never wall-of-text.
- Use `##` headers ONLY if the answer covers 2+ distinct topics.
- Keep paragraphs at most 3 sentences. Cite specific numbers from the context.
- End with a follow-up suggestion only when genuinely useful (skip otherwise).

If the data below cannot answer the user's question, say so directly in one
sentence rather than guessing.

CURRENT DASHBOARD DATA
----------------------
{context}
----------------------
"""


@app.get("/api/assistant/chat_stream")
async def api_assistant_chat_stream(
    message: str,
    history: str = "[]",
    provider: str | None = None,
    model: str | None = None,
) -> StreamingResponse:
    """SSE stream of the assistant's reply, with optional provider/model override."""
    if not message.strip():
        raise HTTPException(400, "message required")
    try:
        past = json.loads(history)
        assert isinstance(past, list)
    except Exception:  # noqa: BLE001
        past = []

    async def gen():
        try:
            from core.utils import (  # noqa: PLC0415
                active_provider as _ap, active_model_id as _am,
                get_anthropic_client, get_deepseek_client,
            )
            chosen_provider = (provider or _ap()).lower()
            chosen_model = model or _am()
            system = ASSISTANT_SYSTEM.format(context=_assistant_context())
            messages = past + [{"role": "user", "content": message}]

            yield _sse("meta", {"provider": chosen_provider, "model": chosen_model})

            if chosen_provider == "anthropic":
                client = get_anthropic_client()
                with client.messages.stream(
                    model=chosen_model, max_tokens=1200,
                    system=system, messages=messages,
                ) as stream:
                    for text in stream.text_stream:
                        yield _sse("delta", {"text": text})
            elif chosen_provider == "deepseek":
                client = get_deepseek_client()
                stream = client.chat.completions.create(
                    model=chosen_model, max_tokens=1200,
                    messages=[{"role": "system", "content": system}, *messages],
                    stream=True,
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        yield _sse("delta", {"text": delta})
            else:
                yield _sse("error", {"message": f"unknown provider {chosen_provider}"})
                return

            yield _sse("done", {"provider": chosen_provider, "model": chosen_model})
        except Exception as e:  # noqa: BLE001
            yield _sse("error", {"message": str(e)})

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ----------------------------- Models + Assistant cache -----------------------------

@app.get("/api/models")
async def api_models() -> JSONResponse:
    """Return the catalog of providers and models for the UI switcher."""
    try:
        active_p, active_m = active_provider(), active_model_id()
    except Exception:  # noqa: BLE001
        active_p, active_m = "anthropic", "claude-sonnet-4-6"
    return JSONResponse({
        "active_provider": active_p,
        "active_model": active_m,
        "providers": AVAILABLE_MODELS,
    })


CONV_PATH = RESULTS_DIR / "assistant_history.json"


def _load_conversations() -> dict:
    return load_cached(CONV_PATH) or {"conversations": []}


def _save_conversations(payload: dict) -> None:
    save_cached(CONV_PATH, payload)


@app.get("/api/assistant/conversations")
async def api_assistant_list() -> JSONResponse:
    payload = _load_conversations()
    # Return lightweight summaries — drop messages from list view
    summaries = []
    for c in payload["conversations"]:
        summaries.append({
            "id": c["id"], "title": c["title"],
            "provider": c.get("provider"), "model": c.get("model"),
            "created_at": c["created_at"], "updated_at": c["updated_at"],
            "n_messages": len(c.get("messages", [])),
        })
    summaries.sort(key=lambda r: r["updated_at"], reverse=True)
    return JSONResponse({"conversations": summaries})


@app.get("/api/assistant/conversations/{conv_id}")
async def api_assistant_load(conv_id: str) -> JSONResponse:
    payload = _load_conversations()
    for c in payload["conversations"]:
        if c["id"] == conv_id:
            return JSONResponse(c)
    raise HTTPException(404, "conversation not found")


@app.post("/api/assistant/save")
async def api_assistant_save(
    id: str = Form(...),
    title: str = Form(...),
    provider: str = Form(""),
    model: str = Form(""),
    messages: str = Form(...),
) -> JSONResponse:
    try:
        msgs = json.loads(messages)
        assert isinstance(msgs, list)
    except Exception:  # noqa: BLE001
        raise HTTPException(400, "messages must be JSON list")

    import time
    payload = _load_conversations()
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    existing = next((c for c in payload["conversations"] if c["id"] == id), None)
    if existing:
        existing.update({
            "title": title[:80], "provider": provider, "model": model,
            "messages": msgs, "updated_at": now,
        })
    else:
        payload["conversations"].append({
            "id": id, "title": title[:80],
            "provider": provider, "model": model,
            "messages": msgs, "created_at": now, "updated_at": now,
        })
    _save_conversations(payload)
    return JSONResponse({"ok": True, "id": id})


@app.delete("/api/assistant/conversations/{conv_id}")
async def api_assistant_delete(conv_id: str) -> JSONResponse:
    payload = _load_conversations()
    before = len(payload["conversations"])
    payload["conversations"] = [c for c in payload["conversations"] if c["id"] != conv_id]
    _save_conversations(payload)
    return JSONResponse({"ok": True, "removed": before - len(payload["conversations"])})


# ----------------------------- Health -----------------------------

@app.get("/healthz")
async def healthz() -> JSONResponse:
    return JSONResponse({"ok": True})


@app.get("/favicon.ico")
async def favicon() -> RedirectResponse:
    return RedirectResponse(url="/static/favicon.svg")
