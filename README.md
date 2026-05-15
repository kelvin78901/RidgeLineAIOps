# Ridgeline AI Ops

A working demo of five AI modules for a customer-service outsourcing agency, built for a Johns Hopkins IT Consulting Practicum. Each module solves a real pain point on open-source data — no real customer data is used.

| Module | What it does | Stack |
|---|---|---|
| **Slack Hub** | Scores 50 simulated client channels (R/Y/G health, churn signal, urgency) | LLM scoring of GoEmotions text |
| **Churn Model** | Ranks customers by churn risk with per-client SHAP drivers | XGBoost + SHAP on IBM Telco Churn (7,043 customers) + synthetic Ridgeline signals |
| **LLM QA** | Scores support replies against multi-brand rubrics to expose voice divergence | LLM-as-judge, weighted rubric scoring |
| **Security Scanner** | Two-layer code review (static + semantic) on intentionally vulnerable Python | Semgrep + LLM |
| **AI Agent** | Bounded-autonomy customer-service agent with scoped tool catalog + human handoff | LLM tool use with server-enforced policy caps |
| **Assistant** | Co-pilot chat that answers questions about live dashboard data | Streaming LLM with persisted history |

## Quick start

```bash
# 1. Clone & enter
git clone https://github.com/kelvin78901/ridgeline-ai-ops.git
cd ridgeline-ai-ops

# 2. Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# macOS only — XGBoost needs libomp
brew install libomp
# Optional but recommended for Module 4
pip install semgrep

# 3. API key (see "LLM providers" below)
cp .env.example .env
$EDITOR .env   # add ANTHROPIC_API_KEY or DEEPSEEK_API_KEY

# 4. Download datasets
bash data/download.sh

# 5. Train churn model (no API needed)
python -c "from core.churn_trainer import train; train()"

# 6. Precompute LLM-driven modules (uses your API key)
python -c "from core.qa_evaluator import batch_evaluate; batch_evaluate()"
python -c "from core.slack_analyzer import batch_analyze;  batch_analyze()"
python -c "from core.security_scanner import scan_all;     scan_all()"
python -c "from core.agent_simulator import precompute_scenarios; precompute_scenarios()"

# 7. Run the dashboard
uvicorn web.app:app --host 127.0.0.1 --port 8000 --reload
# open http://127.0.0.1:8000
```

Step 4 takes ~30s. Step 5 trains in ~10s on CPU (AUC ≈ 0.85). Step 6 makes ~50 LLM calls and costs roughly **$0.50 to $2** depending on which provider and model you pick.

## LLM providers

Two providers are supported behind a single `chat_with_llm()` helper. Switch them by editing `.env`:

```ini
LLM_PROVIDER=anthropic        # or "deepseek"

ANTHROPIC_API_KEY=sk-ant-...  # https://console.anthropic.com/
DEEPSEEK_API_KEY=sk-...       # https://platform.deepseek.com/api_keys

# optional overrides — leave blank to use defaults
# ANTHROPIC_MODEL_ID=claude-sonnet-4-6
# DEEPSEEK_MODEL_ID=deepseek-v4-flash
```

The dashboard also exposes a **model dropdown in the sidebar** that overrides the env default per request (applies to the Assistant page and live agent runs).

| Provider | Models in the picker |
|---|---|
| Anthropic | Claude Sonnet 4.6 · Opus 4.7 · Haiku 4.5 |
| DeepSeek  | V4 Flash · Chat · Reasoner |

Batch precompute scripts (`batch_*`, `scan_all`, `precompute_scenarios`) always use the env-configured default.

## Architecture

```
                 ┌─────────────┐    ┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────────┐
                 │  Slack Hub  │    │ Churn Model │    │   LLM QA     │    │  Security   │    │  AI Agent   │
                 │  (LLM)      │    │ (XGBoost)   │    │  (LLM judge) │    │  (Semgrep +  │    │  (LLM tools)│
                 │             │    │             │    │              │    │   LLM)       │    │             │
                 └─────┬───────┘    └─────┬───────┘    └──────┬───────┘    └──────┬──────┘    └──────┬──────┘
                       │ JSON             │ pkl + JSON       │ JSON              │ JSON             │ JSON
                       ▼                  ▼                  ▼                   ▼                  ▼
                  ┌────────────────────────────────────────────────────────────────────────────────────┐
                  │                                results/ + model/                                   │
                  └──────────────────────────────────────┬─────────────────────────────────────────────┘
                                                         │
                                                         ▼
                                            ┌────────────────────────┐
                                            │  FastAPI + Jinja2 +    │
                                            │  TailwindCSS dashboard │
                                            │  + SSE streaming       │
                                            └────────────────────────┘
```

```
ridgeline-ai-ops/
├── README.md
├── CLAUDE.md                  # original project brief
├── requirements.txt
├── .env.example
├── core/                      # data pipeline (the heart)
│   ├── utils.py                  # shared helpers + LLM dispatch (chat_with_llm)
│   ├── churn_trainer.py          # train() — XGBoost + SHAP, no API
│   ├── slack_analyzer.py         # build_manifest() + batch_analyze()
│   ├── qa_evaluator.py           # batch_evaluate() — reply × rubric scoring
│   ├── security_scanner.py       # scan_all() — Semgrep + LLM review
│   └── agent_simulator.py        # run_agent() + stream_agent() + tool catalog
├── data/
│   ├── download.sh               # fetches Telco Churn + GoEmotions
│   ├── telco_churn.csv           # generated
│   ├── goemotions.tsv            # generated
│   ├── channel_manifest.json     # 50 simulated client channels
│   ├── qa_replies.json           # 20 synthetic support replies
│   └── mock_crm.json             # CRM the agent's tools query
├── rubrics/
│   ├── nimbus_formal.json        # formal brand rubric
│   └── orbit_casual.json         # casual brand rubric
├── sample-code/
│   ├── vulnerable.py             # intentionally vulnerable
│   └── fixed.py                  # fixed reference
├── model/                     # generated by training step
├── results/                   # generated by precompute steps
│   ├── churn_predictions.json
│   ├── channel_health.json
│   ├── qa_results.json
│   ├── security_report.json
│   ├── agent_scenarios.json
│   ├── agent_archive.json
│   ├── assistant_history.json    # chat history (auto-saved)
│   └── shap_summary.png
├── web/                       # FastAPI dashboard
│   ├── app.py                    # routes + SSE endpoints + history API
│   ├── templates/
│   │   ├── base.html             # layout + sidebar + model picker
│   │   ├── _macros.html          # kpi, tag, health_pill, empty
│   │   ├── overview.html
│   │   ├── slack.html
│   │   ├── churn.html
│   │   ├── qa.html
│   │   ├── security.html
│   │   ├── agent.html            # streaming live agent UI
│   │   └── assistant.html        # chat with history + stop + markdown
│   └── static/
│       ├── app.css               # bubbles, code viewer, markdown, table
│       └── app.js                # model picker, markdown render, escape
├── pages/                     # legacy Streamlit (kept for reference)
└── docs/                      # original engineering spec + architecture
```

## Modules in detail

### 1 · Slack Hub
50 simulated client channels of ~30 messages each, drawn from Google Research GoEmotions (43K Reddit comments simplified). For each channel the LLM returns JSON with `satisfaction`, `urgency`, `churn_signal`, `tone_trajectory`, `unanswered_count`, `summary`, `health`. A ground-truth sentiment score is computed independently from the GoEmotions emotion labels so the dashboard can show LLM-vs-human agreement.

### 2 · Churn Model
XGBoost (200 trees, depth 5) on IBM Telco Customer Churn (7,043 customers, 21 features). Four synthetic Ridgeline-style features are added with deliberate ~25% label noise so they behave like weak real-world indicators rather than oracles:
- `slack_sentiment` · `qa_score_trend` · `sla_breach_count` · `csat_14d_change`

Test AUC ≈ **0.85**. Per-client SHAP waterfall + global SHAP summary plot.

### 3 · LLM QA
20 hand-authored support replies × 2 brand rubrics (`nimbus_formal`, `orbit_casual`) = 40 LLM evaluations. The same casual reply commonly scores ~0.97 for the casual brand and ~0.42 for the formal brand, demonstrating why per-client rubrics matter for multi-client agencies. Heatmap + box plots + divergence cards in the UI.

### 4 · Security Scanner
Two-layer pipeline on `sample-code/vulnerable.py` (6 marked vulnerabilities):
- **Semgrep** with `--config=auto` — catches Stripe key, SQL injection, XSS, debug mode (~11 findings)
- **LLM semantic review** — catches the IDOR/auth bypass at line 35 and the missing refund-amount validation that Semgrep misses (~10 findings)

### 5 · AI Agent (bounded autonomy)
A real Claude tool-use loop, not a script. The agent receives a customer message and decides which tools to call:

| Tool | Type | Server-enforced cap |
|---|---|---|
| `lookup_customer`, `lookup_order`, `check_refund_eligibility` | read | — |
| `issue_refund` | write | ≤ $500 |
| `apply_discount_code` | write | ≤ 15% |
| `update_shipping_address` | write | only while `ready_to_ship` / `processing` |
| `escalate_to_human` | handoff | always available |
| `archive_conversation` | log | required at end of resolved cases |

Blocked intents (never reachable by any tool, force escalation): `delete_account`, `share_other_customer_data`, `modify_employee_records`, `release_unreleased_product_info`.

5 preset scenarios are pre-run and cached; the page also has a **live streaming runner** that pipes each tool call to the browser via Server-Sent Events as it happens.

### 6 · Assistant
A streaming co-pilot chat. The system prompt receives a compact summary of every module so it can answer dashboard questions with the real numbers. Features:

- **Sidebar history** — every conversation auto-saves to `results/assistant_history.json`, click any past chat to reload
- **Stop button** — turns into "Stop" mid-stream; closes the EventSource and preserves any partial reply
- **Model picker in the sidebar** — choose Anthropic Sonnet/Opus/Haiku or DeepSeek V4 Flash / Chat / Reasoner per request
- **Markdown rendering** — replies are rendered with `marked` + `DOMPurify` so lists, code blocks, headers, tables, bold/italic all display correctly

## Verification

```bash
# Type-check / smoke-test pages
curl -sf http://127.0.0.1:8000/healthz
for p in / /slack /churn /qa /security /agent /assistant; do
  curl -s -o /dev/null -w "$p → %{http_code}\n" http://127.0.0.1:8000$p
done

# SSE smoke
curl -sN "http://127.0.0.1:8000/api/assistant/chat_stream?message=Reply%20with%20pong"
curl -sN "http://127.0.0.1:8000/api/agent/run_stream?message=lookup%20order%20ORD-4521"
```

## Tech stack

- **Backend**: FastAPI · Uvicorn · Jinja2 · pandas · NumPy · XGBoost · SHAP · scikit-learn
- **LLM**: Anthropic SDK (streaming + tool use) · OpenAI-compatible client for DeepSeek
- **Static analysis**: Semgrep
- **Frontend**: TailwindCSS (CDN) · Plotly.js (CDN) · marked + DOMPurify (CDN) · vanilla JS + EventSource

## License & data

All datasets are open-source (Apache 2.0 / CC-BY) and synthetic data is generated in this repo. No real Ridgeline customer data is used anywhere. The `sample-code/vulnerable.py` is intentionally insecure for demo purposes — do not deploy.

## Acknowledgements

- IBM Telco Customer Churn dataset
- Google Research GoEmotions dataset (Demszky et al. 2020)
- Anthropic Claude · DeepSeek · Semgrep
