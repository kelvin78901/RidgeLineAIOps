# Ridgeline AI Ops — Claude Code Project Instructions

## What This Project Is

A working demo of 5 AI modules for Ridgeline Agency (a customer service outsourcing company with 60 people). The demo uses open-source datasets to prove each module works. This is for a Johns Hopkins IT Consulting Practicum final presentation.

## Tech Stack

- **Language:** Python 3.11+
- **UI:** Streamlit (multi-page app)
- **ML:** XGBoost + SHAP (churn prediction)
- **LLM:** Anthropic Claude API (sentiment analysis, QA evaluation, code review)
- **Data:** pandas, numpy
- **Viz:** plotly, matplotlib

## Project Structure

```
ridgeline-ai-ops/
├── CLAUDE.md                  ← You are here
├── requirements.txt
├── .env.example
├── app.py                     ← Unified Streamlit entry point
├── pages/
│   ├── 1_slack_hub.py         ← Module 1: Slack sentiment dashboard
│   ├── 2_churn_model.py       ← Module 2: Churn prediction + SHAP
│   ├── 3_llm_qa.py            ← Module 3: LLM-as-Judge QA evaluator
│   └── 4_security.py          ← Module 4: Code security scanner
├── core/
│   ├── slack_analyzer.py      ← Claude sentiment analysis pipeline
│   ├── churn_trainer.py       ← XGBoost training + SHAP explainer
│   ├── qa_evaluator.py        ← Claude multi-trait QA evaluation
│   ├── security_scanner.py    ← Semgrep + Claude code review
│   └── utils.py               ← Shared utilities
├── data/
│   ├── download.sh            ← Dataset download script
│   ├── telco_churn.csv        ← IBM Telco Churn dataset (7,043 rows)
│   └── goemotions.csv         ← Google GoEmotions (58K rows)
├── rubrics/
│   ├── nimbus_formal.json     ← Formal brand rubric
│   └── orbit_casual.json      ← Casual brand rubric
├── sample-code/
│   ├── vulnerable.py          ← Intentionally vulnerable code
│   └── fixed.py               ← Fixed version
├── results/                   ← Pre-computed results (cached)
│   ├── channel_health.json
│   ├── churn_predictions.json
│   ├── qa_results.json
│   └── security_report.json
├── model/                     ← Trained model artifacts
│   ├── churn_model.pkl
│   └── shap_explainer.pkl
└── docs/
    ├── engineering_spec.html  ← Full technical spec
    ├── architecture.md        ← Architecture overview
    └── datasets.md            ← Dataset documentation
```

## Build Order (Follow This Sequence)

### Step 1: Setup
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add ANTHROPIC_API_KEY
bash data/download.sh
```

### Step 2: Module 2 — Churn Model (Build First, No API Needed)
1. Load `data/telco_churn.csv`
2. Clean data, encode categoricals, add synthetic Ridgeline features
3. Train XGBoost, save model + SHAP explainer to `model/`
4. Generate SHAP summary plot
5. Build `pages/2_churn_model.py` Streamlit page
6. Test: `streamlit run pages/2_churn_model.py`

### Step 3: Module 3 — LLM QA (Needs Claude API)
1. Load rubrics from `rubrics/`
2. Create 20 synthetic support replies (or sample from dataset)
3. Run Claude evaluator on each reply × each rubric
4. Cache results to `results/qa_results.json`
5. Build `pages/3_llm_qa.py` with comparison view
6. Test: `streamlit run pages/3_llm_qa.py`

### Step 4: Module 1 — Slack Hub (Needs Claude API)
1. Load GoEmotions data, group into 50 simulated channels
2. Run Claude batch sentiment on each channel (cache aggressively!)
3. Save to `results/channel_health.json`
4. Build `pages/1_slack_hub.py` with R/Y/G dashboard
5. Test: `streamlit run pages/1_slack_hub.py`

### Step 5: Module 4 — Security Scanner
1. Load `sample-code/vulnerable.py`
2. Run Semgrep (if available) or simulate results
3. Run Claude semantic review
4. Save to `results/security_report.json`
5. Build `pages/4_security.py`

### Step 6: Unified Dashboard
1. Build `app.py` as Streamlit multi-page app
2. Overview page: 4 KPI cards + client health table
3. Each module accessible via sidebar navigation
4. Test: `streamlit run app.py`

## Critical Design Rules

1. **Cache everything.** Every Claude API call result must be saved to `results/` as JSON. The demo must run without live API calls during presentation. Use `st.cache_data` + file-based caching.

2. **Pre-compute, don't live-compute.** Train the model once, save artifacts. Run Claude evaluations once, save results. The Streamlit app should load pre-computed data and display it instantly.

3. **Dark theme.** Use `st.set_page_config(layout="wide")` and custom CSS with dark background (#0f172a), blue accents (#38bdf8), clean typography.

4. **One insight per page.** Each Streamlit page should have: one headline metric at top, one interactive visualization in the middle, one detail table at the bottom.

5. **Real data, real numbers.** Don't use placeholder text. Show actual model accuracy, actual SHAP values, actual Claude evaluation scores.

## Environment Variables

```
ANTHROPIC_API_KEY=sk-ant-xxxxx   # Required for modules 1, 3, 4
```

## Key Dependencies

See `requirements.txt`. Do NOT use: tensorflow, torch, keras (too heavy). Do NOT use: langchain (unnecessary abstraction). Keep it simple: anthropic SDK + scikit-learn + xgboost + streamlit.
