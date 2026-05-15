# Architecture Overview

## System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA SOURCES                                  │
│  Slack (400+ ch)  │  Gorgias/Intercom  │  Compass QA  │  GitHub │
└────────┬──────────┴────────┬───────────┴──────┬───────┴────┬────┘
         │                   │                  │            │
         ▼                   ▼                  ▼            ▼
┌─────────────┐    ┌─────────────┐    ┌──────────────┐  ┌─────────┐
│  MODULE 1   │    │  MODULE 2   │    │   MODULE 3   │  │MODULE 4 │
│  Slack Hub  │    │ Churn Model │    │   LLM QA     │  │Security │
│             │    │             │    │              │  │Scanner  │
│ Claude API  │    │  XGBoost    │    │  Claude API  │  │Semgrep  │
│ LLM-Judge   │    │  + SHAP     │    │  Multi-Trait │  │+ Claude │
└──────┬──────┘    └──────┬──────┘    └──────┬───────┘  └────┬────┘
       │                  │                  │               │
       │    sentiment     │   risk score     │  qa scores    │ vulns
       ▼                  ▼                  ▼               ▼
┌─────────────────────────────────────────────────────────────────┐
│                  UNIFIED DATA LAYER                              │
│                  PostgreSQL / JSON files                         │
│                  (per-client health records)                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  UNIFIED DASHBOARD                               │
│                  Streamlit multi-page app                        │
│                                                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │ Overview │ │Slack Hub │ │Churn Risk│ │  LLM QA  │           │
│  │ 4 KPIs   │ │ R/Y/G    │ │ SHAP     │ │ Rubric   │           │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

## Module Dependencies

| Module | Depends On | API Required | Can Run Offline |
|--------|-----------|-------------|-----------------|
| Churn Model | telco_churn.csv | No | Yes (after training) |
| LLM QA | rubrics/*.json | Claude API | Yes (with cached results) |
| Slack Hub | goemotions.csv | Claude API | Yes (with cached results) |
| Security | sample-code/*.py | Claude API | Partially (Semgrep only) |
| Dashboard | All module outputs | No | Yes (reads cached JSON) |

## Data Flow

1. **Ingest** — Each module reads from its data source (CSV, JSON, code files)
2. **Process** — ML model or Claude API evaluates the data
3. **Cache** — Results saved to `results/*.json` (critical for demo performance)
4. **Display** — Streamlit reads cached results and renders dashboards

## Key Design Decisions

- **XGBoost over deep learning** — 7K rows is too small for neural nets. XGBoost + SHAP gives explainability Robert needs.
- **Claude over fine-tuned models** — Ridgeline is a Claude Partner. Zero-shot with detailed prompts beats fine-tuning for their scale.
- **Streamlit over React** — Fastest path to working demo. Team can maintain it. No frontend build tooling needed.
- **File-based caching over database** — 60-person company, 50-80 clients. JSON files are sufficient. PostgreSQL is for production roadmap.
