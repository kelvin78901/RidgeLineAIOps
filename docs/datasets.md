# Datasets

## Module 1: Slack Sentiment — GoEmotions

- **Source:** [Google Research GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions)
- **Paper:** Demszky et al. (2020) "GoEmotions: A Dataset of Fine-Grained Emotions"
- **Size:** 58K Reddit comments, 27 emotion labels + neutral
- **License:** Apache 2.0
- **Usage:** Group messages into 50 simulated "client channels", run sentiment analysis, compare Claude output against ground truth labels

### Key columns
- `text` — the message content
- `admiration`, `amusement`, `anger`, `annoyance`, `disapproval`, `disgust`, `fear`, `gratitude`, `joy`, `sadness`, `surprise` — binary emotion labels (0/1)

## Module 2: Churn Prediction — IBM Telco Customer Churn

- **Source:** [Kaggle / IBM](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,043 customers, 21 features
- **License:** Apache 2.0
- **Usage:** Train XGBoost churn classifier. Add synthetic Ridgeline features (slack_sentiment, qa_score_trend, sla_breach_count).

### Key columns
- `customerID` — unique identifier
- `tenure` — months as customer
- `MonthlyCharges`, `TotalCharges` — revenue metrics
- `Contract` — month-to-month / one year / two year
- `Churn` — target variable (Yes/No)

### Synthetic features we add
- `slack_sentiment` — normal(0.5, 0.3) for retained, normal(-0.3, 0.4) for churned
- `qa_score_trend` — normal(0.1, 0.1) for retained, normal(-0.2, 0.15) for churned
- `sla_breach_count` — poisson(0.5) for retained, poisson(3) for churned
- `csat_14d_change` — normal(0.05, 0.1) for retained, normal(-0.3, 0.2) for churned

## Module 3: LLM QA — Synthetic + Optional Twitter Support

- **Primary:** 20 synthetic support replies generated in code (covers range from excellent to poor)
- **Optional:** [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) — 3M tweets, sample 100
- **License:** CC0 1.0

## Module 4: Security — Custom Vulnerable Code

- **Source:** Hand-crafted `sample-code/vulnerable.py`
- **Patterns:** SQL injection, auth bypass (IDOR), XSS, hardcoded secrets, insecure object reference, debug mode
- **Reference:** OWASP Top 10 (2021)
