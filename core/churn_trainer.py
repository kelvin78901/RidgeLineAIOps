"""Module 2 — XGBoost churn model + SHAP explainability.

Trains on the IBM Telco Customer Churn dataset, then augments with synthetic
"Ridgeline-style" features (slack_sentiment, qa_score_trend, sla_breach_count,
csat_14d_change) so the model demonstrates the multi-signal approach described
in the consulting pitch.

Usage:
    python -c "from core.churn_trainer import train; train()"
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from .utils import DATA_DIR, MODEL_DIR, RESULTS_DIR, ensure_dirs, save_cached

CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

SYNTHETIC_FEATURES = [
    "slack_sentiment", "qa_score_trend", "sla_breach_count", "csat_14d_change",
]


def _load_telco() -> pd.DataFrame:
    path = DATA_DIR / "telco_churn.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `bash data/download.sh` first."
        )
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df


def _add_synthetic_features(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Inject four noisy Ridgeline-style multi-signal features.

    Churned customers have *slightly* worse Slack sentiment, QA trend, SLA
    breaches, and CSAT change — but with enough variance that the signals
    don't trivially leak the label. We also flip ~25% of each class so the
    synthetic columns behave like real-world weak indicators rather than
    perfect oracles.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    churned = df["Churn"].to_numpy() == 1

    # 25% label noise — these "signals" are wrong about the customer a quarter of the time
    noise_mask = rng.random(n) < 0.25
    effective_churn = np.where(noise_mask, ~churned, churned)

    def per_class(churn_mean, churn_sd, retain_mean, retain_sd):
        return np.where(
            effective_churn,
            rng.normal(churn_mean, churn_sd, n),
            rng.normal(retain_mean, retain_sd, n),
        )

    df = df.copy()
    df["slack_sentiment"] = per_class(-0.05, 0.55, 0.15, 0.55)
    df["qa_score_trend"] = per_class(-0.05, 0.25, 0.05, 0.25)
    df["sla_breach_count"] = np.where(
        effective_churn, rng.poisson(1.5, n), rng.poisson(0.8, n)
    ).astype(float)
    df["csat_14d_change"] = per_class(-0.05, 0.25, 0.03, 0.20)
    return df


def _encode(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(
        df.drop(columns=["customerID"]), columns=CATEGORICAL_COLS, drop_first=False
    )


def train(seed: int = 42, test_size: float = 0.2) -> dict:
    ensure_dirs()
    print("[churn] loading telco data...")
    df = _load_telco()
    print(f"[churn] {len(df)} rows, {df['Churn'].mean():.1%} churn rate")

    df = _add_synthetic_features(df, seed=seed)
    encoded = _encode(df)

    y = encoded["Churn"].astype(int)
    X = encoded.drop(columns=["Churn"]).astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    print("[churn] training XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    auc = float(roc_auc_score(y_test, proba))
    print(f"\n=== Classification report ===\n{classification_report(y_test, pred)}")
    print(f"=== AUC: {auc:.4f} ===\n")

    print("[churn] computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Persist artifacts
    joblib.dump(model, MODEL_DIR / "churn_model.pkl")
    joblib.dump(explainer, MODEL_DIR / "shap_explainer.pkl")

    test_set = X_test.copy()
    test_set["_churn_actual"] = y_test.values
    test_set["_risk_score"] = (proba * 100).round(1)
    # Stable, friendly client IDs derived from the row position
    test_set["_client_id"] = [f"CLI-{i:04d}" for i in range(len(test_set))]
    test_set.to_parquet(MODEL_DIR / "test_set.parquet")

    # SHAP values keyed by client id for fast lookup in the page
    shap_records = {}
    for idx, cid in enumerate(test_set["_client_id"].tolist()):
        shap_records[cid] = {
            "base_value": float(explainer.expected_value),
            "values": [float(v) for v in shap_values[idx]],
            "feature_names": X.columns.tolist(),
            "feature_values": [float(v) for v in X_test.iloc[idx].tolist()],
        }
    save_cached(MODEL_DIR / "shap_values.json", shap_records)

    # Top-50 at-risk customers, with their top SHAP drivers, into results/
    high_risk_idx = np.argsort(-proba)[:50]
    top_risk = []
    for rank, idx in enumerate(high_risk_idx, start=1):
        contribs = list(zip(X.columns, shap_values[idx], X_test.iloc[idx]))
        contribs.sort(key=lambda r: abs(r[1]), reverse=True)
        top_risk.append({
            "rank": rank,
            "client_id": test_set["_client_id"].iloc[idx],
            "risk_score": float(round(proba[idx] * 100, 1)),
            "actually_churned": bool(y_test.iloc[idx]),
            "top_drivers": [
                {"feature": f, "shap": float(s), "value": float(v)}
                for f, s, v in contribs[:5]
            ],
        })

    summary = {
        "trained_at": pd.Timestamp.utcnow().isoformat(),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "auc": auc,
        "churn_rate": float(df["Churn"].mean()),
        "high_risk_threshold": 70,
        "synthetic_features": SYNTHETIC_FEATURES,
        "top_50_at_risk": top_risk,
    }
    save_cached(RESULTS_DIR / "churn_predictions.json", summary)

    # SHAP summary plot for the page banner
    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, X_test, show=False, plot_size=(9, 6))
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "shap_summary.png", dpi=140, bbox_inches="tight",
                facecolor="#0f172a")
    plt.close()
    print(f"[churn] artifacts saved to {MODEL_DIR} and {RESULTS_DIR}")
    return summary


if __name__ == "__main__":
    train()
