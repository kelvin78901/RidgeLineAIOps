#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$SCRIPT_DIR"

echo "=== Ridgeline AI Ops — Dataset Download ==="

# 1. Telco Customer Churn (IBM) — churn prediction
if [ ! -f "$DATA_DIR/telco_churn.csv" ]; then
  echo "[1/2] Downloading Telco Customer Churn dataset..."
  curl -sL -o "$DATA_DIR/telco_churn.csv" \
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
  echo "      → telco_churn.csv (7,043 rows)"
else
  echo "[1/2] telco_churn.csv already exists, skipping."
fi

# 2. GoEmotions (Google Research) — sentiment analysis
# The legacy full_dataset/goemotions_{1,2,3}.csv files were removed from the
# upstream repo; we use the simplified train.tsv (43K rows, text + emotion_ids).
if [ ! -f "$DATA_DIR/goemotions.tsv" ]; then
  echo "[2/2] Downloading GoEmotions train.tsv..."
  curl -sL -o "$DATA_DIR/goemotions.tsv" \
    "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/train.tsv"
  curl -sL -o "$DATA_DIR/goemotions_labels.txt" \
    "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/emotions.txt"
  echo "      → goemotions.tsv (43K rows) + goemotions_labels.txt"
else
  echo "[2/2] goemotions.tsv already exists, skipping."
fi

echo ""
echo "=== Done. Datasets saved to $DATA_DIR ==="
echo ""
echo "Optional: For Twitter Customer Support data (3M rows):"
echo "  kaggle datasets download -d thoughtvector/customer-support-on-twitter"
