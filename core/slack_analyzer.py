"""Module 1 — Slack Intelligence Hub.

Loads the simplified GoEmotions train.tsv (text, emotion_ids, comment_id),
partitions ~1500 messages into 50 simulated "client Slack channels", and
runs Claude per channel to produce R/Y/G health scores.

We also compute a *ground-truth* sentiment per channel from the GoEmotions
emotion labels, so the page can report a Claude-vs-human accuracy number.

Usage:
    # 1. Build the manifest (no API key needed)
    python -c "from core.slack_analyzer import build_manifest; build_manifest()"
    # 2. Run Claude scoring (needs ANTHROPIC_API_KEY)
    python -c "from core.slack_analyzer import batch_analyze; batch_analyze()"
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import (
    DATA_DIR, RESULTS_DIR,
    active_model_id, active_provider, chat_with_llm,
    ensure_dirs, load_cached, save_cached,
)

TSV_PATH = DATA_DIR / "goemotions.tsv"
LABELS_PATH = DATA_DIR / "goemotions_labels.txt"
MANIFEST_PATH = DATA_DIR / "channel_manifest.json"
CACHE_PATH = RESULTS_DIR / "channel_health.json"

# Map GoEmotions categories to a coarse sentiment direction.
POSITIVE_EMOTIONS = {
    "admiration", "amusement", "approval", "caring", "desire", "excitement",
    "gratitude", "joy", "love", "optimism", "pride", "relief",
}
NEGATIVE_EMOTIONS = {
    "anger", "annoyance", "disappointment", "disapproval", "disgust",
    "embarrassment", "fear", "grief", "nervousness", "remorse", "sadness",
}

N_CHANNELS = 50
MESSAGES_PER_CHANNEL = 30

SYSTEM_PROMPT = """You are a customer service channel health analyst.
Given a batch of messages from a client Slack channel (the past 4 hours of
support conversation), score the channel on the following dimensions.
Respond with *ONLY* a JSON object — no prose, no markdown fence:
{
  "satisfaction": 1-5,
  "urgency": 1-5,
  "churn_signal": "none" | "possible" | "likely",
  "unanswered_count": int,
  "tone_trajectory": "improving" | "stable" | "deteriorating",
  "summary": "<=20 word explanation",
  "health": "green" | "yellow" | "red"
}"""


def _load_labels() -> list[str]:
    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            f"{LABELS_PATH} not found. Run `bash data/download.sh` first."
        )
    return [line.strip() for line in LABELS_PATH.read_text().splitlines() if line.strip()]


def _load_goemotions() -> pd.DataFrame:
    if not TSV_PATH.exists():
        raise FileNotFoundError(
            f"{TSV_PATH} not found. Run `bash data/download.sh` first."
        )
    df = pd.read_csv(
        TSV_PATH, sep="\t", names=["text", "emotion_ids", "comment_id"], header=None
    )
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.len() > 8]
    return df.reset_index(drop=True)


def _ground_truth_score(emotion_id_strs: list[str], labels: list[str]) -> float:
    """Map a channel's emotion-label mix to a sentiment score in [-1, +1]."""
    pos = neg = total = 0
    for s in emotion_id_strs:
        ids = [int(i) for i in s.split(",") if i.strip().isdigit()]
        for i in ids:
            if 0 <= i < len(labels):
                name = labels[i]
                total += 1
                if name in POSITIVE_EMOTIONS:
                    pos += 1
                elif name in NEGATIVE_EMOTIONS:
                    neg += 1
    if total == 0:
        return 0.0
    return round((pos - neg) / total, 3)


def _stable_channel_id(seed: int, i: int) -> str:
    h = hashlib.sha1(f"ch-{seed}-{i}".encode()).hexdigest()[:6]
    return f"#client-{h}"


def build_manifest(seed: int = 7, force: bool = False) -> dict:
    """Group GoEmotions messages into 50 simulated channels.

    No API key required — this is pure data preparation. Saves
    data/channel_manifest.json with messages and ground-truth sentiment.
    """
    ensure_dirs()
    if MANIFEST_PATH.exists() and not force:
        print(f"[slack] manifest exists, loading {MANIFEST_PATH}")
        return load_cached(MANIFEST_PATH)

    labels = _load_labels()
    df = _load_goemotions()
    rng = np.random.default_rng(seed)
    n_needed = N_CHANNELS * MESSAGES_PER_CHANNEL
    sample_idx = rng.choice(len(df), size=min(n_needed, len(df)), replace=False)
    sample = df.iloc[sample_idx].reset_index(drop=True)

    channels = []
    for i in range(N_CHANNELS):
        slice_df = sample.iloc[i * MESSAGES_PER_CHANNEL:(i + 1) * MESSAGES_PER_CHANNEL]
        if slice_df.empty:
            break
        gt = _ground_truth_score(slice_df["emotion_ids"].tolist(), labels)
        channels.append({
            "channel_id": _stable_channel_id(seed, i),
            "client_name": f"Client {chr(ord('A') + (i % 26))}{i // 26 + 1}",
            "messages": slice_df["text"].tolist(),
            "ground_truth_sentiment": gt,
        })

    manifest = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_channels": len(channels),
        "messages_per_channel": MESSAGES_PER_CHANNEL,
        "source": "goemotions train.tsv (Google Research)",
        "channels": channels,
    }
    save_cached(MANIFEST_PATH, manifest)
    print(f"[slack] saved {MANIFEST_PATH} with {len(channels)} channels")
    return manifest


def _parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found: {text[:200]}")
    return json.loads(match.group(0))


def analyze_channel(messages: list[str]) -> dict:
    batch = "\n".join(f"[{i + 1}] {m}" for i, m in enumerate(messages))
    text = chat_with_llm(
        system=SYSTEM_PROMPT,
        user=f"Channel messages from the past 4 hours:\n{batch}",
        max_tokens=500,
    )
    return _parse_json(text)


def batch_analyze(force: bool = False) -> dict:
    """Score every channel in the manifest. Caches to results/channel_health.json."""
    ensure_dirs()
    if CACHE_PATH.exists() and not force:
        cached = load_cached(CACHE_PATH)
        if cached:
            print(f"[slack] using cached {CACHE_PATH} (force=True to recompute)")
            return cached

    manifest = build_manifest()
    provider = active_provider()
    model = active_model_id()
    out = []
    print(f"[slack] analyzing {len(manifest['channels'])} channels via "
          f"{provider} ({model})...")
    for i, ch in enumerate(manifest["channels"], start=1):
        try:
            scores = analyze_channel(ch["messages"])
        except Exception as e:  # noqa: BLE001
            print(f"[slack] error on {ch['channel_id']}: {e}")
            continue
        out.append({
            "channel_id": ch["channel_id"],
            "client_name": ch["client_name"],
            "ground_truth_sentiment": ch["ground_truth_sentiment"],
            **scores,
        })
        print(f"[slack] {i}/{len(manifest['channels'])} "
              f"{ch['channel_id']} → {scores['health']} ({scores['summary']})")
        time.sleep(0.15)

    result = {
        "scored_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provider": provider,
        "model": model,
        "n_channels": len(out),
        "channels": out,
    }
    save_cached(CACHE_PATH, result)
    print(f"[slack] saved {CACHE_PATH}")
    return result


if __name__ == "__main__":
    batch_analyze()
