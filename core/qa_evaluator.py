"""Module 3 — LLM-as-Judge QA evaluation with per-client brand rubrics.

Loads the synthetic support replies in data/qa_replies.json and scores each
reply against each rubric in rubrics/*.json using Claude. Demonstrates that
the same reply scores very differently depending on the client brand voice.

Usage:
    python -c "from core.qa_evaluator import batch_evaluate; batch_evaluate()"
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from .utils import (
    DATA_DIR, RESULTS_DIR, RUBRICS_DIR,
    active_model_id, active_provider, chat_with_llm,
    ensure_dirs, load_cached, save_cached,
)

DEFAULT_REPLIES_PATH = DATA_DIR / "qa_replies.json"
CACHE_PATH = RESULTS_DIR / "qa_results.json"


def _build_system_prompt(rubric: dict) -> str:
    return f"""You are a QA evaluator for a multi-client customer service agency.
Score the agent's reply against this client's brand rubric.

CLIENT: {rubric['client_name']}
INDUSTRY: {rubric['industry']}
BRAND VOICE: {rubric['voice_description']}
FORBIDDEN WORDS / PHRASES: {', '.join(rubric['forbidden_words'])}
REQUIRED ELEMENTS: {', '.join(rubric['required_elements'])}
TONE KEYWORDS (positive signals): {', '.join(rubric['tone_keywords'])}

EXCELLENT EXAMPLE:
{rubric['examples']['excellent']}

ACCEPTABLE EXAMPLE:
{rubric['examples']['acceptable']}

POOR EXAMPLE:
{rubric['examples']['poor']}

Score on five dimensions (1=poor, 5=excellent unless noted). Respond with
*ONLY* a JSON object — no prose, no markdown fence:
{{
  "accuracy":     {{"score": 1-5, "reason": "<=15 words"}},
  "brand_voice":  {{"score": 1-5, "reason": "<=15 words"}},
  "completeness": {{"score": 1-5, "reason": "<=15 words"}},
  "sla":          "pass" | "fail",
  "escalation":   "pass" | "fail" | "n/a",
  "flag_for_review": true | false,
  "summary": "<=20 word verdict"
}}"""


def _parse_json(text: str) -> dict:
    """Extract the first JSON object from Claude's response."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in response: {text[:200]}")
    return json.loads(match.group(0))


def _weighted_overall(parsed: dict, weights: dict) -> float:
    """Weighted average on a 0-1 scale, using rubric scoring_weights."""
    s_acc = parsed["accuracy"]["score"] / 5
    s_voice = parsed["brand_voice"]["score"] / 5
    s_comp = parsed["completeness"]["score"] / 5
    s_sla = 1.0 if parsed["sla"] == "pass" else 0.0
    s_esc = {"pass": 1.0, "fail": 0.0, "n/a": 0.7}.get(parsed["escalation"], 0.7)
    return round(
        weights["accuracy"] * s_acc
        + weights["brand_voice"] * s_voice
        + weights["completeness"] * s_comp
        + weights["sla"] * s_sla
        + weights["escalation"] * s_esc,
        3,
    )


def evaluate_reply(reply_text: str, rubric: dict) -> dict:
    """Score one reply against one rubric. Returns the parsed JSON + overall."""
    system = _build_system_prompt(rubric)
    text = chat_with_llm(
        system=system,
        user=f"AGENT REPLY:\n{reply_text}",
        max_tokens=600,
    )
    parsed = _parse_json(text)
    parsed["overall"] = _weighted_overall(parsed, rubric["scoring_weights"])
    return parsed


def _load_rubrics() -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in RUBRICS_DIR.glob("*.json")}


def _load_replies(path: Path = DEFAULT_REPLIES_PATH) -> list[dict]:
    return json.loads(path.read_text())["replies"]


def batch_evaluate(force: bool = False) -> dict[str, Any]:
    """Score every reply × every rubric. Caches to results/qa_results.json."""
    ensure_dirs()
    if CACHE_PATH.exists() and not force:
        cached = load_cached(CACHE_PATH)
        if cached:
            print(f"[qa] using cached {CACHE_PATH} (force=True to recompute)")
            return cached

    rubrics = _load_rubrics()
    replies = _load_replies()
    provider = active_provider()
    model = active_model_id()
    out: list[dict] = []
    print(f"[qa] evaluating {len(replies)} replies × {len(rubrics)} rubrics "
          f"= {len(replies) * len(rubrics)} {provider} ({model}) calls...")

    for i, reply in enumerate(replies, start=1):
        for rubric_id, rubric in rubrics.items():
            try:
                scores = evaluate_reply(reply["agent_reply"], rubric)
            except Exception as e:  # noqa: BLE001
                print(f"[qa] error on {reply['id']} × {rubric_id}: {e}")
                continue
            out.append({
                "reply_id": reply["id"],
                "rubric_id": rubric_id,
                "client_name": rubric["client_name"],
                "customer_message": reply["customer_message"],
                "agent_reply": reply["agent_reply"],
                "intended_tier": reply["intended_tier"],
                "scores": scores,
            })
            print(f"[qa] {i}/{len(replies)} {reply['id']} × {rubric_id} "
                  f"→ overall={scores['overall']}")
        time.sleep(0.2)  # gentle rate-limit cushion

    result = {
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provider": provider,
        "model": model,
        "n_replies": len(replies),
        "rubric_ids": list(rubrics.keys()),
        "evaluations": out,
    }
    save_cached(CACHE_PATH, result)
    print(f"[qa] saved {CACHE_PATH}")
    return result


if __name__ == "__main__":
    batch_evaluate()
