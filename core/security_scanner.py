"""Module 4 — Security scanner: Semgrep (static) + Claude (semantic).

Runs Semgrep with the `auto` ruleset on a target file, then has Claude do a
semantic review focused on the vulnerability classes static analyzers tend to
miss (auth bypass, IDOR, business-logic flaws).

Usage:
    # Full pipeline (needs ANTHROPIC_API_KEY)
    python -c "from core.security_scanner import scan_all; scan_all()"
    # Semgrep only — no API key needed
    python -c "from core.security_scanner import scan_all; scan_all(claude=False)"
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from pathlib import Path

from .utils import (
    RESULTS_DIR, SAMPLE_CODE_DIR,
    active_model_id, active_provider, chat_with_llm,
    ensure_dirs, save_cached,
)

CACHE_PATH = RESULTS_DIR / "security_report.json"

CLAUDE_SYSTEM = """You are a senior security code reviewer.
Find vulnerabilities in the supplied source code. Focus on issues that static
analyzers commonly miss: authentication/authorization bypass, IDOR (insecure
direct object reference), broken access control, business-logic flaws, unsafe
deserialization, and information disclosure.

Respond with *ONLY* a JSON array — no prose, no markdown fence:
[
  {
    "severity": "high" | "medium" | "low",
    "category": "<short category, e.g. 'auth_bypass', 'sql_injection'>",
    "line": <int>,
    "issue": "<one sentence describing the problem>",
    "fix": "<one sentence describing the recommended fix>"
  }
]"""


def _semgrep_available() -> bool:
    return shutil.which("semgrep") is not None


def run_semgrep(filepath: Path) -> list[dict]:
    """Run Semgrep with --config=auto and return normalized findings."""
    if not _semgrep_available():
        return []
    proc = subprocess.run(
        ["semgrep", "--config=auto", "--json", "--quiet", str(filepath)],
        capture_output=True, text=True, timeout=180,
    )
    # Semgrep returns 0 for "no findings", 1 for "findings present", >1 for error
    if proc.returncode not in (0, 1) or not proc.stdout.strip():
        return []
    payload = json.loads(proc.stdout)
    out = []
    for r in payload.get("results", []):
        out.append({
            "rule": r.get("check_id", "?"),
            "severity": r.get("extra", {}).get("severity", "INFO").lower(),
            "line": r.get("start", {}).get("line", 0),
            "message": r.get("extra", {}).get("message", "").strip(),
        })
    return out


def _parse_json_array(text: str) -> list[dict]:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON array found: {text[:200]}")
    return json.loads(match.group(0))


def run_llm_review(filepath: Path) -> list[dict]:
    """Semantic review using the active LLM provider (Claude or DeepSeek)."""
    code = filepath.read_text()
    text = chat_with_llm(system=CLAUDE_SYSTEM, user=code, max_tokens=3000)
    return _parse_json_array(text)


def scan_file(filepath: Path, llm: bool = True) -> dict:
    """Run both scanners on a single file."""
    findings = {
        "file": str(filepath.relative_to(SAMPLE_CODE_DIR.parent)),
        "semgrep_available": _semgrep_available(),
        "semgrep": run_semgrep(filepath),
        "llm": None,
    }
    if llm:
        try:
            findings["llm"] = run_llm_review(filepath)
        except Exception as e:  # noqa: BLE001
            findings["llm_error"] = str(e)
    return findings


def scan_all(llm: bool = True, claude: bool | None = None) -> dict:
    """Scan sample-code/vulnerable.py and cache the report.

    When llm=False, only Semgrep runs (no API key needed). The `claude`
    kwarg is kept as an alias for backwards compatibility with earlier callers.
    """
    if claude is not None:
        llm = claude
    ensure_dirs()
    target = SAMPLE_CODE_DIR / "vulnerable.py"
    if not target.exists():
        raise FileNotFoundError(target)

    provider = active_provider() if llm else None
    model = active_model_id() if llm else None
    print(f"[sec] scanning {target} (llm={'%s/%s' % (provider, model) if llm else 'off'})...")
    report = scan_file(target, llm=llm)

    llm_findings = report["llm"] or []
    high = sum(1 for f in llm_findings if f.get("severity") == "high")
    severities = [f["severity"] for f in report["semgrep"]]

    summary = {
        "scanned_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provider": provider,
        "model": model,
        "target": report["file"],
        "semgrep_available": report["semgrep_available"],
        "semgrep_findings": report["semgrep"],
        # Keep `claude_*` keys for back-compat with the existing dashboard.
        "claude_findings": llm_findings,
        "claude_error": report.get("llm_error"),
        "llm_findings": llm_findings,
        "llm_error": report.get("llm_error"),
        "totals": {
            "semgrep": len(report["semgrep"]),
            "semgrep_high": sum(1 for s in severities if s.upper() in ("ERROR", "HIGH")),
            "claude": len(llm_findings),
            "claude_high": high,
            "llm": len(llm_findings),
            "llm_high": high,
        },
    }
    save_cached(CACHE_PATH, summary)
    print(f"[sec] saved {CACHE_PATH}")
    print(f"[sec] semgrep={summary['totals']['semgrep']} "
          f"llm={summary['totals']['llm']}")
    return summary


if __name__ == "__main__":
    scan_all()
