"""Module 5 — AI service agent with bounded autonomy.

The agent answers a single customer message via a multi-turn Claude tool-use
loop. Tools query a mock CRM and execute write actions (refunds, discounts,
escalations, archive). Hard limits are enforced server-side so the LLM cannot
breach policy even if it tries — the tool returns a permission_denied result
and the agent is expected to either correct its plan or escalate.

Usage:
    python -c "from core.agent_simulator import precompute_scenarios; precompute_scenarios()"

The Streamlit page can also drive the agent live via run_agent(customer_message).
"""
from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from .utils import (
    DATA_DIR, RESULTS_DIR, active_model_id, active_provider,
    ensure_dirs, get_anthropic_client, load_cached, save_cached,
)

CRM_PATH = DATA_DIR / "mock_crm.json"
ARCHIVE_PATH = RESULTS_DIR / "agent_archive.json"
SCENARIOS_PATH = RESULTS_DIR / "agent_scenarios.json"

MAX_TURNS = 8


# ----------------------------- Tool catalog -----------------------------

TOOLS: list[dict] = [
    {
        "name": "lookup_customer",
        "description": "Look up a customer record by email. Returns name, tier, lifetime value, join date.",
        "input_schema": {
            "type": "object",
            "properties": {"email": {"type": "string", "description": "Customer email address"}},
            "required": ["email"],
        },
    },
    {
        "name": "lookup_order",
        "description": "Look up an order by order ID. Returns status, amount, items, shipping address.",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string", "description": "Order ID, e.g. ORD-4521"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "check_refund_eligibility",
        "description": "Check whether an order is eligible for refund per policy (within 30 days, not already refunded).",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "issue_refund",
        "description": "Issue a refund for an order. HARD LIMIT: amount_cents <= 50000 ($500). Larger refunds require human approval and will be rejected.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "amount_cents": {"type": "integer", "description": "Refund amount in cents"},
                "reason": {"type": "string"},
            },
            "required": ["order_id", "amount_cents", "reason"],
        },
    },
    {
        "name": "apply_discount_code",
        "description": "Generate a one-time discount code for the customer. HARD LIMIT: percent <= 15. Larger discounts will be rejected.",
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string"},
                "percent": {"type": "integer", "description": "Discount percentage 1-15"},
                "reason": {"type": "string"},
            },
            "required": ["customer_id", "percent", "reason"],
        },
    },
    {
        "name": "update_shipping_address",
        "description": "Update the shipping address of an order. Only allowed while order status is 'ready_to_ship' or 'processing'. In-transit or delivered orders cannot be modified.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "new_address": {"type": "string"},
            },
            "required": ["order_id", "new_address"],
        },
    },
    {
        "name": "escalate_to_human",
        "description": "Hand the conversation to a human agent. Use this when the request is outside policy, requires approval, or after 2 failed permission denials. The conversation ends after this call.",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": ["billing", "shipping", "product", "account", "other"]},
                "priority": {"type": "string", "enum": ["low", "normal", "high", "urgent"]},
                "summary": {"type": "string", "description": "One-sentence summary for the human agent"},
                "reason": {"type": "string", "description": "Why escalation is needed (e.g. 'requested 50% discount, exceeds 15% cap')"},
            },
            "required": ["category", "priority", "summary", "reason"],
        },
    },
    {
        "name": "archive_conversation",
        "description": "Archive the conversation outcome. Call this exactly once at the end of every conversation that was resolved without escalation. Captures category, sentiment, summary and actions taken for compliance.",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": ["billing", "shipping", "product", "account", "other"]},
                "sentiment_at_close": {"type": "string", "enum": ["positive", "neutral", "negative"]},
                "summary": {"type": "string"},
                "actions_taken": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["category", "sentiment_at_close", "summary", "actions_taken"],
        },
    },
]


SYSTEM_PROMPT = """You are a customer service AI agent operating under bounded autonomy.

POLICY HARD LIMITS (enforced server-side — the tool will reject violations):
- Refunds: max $500 (50000 cents). Larger refunds REQUIRE escalation.
- Discounts: max 15%. Larger discounts REQUIRE escalation.
- Shipping address changes: only while order status is ready_to_ship or processing.
- You may NOT: delete accounts, share another customer's data, modify employee
  records, or disclose unreleased product information. Any such request must
  be escalated.

WORKFLOW (follow strictly):
1. Identify the customer if possible (ask for or look up their email / order ID).
2. Look up the relevant order(s) before making any commitment about status.
3. Take the smallest in-policy action that resolves the issue. Prefer one tool
   call at a time so you can react to its result.
4. If you hit a permission_denied result, STOP and reason about whether:
   (a) a smaller in-policy action exists (try once), OR
   (b) the request is fundamentally out of scope (escalate immediately).
5. Before ending: either call archive_conversation (if you resolved it) OR
   call escalate_to_human (if you couldn't). Never both.

TONE: professional, concise, empathetic. Acknowledge the customer's issue
before stating what you can do.

OUTPUT: Always communicate with the customer in plain text, then make any
required tool call. Keep customer-facing replies under 3 sentences.
"""


# ----------------------------- Mock backend -----------------------------

def _load_crm() -> dict:
    if not CRM_PATH.exists():
        raise FileNotFoundError(CRM_PATH)
    return json.loads(CRM_PATH.read_text())


def _find(items: list[dict], key: str, value: str) -> dict | None:
    for it in items:
        if it.get(key) == value:
            return it
    return None


def _check_policy(name: str, args: dict, crm: dict) -> tuple[bool, str]:
    pol = crm["policies"]
    if name == "issue_refund":
        if args.get("amount_cents", 0) > pol["max_refund_cents"]:
            return False, (
                f"permission_denied: requested refund exceeds policy cap of "
                f"${pol['max_refund_cents']/100:.0f}. Escalate to a human supervisor."
            )
    if name == "apply_discount_code":
        if args.get("percent", 0) > pol["max_discount_pct"]:
            return False, (
                f"permission_denied: requested discount exceeds policy cap of "
                f"{pol['max_discount_pct']}%. Escalate to a human supervisor."
            )
    if name == "update_shipping_address":
        order = _find(crm["orders"], "order_id", args.get("order_id", ""))
        if not order:
            return False, "order_not_found"
        if order["status"] not in pol["shipping_address_change_window_status"]:
            return False, (
                f"permission_denied: address change not allowed once order "
                f"status is '{order['status']}'. Escalate to a human."
            )
    return True, ""


def _execute(name: str, args: dict, crm: dict, audit: list[dict]) -> dict:
    """Execute a tool call against the in-memory CRM snapshot."""
    ok, deny = _check_policy(name, args, crm)
    if not ok:
        audit.append({"tool": name, "args": args, "result": "DENIED", "detail": deny})
        return {"status": "permission_denied", "detail": deny}

    if name == "lookup_customer":
        c = _find(crm["customers"], "email", args["email"])
        result = c or {"status": "not_found", "email": args["email"]}

    elif name == "lookup_order":
        o = _find(crm["orders"], "order_id", args["order_id"])
        result = o or {"status": "not_found", "order_id": args["order_id"]}

    elif name == "check_refund_eligibility":
        o = _find(crm["orders"], "order_id", args["order_id"])
        if not o:
            result = {"eligible": False, "reason": "order_not_found"}
        else:
            already_refunded = any(r["order_id"] == o["order_id"]
                                   for r in crm["refunds"])
            result = {
                "eligible": not already_refunded,
                "reason": "already_refunded" if already_refunded else "ok",
                "max_refundable_cents": o["amount_cents"],
            }

    elif name == "issue_refund":
        crm["refunds"].append({
            "refund_id": f"RFD-{len(crm['refunds'])+1:03d}",
            "order_id": args["order_id"],
            "amount_cents": args["amount_cents"],
            "issued_at": time.strftime("%Y-%m-%d"),
            "status": "processed",
            "reason": args["reason"],
        })
        result = {"status": "refund_processed",
                  "amount_cents": args["amount_cents"],
                  "eta_business_days": "3-5"}

    elif name == "apply_discount_code":
        code = f"SAVE{args['percent']}-{int(time.time())%10000:04d}"
        result = {"status": "code_issued", "code": code,
                  "percent": args["percent"], "expires_in_days": 30}

    elif name == "update_shipping_address":
        order = _find(crm["orders"], "order_id", args["order_id"])
        order["shipping_address"] = args["new_address"]
        result = {"status": "address_updated", "new_address": args["new_address"]}

    elif name == "escalate_to_human":
        result = {"status": "escalated", **args}

    elif name == "archive_conversation":
        result = {"status": "archived", **args}

    else:
        result = {"status": "unknown_tool"}

    audit.append({"tool": name, "args": args, "result": "OK", "detail": result})
    return result


# ----------------------------- Agent loop -----------------------------

def _serialize_tool_use(block) -> dict:
    return {"type": "tool_use", "id": block.id, "name": block.name,
            "input": block.input}


def run_agent(customer_message: str, max_turns: int = MAX_TURNS) -> dict:
    """Run the agent on one customer message. Returns a structured transcript."""
    client = get_anthropic_client()
    model = active_model_id()
    crm = deepcopy(_load_crm())  # isolate mutations to this conversation

    audit: list[dict] = []
    transcript: list[dict] = []  # [{role: "customer"|"agent"|"tool", ...}]
    transcript.append({"role": "customer", "text": customer_message})

    messages = [{"role": "user", "content": customer_message}]
    final_status = "ongoing"
    archive_record: dict | None = None
    escalation_record: dict | None = None

    for turn in range(max_turns):
        resp = client.messages.create(
            model=model,
            max_tokens=900,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # Capture any text the assistant produced this turn
        text_parts = [b.text for b in resp.content if b.type == "text"]
        if text_parts:
            transcript.append({"role": "agent", "text": "\n".join(text_parts)})

        if resp.stop_reason != "tool_use":
            final_status = "ended_no_action"
            break

        # Append the assistant's tool-using turn to the message history verbatim
        messages.append({"role": "assistant",
                         "content": [b.model_dump() for b in resp.content]})

        tool_results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            result = _execute(block.name, block.input, crm, audit)
            transcript.append({
                "role": "tool",
                "tool": block.name,
                "input": block.input,
                "result": result,
            })
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result),
            })
            if block.name == "archive_conversation" and result.get("status") == "archived":
                archive_record = block.input
                final_status = "resolved"
            elif block.name == "escalate_to_human" and result.get("status") == "escalated":
                escalation_record = block.input
                final_status = "escalated"

        messages.append({"role": "user", "content": tool_results})

        if final_status in ("resolved", "escalated"):
            # Give the assistant one final turn to send a customer-facing
            # closer message after archive/escalate (without another tool call).
            try:
                closer = client.messages.create(
                    model=model, max_tokens=200, system=SYSTEM_PROMPT,
                    tools=TOOLS, messages=messages,
                )
                tail = [b.text for b in closer.content if b.type == "text"]
                if tail:
                    transcript.append({"role": "agent", "text": "\n".join(tail)})
            except Exception:  # noqa: BLE001
                pass
            break

    return {
        "model": model,
        "provider": active_provider(),
        "status": final_status,
        "turns": turn + 1,
        "customer_message": customer_message,
        "transcript": transcript,
        "audit": audit,
        "archive": archive_record,
        "escalation": escalation_record,
    }


# ----------------------------- Preset scenarios -----------------------------

SCENARIOS = [
    {
        "id": "S1",
        "label": "Duplicate charge (in-policy refund)",
        "message": "Hi, I think I was charged twice for order ORD-4521. My email is alex@example.com. Can you refund the duplicate?",
        "expectation": "Look up customer + order, confirm duplicate (ORD-4522), issue $129.99 refund (under cap), archive.",
    },
    {
        "id": "S2",
        "label": "Discount request beyond cap (escalation)",
        "message": "I've been a customer since 2022. casey@example.com here. I want a 50% off code for my next order or I'm leaving.",
        "expectation": "Look up customer (platinum), try 15% (cap), customer insistence → escalate as billing/high.",
    },
    {
        "id": "S3",
        "label": "Address change before ship",
        "message": "Order ORD-8821 — please change my shipping address to 88 Cedar Way, Austin TX 78704. drew@example.com",
        "expectation": "Look up order (ready_to_ship), update address, confirm + archive.",
    },
    {
        "id": "S4",
        "label": "Delete my account (out of scope)",
        "message": "Cancel my account and erase all my data. eve@example.com",
        "expectation": "Account deletion is in blocked_intents → escalate as account/normal.",
    },
    {
        "id": "S5",
        "label": "Large refund (above cap)",
        "message": "I want a full refund on ORD-9012 for $780. The jacket has stitching defects. casey@example.com",
        "expectation": "Eligibility ok, but $780 > $500 cap → permission_denied, then escalate.",
    },
]


def precompute_scenarios(force: bool = False) -> dict:
    """Run all 5 scenarios live and cache transcripts for the dashboard."""
    ensure_dirs()
    if SCENARIOS_PATH.exists() and not force:
        return load_cached(SCENARIOS_PATH)

    archive_existing = load_cached(ARCHIVE_PATH) or {"tickets": []}
    out = []
    print(f"[agent] running {len(SCENARIOS)} preset scenarios via "
          f"{active_provider()} ({active_model_id()})...")
    for s in SCENARIOS:
        print(f"[agent]  · {s['id']} {s['label']}")
        result = run_agent(s["message"])
        out.append({**s, "result": result})
        # Push to archive
        ticket_id = f"T-{len(archive_existing['tickets'])+1:04d}"
        archive_existing["tickets"].append({
            "ticket_id": ticket_id,
            "scenario_id": s["id"],
            "customer_message": s["message"],
            "status": result["status"],
            "turns": result["turns"],
            "archive": result["archive"],
            "escalation": result["escalation"],
        })

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provider": active_provider(),
        "model": active_model_id(),
        "scenarios": out,
    }
    save_cached(SCENARIOS_PATH, payload)
    save_cached(ARCHIVE_PATH, archive_existing)
    print(f"[agent] saved {SCENARIOS_PATH} and {ARCHIVE_PATH}")
    return payload


def append_to_archive(result: dict) -> str:
    """Append a live conversation outcome to the archive. Returns ticket id."""
    ensure_dirs()
    archive = load_cached(ARCHIVE_PATH) or {"tickets": []}
    ticket_id = f"T-{len(archive['tickets'])+1:04d}"
    archive["tickets"].append({
        "ticket_id": ticket_id,
        "scenario_id": "LIVE",
        "customer_message": result["customer_message"],
        "status": result["status"],
        "turns": result["turns"],
        "archive": result["archive"],
        "escalation": result["escalation"],
    })
    save_cached(ARCHIVE_PATH, archive)
    return ticket_id


def stream_agent(customer_message: str, max_turns: int = MAX_TURNS):
    """Generator version of run_agent — yields events as they happen.

    Yields dicts of shape {type: str, payload: dict}. Event types:
      - agent      : agent text reply for this turn
      - tool       : a single tool call + its result
      - _final     : the structured result (same shape as run_agent return)
    """
    client = get_anthropic_client()
    model = active_model_id()
    crm = deepcopy(_load_crm())
    audit: list[dict] = []
    transcript: list[dict] = [{"role": "customer", "text": customer_message}]
    messages = [{"role": "user", "content": customer_message}]
    final_status = "ongoing"
    archive_record: dict | None = None
    escalation_record: dict | None = None
    turn = 0

    for turn in range(max_turns):
        resp = client.messages.create(
            model=model, max_tokens=900, system=SYSTEM_PROMPT,
            tools=TOOLS, messages=messages,
        )

        text_parts = [b.text for b in resp.content if b.type == "text"]
        if text_parts:
            text = "\n".join(text_parts)
            transcript.append({"role": "agent", "text": text})
            yield {"type": "agent", "payload": {"text": text}}

        if resp.stop_reason != "tool_use":
            final_status = "ended_no_action"
            break

        messages.append({"role": "assistant",
                         "content": [b.model_dump() for b in resp.content]})

        tool_results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            result = _execute(block.name, block.input, crm, audit)
            transcript.append({"role": "tool", "tool": block.name,
                               "input": block.input, "result": result})
            yield {"type": "tool", "payload": {
                "tool": block.name, "input": block.input, "result": result,
            }}
            tool_results.append({"type": "tool_result",
                                 "tool_use_id": block.id,
                                 "content": json.dumps(result)})
            if block.name == "archive_conversation" and result.get("status") == "archived":
                archive_record = block.input
                final_status = "resolved"
            elif block.name == "escalate_to_human" and result.get("status") == "escalated":
                escalation_record = block.input
                final_status = "escalated"

        messages.append({"role": "user", "content": tool_results})

        if final_status in ("resolved", "escalated"):
            try:
                closer = client.messages.create(
                    model=model, max_tokens=200, system=SYSTEM_PROMPT,
                    tools=TOOLS, messages=messages,
                )
                tail = [b.text for b in closer.content if b.type == "text"]
                if tail:
                    text = "\n".join(tail)
                    transcript.append({"role": "agent", "text": text})
                    yield {"type": "agent", "payload": {"text": text}}
            except Exception:  # noqa: BLE001
                pass
            break

    final = {
        "model": model,
        "provider": active_provider(),
        "status": final_status,
        "turns": turn + 1,
        "customer_message": customer_message,
        "transcript": transcript,
        "audit": audit,
        "archive": archive_record,
        "escalation": escalation_record,
    }
    yield {"type": "_final", "payload": final}


if __name__ == "__main__":
    precompute_scenarios()
