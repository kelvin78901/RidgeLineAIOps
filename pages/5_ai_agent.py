"""AI Service Agent — bounded-autonomy customer-service demo."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.utils import (  # noqa: E402
    DATA_DIR, RESULTS_DIR, active_model_id, active_provider,
    apply_page_chrome, empty_state, load_cached, plotly_layout, tag,
)

apply_page_chrome(
    "AI Service Agent",
    breadcrumb="Module 5 · Bounded autonomy",
    subtitle="An LLM agent handles customer requests using a scoped tool "
             "catalog. Policy caps are enforced server-side — out-of-scope "
             "asks are escalated, in-scope work is archived for compliance.",
)

scenarios = load_cached(RESULTS_DIR / "agent_scenarios.json")
archive = load_cached(RESULTS_DIR / "agent_archive.json") or {"tickets": []}
crm = load_cached(DATA_DIR / "mock_crm.json")

if not scenarios:
    st.markdown(empty_state(
        "Agent scenarios not yet cached.",
        'python -c "from core.agent_simulator import precompute_scenarios; precompute_scenarios()"',
    ), unsafe_allow_html=True)
    st.stop()


# ---------------- KPIs ----------------

scen_list = scenarios["scenarios"]
total = len(scen_list)
resolved = sum(1 for s in scen_list if s["result"]["status"] == "resolved")
escalated = sum(1 for s in scen_list if s["result"]["status"] == "escalated")
waiting = sum(1 for s in scen_list if s["result"]["status"] == "ended_no_action")
auto_pct = resolved / total if total else 0.0
avg_turns = sum(s["result"]["turns"] for s in scen_list) / total if total else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Scenarios run", total)
c2.metric("Auto-resolved", resolved, f"{auto_pct:.0%}")
c3.metric("Escalated to human", escalated)
c4.metric("Awaiting customer", waiting)
c5.metric("Avg turns", f"{avg_turns:.1f}")

st.markdown(
    f'<div class="rg-row" style="margin:8px 0 16px 0">'
    f'{tag("Provider · " + scenarios.get("provider", active_provider()), "blue")}'
    f'{tag("Model · " + scenarios.get("model", active_model_id()), "violet")}'
    f'{tag("Refund cap $500", "slate")}'
    f'{tag("Discount cap 15%", "slate")}'
    f'{tag("Blocked: account-delete · PII share", "red")}'
    f"</div>",
    unsafe_allow_html=True,
)

st.divider()

# ---------------- Tabs ----------------

tab_play, tab_archive, tab_policy, tab_crm = st.tabs(
    ["Conversation trace", "Archive & escalations", "Policy", "Mock CRM"]
)


def _status_tag(status: str) -> str:
    return {
        "resolved":        tag("RESOLVED", "green"),
        "escalated":       tag("ESCALATED", "red"),
        "ended_no_action": tag("AWAITING CUSTOMER", "amber"),
        "ongoing":         tag("ONGOING", "blue"),
    }.get(status, tag(status.upper(), "slate"))


def _render_transcript(result: dict) -> None:
    """Render the full transcript with tool-call expanders."""
    import html as _html

    for entry in result["transcript"]:
        if entry["role"] == "customer":
            body = _html.escape(entry["text"])
            st.markdown(
                f'<div class="rg-bubble customer">'
                f'<div class="who">Customer</div>'
                f'<div class="body">{body}</div></div>',
                unsafe_allow_html=True,
            )
        elif entry["role"] == "agent":
            body = _html.escape(entry["text"])
            st.markdown(
                f'<div class="rg-bubble">'
                f'<div class="who">Agent</div>'
                f'<div class="body">{body}</div></div>',
                unsafe_allow_html=True,
            )
        elif entry["role"] == "tool":
            res = entry["result"]
            ok = res.get("status") != "permission_denied"
            sev_tag = tag("OK", "green") if ok else tag("DENIED", "red")
            with st.expander(f"tool · {entry['tool']}", expanded=not ok):
                st.markdown(sev_tag, unsafe_allow_html=True)
                cl, cr = st.columns(2)
                cl.markdown('<div class="rg-muted" style="font-size:11px;'
                            'text-transform:uppercase;letter-spacing:.06em;'
                            'font-weight:600">Arguments</div>',
                            unsafe_allow_html=True)
                cl.code(json.dumps(entry["input"], indent=2), language="json")
                cr.markdown('<div class="rg-muted" style="font-size:11px;'
                            'text-transform:uppercase;letter-spacing:.06em;'
                            'font-weight:600">Result</div>',
                            unsafe_allow_html=True)
                cr.code(json.dumps(res, indent=2, default=str), language="json")


# ============ Tab 1 — Conversation trace ============

with tab_play:
    lc, rc = st.columns([1, 2], gap="large")

    with lc:
        st.markdown('<div class="rg-section-title">Preset scenarios</div>',
                    unsafe_allow_html=True)
        if "selected_scenario" not in st.session_state:
            st.session_state.selected_scenario = scen_list[0]["id"]

        for s in scen_list:
            status = s["result"]["status"]
            label = f"{s['id']} — {s['label']}"
            picked = (st.session_state.selected_scenario == s["id"])
            border = "#4f46e5" if picked else "#e2e8f0"
            shadow = "0 0 0 2px #dbeafe" if picked else "0 1px 2px rgba(15,23,42,0.04)"
            cols = st.columns([5, 1])
            with cols[0]:
                if st.button(label, key=f"pick-{s['id']}", width="stretch"):
                    st.session_state.selected_scenario = s["id"]
            with cols[1]:
                st.markdown(_status_tag(status), unsafe_allow_html=True)

        st.markdown('<div class="rg-section-title" style="margin-top:18px">'
                    'Try your own message</div>', unsafe_allow_html=True)
        live_msg = st.text_area(
            "Customer message",
            placeholder="e.g. I need help with order ORD-3344, eve@example.com",
            height=110,
        )
        live = st.button("Run agent live", type="primary", width="stretch",
                         disabled=not live_msg.strip())

    with rc:
        if live and live_msg.strip():
            try:
                from core.agent_simulator import append_to_archive, run_agent
                with st.spinner("Running agent..."):
                    result = run_agent(live_msg.strip())
                ticket_id = append_to_archive(result)
                st.session_state["live_result"] = result
                st.session_state["live_ticket"] = ticket_id
            except Exception as e:  # noqa: BLE001
                st.error(f"Live run failed: {e}")

        if "live_result" in st.session_state and live_msg:
            res = st.session_state["live_result"]
            tid = st.session_state["live_ticket"]
            turns_str = f"{res['turns']} turns"
            st.markdown(
                f'<div class="rg-card">'
                f'<div class="rg-row" style="justify-content:space-between">'
                f'<div><b>Live conversation</b> &nbsp; '
                f'<span class="rg-muted">{tid}</span></div>'
                f"<div>{_status_tag(res['status'])} {tag(turns_str, 'blue')}"
                f"</div></div></div>",
                unsafe_allow_html=True,
            )
            _render_transcript(res)

        # Render the currently-selected preset
        active = next(s for s in scen_list if s["id"] == st.session_state.selected_scenario)
        st.markdown(
            f'<div class="rg-card">'
            f'<div class="rg-row" style="justify-content:space-between">'
            f'<div><b>{active["id"]} — {active["label"]}</b></div>'
            f'<div>{_status_tag(active["result"]["status"])} '
            f'{tag(str(active["result"]["turns"]) + " turns", "blue")}</div>'
            f"</div>"
            f'<div class="rg-muted" style="margin-top:6px;font-size:12px">'
            f'<b>Expectation:</b> {active["expectation"]}</div></div>',
            unsafe_allow_html=True,
        )
        _render_transcript(active["result"])


# ============ Tab 2 — Archive & escalations ============

with tab_archive:
    if not archive["tickets"]:
        st.info("Archive is empty — run scenarios first.")
    else:
        df = pd.DataFrame([
            {
                "ticket": t["ticket_id"],
                "source": t.get("scenario_id", "LIVE"),
                "status": t["status"],
                "turns": t["turns"],
                "category": (t.get("archive") or t.get("escalation") or {}).get("category", "-"),
                "summary": ((t.get("archive") or {}).get("summary")
                            or (t.get("escalation") or {}).get("summary")
                            or "-"),
                "priority": (t.get("escalation") or {}).get("priority", ""),
                "customer_message": t["customer_message"][:120],
            }
            for t in archive["tickets"]
        ])

        # KPI strip for archive
        ac1, ac2, ac3 = st.columns(3)
        cat_counts = df["category"].value_counts()
        ac1.metric("Total tickets", len(df))
        ac2.metric("Resolved", int((df["status"] == "resolved").sum()))
        ac3.metric("Escalated", int((df["status"] == "escalated").sum()))

        # Category mix
        st.markdown('<div class="rg-section-title">By category</div>',
                    unsafe_allow_html=True)
        mix = (df.groupby(["category", "status"]).size()
                 .reset_index(name="count"))
        fig = px.bar(
            mix, x="category", y="count", color="status",
            color_discrete_map={"resolved": "#059669", "escalated": "#dc2626",
                                "ended_no_action": "#ea580c"},
            barmode="stack",
        )
        fig.update_layout(**plotly_layout(height=260, showlegend=True))
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom",
                                      y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, width="stretch")

        # Filter row
        fc1, fc2, fc3 = st.columns([1, 1, 2])
        status_pick = fc1.multiselect("Status",
                                      sorted(df["status"].unique()),
                                      default=list(df["status"].unique()))
        cat_pick = fc2.multiselect("Category",
                                   sorted(df["category"].unique()),
                                   default=list(df["category"].unique()))
        search = fc3.text_input("Search", "")
        view = df[df["status"].isin(status_pick) & df["category"].isin(cat_pick)]
        if search:
            view = view[view.apply(
                lambda r: search.lower() in str(r["summary"]).lower()
                          or search.lower() in str(r["customer_message"]).lower(),
                axis=1,
            )]
        st.dataframe(view, width="stretch", hide_index=True, height=380)


# ============ Tab 3 — Policy ============

with tab_policy:
    pol = (crm or {}).get("policies", {})
    st.markdown('<div class="rg-section-title">Hard policy limits (enforced server-side)</div>',
                unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    p1.metric("Max refund", f"${pol.get('max_refund_cents', 0)/100:,.0f}")
    p2.metric("Max discount", f"{pol.get('max_discount_pct', 0)}%")
    p3.metric("Refund window", f"{pol.get('refund_eligibility_days', 0)} days")

    st.markdown('<div class="rg-section-title" style="margin-top:18px">'
                'Blocked intents (escalation required)</div>',
                unsafe_allow_html=True)
    blocked = pol.get("blocked_intents", [])
    st.markdown(
        '<div class="rg-row">' +
        "".join(tag(b, "red") for b in blocked) +
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="rg-section-title" style="margin-top:18px">'
                'Available tools</div>', unsafe_allow_html=True)
    from core.agent_simulator import TOOLS
    tdf = pd.DataFrame([
        {
            "tool": t["name"],
            "writes_data": t["name"] in {
                "issue_refund", "apply_discount_code",
                "update_shipping_address", "escalate_to_human",
                "archive_conversation",
            },
            "description": t["description"],
        }
        for t in TOOLS
    ])
    st.dataframe(tdf, width="stretch", hide_index=True, height=360)


# ============ Tab 4 — Mock CRM ============

with tab_crm:
    if not crm:
        st.info("CRM data missing.")
    else:
        st.markdown('<div class="rg-section-title">Customers</div>',
                    unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(crm["customers"]),
                     width="stretch", hide_index=True, height=220)
        st.markdown('<div class="rg-section-title">Orders</div>',
                    unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(crm["orders"]),
                     width="stretch", hide_index=True, height=260)
        st.markdown('<div class="rg-section-title">Refunds</div>',
                    unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(crm["refunds"]),
                     width="stretch", hide_index=True, height=180)
