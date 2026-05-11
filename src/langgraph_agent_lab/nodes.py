"""Node skeletons for the LangGraph workflow.

Each function should be small, testable, and return a partial state update. Avoid mutating the
input state in place.
"""

from __future__ import annotations

import re

from .state import AgentState, ApprovalDecision, Route, make_event


def _normalize_query(raw: str) -> str:
    return " ".join(raw.strip().split())


def _tokenize(query: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", query.lower())


def _detect_pii(query: str) -> bool:
    if re.search(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", query):
        return True
    if re.search(r"\b\d{10,}\b", query):
        return True
    return False


def intake_node(state: AgentState) -> dict:
    """Normalize raw query into state fields.
    """
    query = _normalize_query(state.get("query", ""))
    pii_detected = _detect_pii(query)
    word_count = len(query.split())
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [
            make_event(
                "intake",
                "completed",
                "query normalized",
                pii_detected=pii_detected,
                word_count=word_count,
            )
        ],
    }


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route.
    Required routes: simple, tool, missing_info, risky, error.
    """
    query = state.get("query", "")
    tokens = _tokenize(query)
    token_set = set(tokens)
    risky_keywords = {"refund", "delete", "send", "cancel", "remove", "revoke"}
    tool_keywords = {"status", "order", "lookup",
                     "check", "track", "find", "search"}
    error_keywords = {"timeout", "fail",
                      "failure", "error", "crash", "unavailable"}
    missing_pronouns = {"it", "this", "that", "they",
                        "them", "he", "she", "these", "those"}
    route = Route.SIMPLE
    risk_level = "low"
    matched_keyword = ""
    if token_set & risky_keywords:
        route = Route.RISKY
        risk_level = "high"
        matched_keyword = sorted(token_set & risky_keywords)[0]
    elif token_set & tool_keywords:
        route = Route.TOOL
        matched_keyword = sorted(token_set & tool_keywords)[0]
    elif len(tokens) < 5 and token_set & missing_pronouns:
        route = Route.MISSING_INFO
        matched_keyword = sorted(token_set & missing_pronouns)[0]
    elif token_set & error_keywords:
        route = Route.ERROR
        matched_keyword = sorted(token_set & error_keywords)[0]
    return {
        "route": route.value,
        "risk_level": risk_level,
        "events": [
            make_event(
                "classify",
                "completed",
                f"route={route.value}",
                matched_keyword=matched_keyword,
            )
        ],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating.
    """
    query = state.get("query", "")
    token_set = set(_tokenize(query))
    if {"order", "status", "lookup", "track"} & token_set:
        question = "Please share the order id or tracking number so I can look this up."
    elif {"refund", "delete", "cancel", "remove", "revoke"} & token_set:
        question = "Please confirm the customer identifier and the exact action to proceed."
    else:
        question = "Can you share more details about the issue so I can help?"
    return {
        "pending_question": question,
        "final_answer": question,
        "events": [make_event("clarify", "completed", "missing information requested")],
    }


def tool_node(state: AgentState) -> dict:
    """Call a mock tool.

    Simulates transient failures for error-route scenarios to demonstrate retry loops.
    """
    attempt = int(state.get("attempt", 0))
    existing = state.get("tool_results", []) or []
    if existing and f"attempt={attempt}" in existing[-1]:
        return {
            "events": [make_event("tool", "skipped", "tool already executed", attempt=attempt)],
        }
    scenario_id = state.get("scenario_id", "unknown")
    should_retry = bool(state.get("should_retry"))
    route = state.get("route")
    if route == Route.ERROR.value and attempt < 2:
        status = "error"
        detail = "transient_failure"
    elif should_retry and attempt == 0:
        status = "error"
        detail = "simulated_retry"
    else:
        status = "ok"
        detail = "success"
    result = f"status={status}; detail={detail}; attempt={attempt}; scenario={scenario_id}"
    return {
        "tool_results": [result],
        "events": [make_event("tool", "completed", f"tool executed attempt={attempt}")],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for approval.
    """
    query = state.get("query", "")
    risk_level = state.get("risk_level", "unknown")
    proposed_action = f"Propose action based on request: '{query}'. Risk level: {risk_level}."
    return {
        "proposed_action": proposed_action,
        "events": [make_event("risky_action", "pending_approval", "approval required")],
    }


def approval_node(state: AgentState) -> dict:
    """Human approval step with optional LangGraph interrupt().

    Set LANGGRAPH_INTERRUPT=true to use real interrupt() for HITL demos.
    Default uses mock decision so tests and CI run offline.
    """
    import os

    if os.getenv("LANGGRAPH_INTERRUPT", "").lower() == "true":
        from langgraph.types import interrupt

        value = interrupt({
            "proposed_action": state.get("proposed_action"),
            "risk_level": state.get("risk_level"),
        })
        if value is None:
            decision = ApprovalDecision(
                approved=False, decision="timeout", comment="no response")
        elif isinstance(value, dict):
            decision_value = value.get("decision")
            approved_value = value.get("approved")
            if approved_value is None and decision_value:
                approved_value = decision_value == "approved"
            decision = ApprovalDecision(
                approved=bool(approved_value),
                decision=decision_value,
                reviewer=value.get("reviewer", "human"),
                comment=value.get("comment", ""),
            )
        else:
            decision = ApprovalDecision(
                approved=bool(value),
                decision="approved" if value else "rejected",
            )
    else:
        override = os.getenv("LAB_APPROVAL_DECISION", "").strip().lower()
        if override in {"reject", "rejected"}:
            decision = ApprovalDecision(
                approved=False, decision="rejected", comment="mock rejection")
        elif override in {"edit", "edited"}:
            decision = ApprovalDecision(
                approved=False, decision="edit", comment="mock edit request")
        else:
            decision = ApprovalDecision(
                approved=True, decision="approved", comment="mock approval for lab")
    return {
        "approval": decision.model_dump(),
        "events": [make_event("approval", "completed", f"approved={decision.approved}")],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt or fallback decision.
    """
    attempt = int(state.get("attempt", 0)) + 1
    backoff_ms = min(250 * (2 ** (attempt - 1)), 4000)
    errors = [f"transient failure attempt={attempt}"]
    return {
        "attempt": attempt,
        "errors": errors,
        "events": [
            make_event(
                "retry",
                "completed",
                "retry attempt recorded",
                attempt=attempt,
                backoff_ms=backoff_ms,
            )
        ],
    }


def answer_node(state: AgentState) -> dict:
    """Produce a final response.
    """
    tool_results = state.get("tool_results", []) or []
    approval = state.get("approval") or {}
    if tool_results:
        answer = f"Tool result: {tool_results[-1]}"
    else:
        answer = "Thanks for reaching out. I can help with this request."
    if approval:
        decision = approval.get("decision") or (
            "approved" if approval.get("approved") else "rejected")
        answer = f"{answer} Approval decision: {decision}."
    return {
        "final_answer": answer,
        "events": [make_event("answer", "completed", "answer generated")],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate tool results — the 'done?' check that enables retry loops.
    """
    tool_results = state.get("tool_results", [])
    latest = tool_results[-1] if tool_results else ""
    if not latest:
        return {
            "evaluation_result": "needs_retry",
            "events": [make_event("evaluate", "completed", "no tool result, retry needed")],
        }
    if "status=error" in latest or "ERROR" in latest:
        return {
            "evaluation_result": "needs_retry",
            "events": [make_event("evaluate", "completed", "tool result indicates failure, retry needed")],
        }
    return {
        "evaluation_result": "success",
        "events": [make_event("evaluate", "completed", "tool result satisfactory")],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Log unresolvable failures for manual review.

    Third layer of error strategy: retry -> fallback -> dead letter.
    """
    errors = state.get("errors", []) or []
    summary = errors[-1] if errors else "no error details"
    return {
        "final_answer": "Request could not be completed after maximum retry attempts. Logged for manual review.",
        "events": [
            make_event(
                "dead_letter",
                "completed",
                f"max retries exceeded, attempt={state.get('attempt', 0)}",
                last_error=summary,
            )
        ],
    }


def finalize_node(state: AgentState) -> dict:
    """Finalize the run and emit a final audit event."""
    return {"events": [make_event("finalize", "completed", "workflow finished")]}
