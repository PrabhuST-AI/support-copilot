"""triage_routing.py -- Support ticket triage, token budgeting, and priority routing.

Tasks (3 marks):
(a) Add a TOKEN_BUDGET:
    - Before each model call, if running total of usage.total_tokens >= budget,
      make no call and set status="budget_exhausted" (route to human review).
    - Run zero-shot pipeline with TOKEN_BUDGET = 6000.
    - Save result as outputs/triage_budget_demo.csv.
    - Report how many tickets end budget_exhausted.
(b) Produce final outputs/triage_results.csv from a few-shot run with TOKEN_BUDGET = 100000:
    - Reads outputs/customer_segments.csv from Part B.
    - Routing logic:
      * status in {"skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"}
        -> route: human_review, priority: NORMAL
      * otherwise if category == "fraud"
        -> route: security_team, priority: HIGH
      * otherwise if customer's is_premium is True
        -> route: premium_desk, priority: HIGH
      * otherwise
        -> route: support_queue, priority: NORMAL
      (A customer with no row in segment file is treated as not premium).
    - Columns: ticket_id, customer_id, status, category, segment_name, priority, route, tokens_used
      (category and segment_name left blank when not available).
"""

import os
from typing import Any, Dict, List, Tuple
import pandas as pd

from input_guardrails import check_input_guardrail
from mock_llm import reset
from prompts import FEW_SHOT_PROMPT, ZERO_SHOT_PROMPT
from redact_pii import redact_text
from resilient_calls import call_with_retry
from validate_answers import validate_and_classify_reply

RAW_TICKETS_PATH = "data/raw/tickets.csv"
SEGMENTS_PATH = "outputs/customer_segments.csv"
DEMO_OUTPUT_PATH = "outputs/triage_budget_demo.csv"
RESULTS_OUTPUT_PATH = "outputs/triage_results.csv"

OUTPUT_COLUMNS = [
    "ticket_id",
    "customer_id",
    "status",
    "category",
    "segment_name",
    "priority",
    "route",
    "tokens_used",
]


def load_customer_segments(segments_path: str = SEGMENTS_PATH) -> Dict[str, Dict[str, Any]]:
    """Loads customer segmentation table and returns mapping by customer_id."""
    if not os.path.exists(segments_path):
        return {}
    df_seg = pd.read_csv(segments_path)
    return df_seg.set_index("customer_id").to_dict("index")


def execute_triage_pipeline(
    prompt: str,
    token_budget: int,
    tickets_path: str = RAW_TICKETS_PATH,
    segments_path: str = SEGMENTS_PATH,
) -> Tuple[pd.DataFrame, int, Dict[str, int]]:
    """Runs complete triage pipeline with token budgeting and segment-aware routing."""
    df_tickets = pd.read_csv(tickets_path, dtype=str, keep_default_na=False)
    segment_map = load_customer_segments(segments_path)

    # Reset mock LLM call counter immediately before each run
    reset()

    running_tokens = 0
    records = []

    for _, row in df_tickets.iterrows():
        tid = row["ticket_id"]
        cid = row["customer_id"]
        raw_text = row["text"]

        # Segment lookup (if missing, customer is not premium and segment_name is blank)
        seg_info = segment_map.get(cid, {})
        is_premium = bool(seg_info.get("is_premium", False))
        segment_name = seg_info.get("segment_name", "")

        # 1. Input Guardrails
        can_proceed, guard_status, _ = check_input_guardrail(raw_text)
        if not can_proceed:
            records.append({
                "ticket_id": tid,
                "customer_id": cid,
                "status": guard_status,
                "category": "",
                "segment_name": segment_name if segment_name else "",
                "priority": "NORMAL",
                "route": "human_review",
                "tokens_used": 0,
            })
            continue

        # 2. Token Budget check
        if running_tokens >= token_budget:
            records.append({
                "ticket_id": tid,
                "customer_id": cid,
                "status": "budget_exhausted",
                "category": "",
                "segment_name": segment_name if segment_name else "",
                "priority": "NORMAL",
                "route": "human_review",
                "tokens_used": 0,
            })
            continue

        # 3. Model call with retry and PII redaction
        redacted_text, _ = redact_text(raw_text)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": redacted_text},
        ]

        resp = call_with_retry(messages, ticket_id=tid)
        tokens_this_call = resp.usage.total_tokens
        running_tokens += tokens_this_call

        # 4. Validate output
        reply = resp.choices[0].message.content
        status, pred_cat, _ = validate_and_classify_reply(reply)

        # 5. Routing logic
        if status in ("skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"):
            route = "human_review"
            priority = "NORMAL"
            category_val = ""
        elif pred_cat == "fraud":
            route = "security_team"
            priority = "HIGH"
            category_val = pred_cat
        elif is_premium:
            route = "premium_desk"
            priority = "HIGH"
            category_val = pred_cat
        else:
            route = "support_queue"
            priority = "NORMAL"
            category_val = pred_cat

        records.append({
            "ticket_id": tid,
            "customer_id": cid,
            "status": status,
            "category": category_val if category_val else "",
            "segment_name": segment_name if segment_name else "",
            "priority": priority,
            "route": route,
            "tokens_used": tokens_this_call,
        })

    df_out = pd.DataFrame(records)[OUTPUT_COLUMNS]
    status_summary = df_out["status"].value_counts().to_dict()

    return df_out, running_tokens, status_summary


def main():
    os.makedirs("outputs", exist_ok=True)

    # =========================================================================
    # PART (A): ZERO-SHOT WITH TOKEN_BUDGET = 6000
    # =========================================================================
    print("=" * 80)
    print("  PART (A): TOKEN BUDGET DEMO RUN (Zero-Shot, TOKEN_BUDGET = 6000)")
    print("=" * 80)

    df_demo, tokens_demo, status_demo = execute_triage_pipeline(
        prompt=ZERO_SHOT_PROMPT,
        token_budget=6000,
    )

    df_demo.to_csv(DEMO_OUTPUT_PATH, index=False)
    print(f"Saved demo triage results to: {DEMO_OUTPUT_PATH}")
    print(f"Total tickets evaluated: {len(df_demo)}")
    print(f"Tokens consumed before cutoff: {tokens_demo}")

    print("\nStatus distribution in budget demo run:")
    for status, cnt in status_demo.items():
        print(f"  - {status:<24}: {cnt:>3} tickets")

    n_exhausted = status_demo.get("budget_exhausted", 0)
    print(f"\n--> Number of tickets with status 'budget_exhausted': {n_exhausted}")
    assert n_exhausted >= 1, f"At least one ticket must end budget_exhausted, got {n_exhausted}"

    # =========================================================================
    # PART (B): FEW-SHOT FINAL RUN WITH TOKEN_BUDGET = 100000
    # =========================================================================
    print("\n" + "=" * 80)
    print("  PART (B): FINAL TRIAGE RUN (Few-Shot, TOKEN_BUDGET = 100000)")
    print("=" * 80)

    df_results, tokens_final, status_final = execute_triage_pipeline(
        prompt=FEW_SHOT_PROMPT,
        token_budget=100000,
    )

    df_results.to_csv(RESULTS_OUTPUT_PATH, index=False)
    print(f"Saved final triage results to: {RESULTS_OUTPUT_PATH}")
    print(f"Total tickets evaluated: {len(df_results)}")
    print(f"Total tokens consumed: {tokens_final}")

    print("\nFinal Run Status Counts:")
    for status, cnt in status_final.items():
        print(f"  - {status:<24}: {cnt:>3} tickets")

    print("\nFinal Run Routing Destinations:")
    route_counts = df_results["route"].value_counts().to_dict()
    for route, cnt in route_counts.items():
        print(f"  - {route:<24}: {cnt:>3} tickets")

    print("\nFinal Run Priority Distribution:")
    priority_counts = df_results["priority"].value_counts().to_dict()
    for prio, cnt in priority_counts.items():
        print(f"  - {prio:<24}: {cnt:>3} tickets")

    # Verification assertions
    assert len(df_results) == 164, f"Expected 164 tickets, got {len(df_results)}"
    assert status_final.get("budget_exhausted", 0) == 0, "No tickets should be budget exhausted with 100k budget"
    assert status_final.get("classified_json", 0) == 135
    assert status_final.get("classified_keyword", 0) == 12
    assert status_final.get("flagged_invalid_output", 0) == 9
    assert status_final.get("skipped_empty", 0) == 4
    assert status_final.get("flagged_long", 0) == 4

    print("\n" + "=" * 80)
    print("  ALL TRIAGE CHECKPOINTS VERIFIED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
