"""input_guardrails.py -- Input guardrails for support ticket processing.

Input Guardrails (3 marks):
Before any model call:
(a) A ticket whose text is empty or only whitespace gets status "skipped_empty"
    and is never sent to the model.
(b) A ticket whose est_tokens(text) is greater than MAX_INPUT_TOKENS = 400
    gets status "flagged_long" and is never sent to the model.

Reference:
- 4 empty tickets (status: skipped_empty)
- 4 long tickets (status: flagged_long, estimating to 734 tokens)
- 156 valid tickets that proceed to the model
"""

import math
from typing import Dict, List, Optional, Tuple
import pandas as pd
from mock_llm import est_tokens

MAX_INPUT_TOKENS = 400
RAW_TICKETS_PATH = "data/raw/tickets.csv"


def check_input_guardrail(text: Optional[str]) -> Tuple[bool, str, int]:
    """Evaluates input guardrails on ticket text prior to any LLM invocation.

    Rules:
    1. Empty or whitespace-only text -> status='skipped_empty', can_proceed=False.
    2. Estimated token count > MAX_INPUT_TOKENS (400) -> status='flagged_long', can_proceed=False.
    3. Within limit -> status='passed', can_proceed=True.

    Returns:
        (can_proceed, status, token_count)
    """
    if text is None or not str(text).strip():
        return False, "skipped_empty", 0

    tokens = est_tokens(str(text))
    if tokens > MAX_INPUT_TOKENS:
        return False, "flagged_long", tokens

    return True, "passed", tokens


def apply_guardrails_to_dataset(csv_path: str = RAW_TICKETS_PATH) -> pd.DataFrame:
    """Loads tickets dataset, applies guardrails, and records audit status."""
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

    results = []
    for _, row in df.iterrows():
        tid = row["ticket_id"]
        cid = row["customer_id"]
        oid = row["order_id"]
        txt = row["text"]
        cat = row["true_category"]

        can_proceed, status, tokens = check_input_guardrail(txt)

        results.append({
            "ticket_id": tid,
            "customer_id": cid,
            "order_id": oid,
            "text": txt,
            "true_category": cat,
            "estimated_tokens": tokens,
            "guardrail_status": status,
            "can_call_model": can_proceed,
        })

    return pd.DataFrame(results)


def main():
    print("=" * 80)
    print("  INPUT GUARDRAILS AUDIT")
    print("=" * 80)
    print(f"Max Allowed Input Tokens: {MAX_INPUT_TOKENS}")

    df_guardrailed = apply_guardrails_to_dataset(RAW_TICKETS_PATH)

    total_tickets = len(df_guardrailed)
    status_counts = df_guardrailed["guardrail_status"].value_counts().to_dict()

    empty_df = df_guardrailed[df_guardrailed["guardrail_status"] == "skipped_empty"]
    long_df = df_guardrailed[df_guardrailed["guardrail_status"] == "flagged_long"]
    passed_df = df_guardrailed[df_guardrailed["guardrail_status"] == "passed"]

    print(f"\nTotal Tickets Evaluated: {total_tickets}")
    print(f"  - Passed Guardrails (sent to model):    {len(passed_df)}")
    print(f"  - Skipped (Empty/Whitespace):           {len(empty_df)}")
    print(f"  - Flagged Long (> {MAX_INPUT_TOKENS} tokens):           {len(long_df)}")

    # Display empty tickets
    print("\n--- (a) Empty Tickets (status: skipped_empty) ---")
    for _, row in empty_df.iterrows():
        print(f"  Ticket {row['ticket_id']}: customer_id={row['customer_id']}, text='{row['text']}' -> status={row['guardrail_status']}")

    # Display long tickets
    print("\n--- (b) Long Tickets (status: flagged_long) ---")
    for _, row in long_df.iterrows():
        print(f"  Ticket {row['ticket_id']}: est_tokens={row['estimated_tokens']} (> {MAX_INPUT_TOKENS}) -> status={row['guardrail_status']}")

    # Verification assertions against reference benchmarks
    assert len(empty_df) == 4, f"Expected exactly 4 empty tickets, found {len(empty_df)}"
    assert len(long_df) == 4, f"Expected exactly 4 long tickets, found {len(long_df)}"
    assert len(passed_df) == 156, f"Expected exactly 156 passed tickets, found {len(passed_df)}"

    for _, row in long_df.iterrows():
        assert row["estimated_tokens"] == 734, (
            f"Expected long ticket to estimate to 734 tokens, got {row['estimated_tokens']}"
        )

    print("\n" + "=" * 80)
    print("  ALL GUARDRAIL VERIFICATIONS PASSED")
    print("=" * 80)
    print("Reference check:")
    print("  - 4 empty tickets: verified (status='skipped_empty', can_call_model=False)")
    print("  - 4 long tickets:  verified (status='flagged_long', est_tokens=734, can_call_model=False)")
    print("  - Neither empty nor long tickets are ever sent to the model.")
    print("=" * 80)


if __name__ == "__main__":
    main()
