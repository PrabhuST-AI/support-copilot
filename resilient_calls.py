"""resilient_calls.py -- Resilient model calls with exponential backoff and jitter.

Tasks (4 marks):
Wrap every model call in a retry function with:
- Maximum of 3 retries per ticket.
- Exponential backoff with jitter:
    wait before retry number n (n = 0, 1, 2) is:
    BASE_DELAY * (2 ** n) * random.uniform(0.9, 1.15) with BASE_DELAY = 0.05 seconds.
- One log line per retry showing the ticket id, the retry number, and the wait time.
- Call mock_llm.reset() immediately before each full run.

Reference:
- A full run over the 156 non-empty, non-long tickets produces exactly 19 TransientLLMErrors.
- All 19 are recovered by a retry; no ticket ends with a permanent error.
"""

import logging
import random
import time
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from input_guardrails import check_input_guardrail
from mock_llm import TransientLLMError, chat, reset
from prompts import FEW_SHOT_PROMPT, ZERO_SHOT_PROMPT
from redact_pii import redact_text

BASE_DELAY = 0.05
MAX_RETRIES = 3
RAW_TICKETS_PATH = "data/raw/tickets.csv"

# Configure logger if needed, but stdout print / logging is available
logger = logging.getLogger("resilient_calls")


def call_with_retry(
    messages: List[Dict[str, str]],
    ticket_id: str,
    max_retries: int = MAX_RETRIES,
    base_delay: float = BASE_DELAY,
    tools: Optional[Any] = None,
) -> Any:
    """Invokes mock_llm.chat with exponential backoff and jitter upon TransientLLMError.

    Parameters:
        messages: Chat conversation message payload.
        ticket_id: Unique identifier for logging.
        max_retries: Maximum number of retries before giving up (default 3).
        base_delay: Base delay in seconds (default 0.05s).
        tools: Optional tool definitions.

    Returns:
        Chat completion response object.
    """
    for n in range(max_retries):
        try:
            return chat(messages, tools=tools)
        except TransientLLMError as err:
            wait = base_delay * (2 ** n) * random.uniform(0.9, 1.15)
            print(f"[RETRY] Ticket ID: {ticket_id} | Retry: {n} | Wait: {wait:.4f}s | Error: {err}")
            time.sleep(wait)

    # Final attempt after retries exhausted
    return chat(messages, tools=tools)


def run_resilient_batch(
    system_prompt: str = ZERO_SHOT_PROMPT,
    tickets_path: str = RAW_TICKETS_PATH,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Runs a full classification pass over non-empty, non-long tickets using resilient calls."""
    df = pd.read_csv(tickets_path, dtype=str, keep_default_na=False)

    # 1. Filter out empty and long tickets using input guardrails
    valid_tickets = []
    skipped_tickets = []
    for _, row in df.iterrows():
        can_proceed, status, tokens = check_input_guardrail(row["text"])
        if can_proceed:
            valid_tickets.append(row)
        else:
            skipped_tickets.append((row["ticket_id"], status))

    print(f"Total tickets in dataset:   {len(df)}")
    print(f"Skipped by guardrails:      {len(skipped_tickets)} (4 empty, 4 long)")
    print(f"Valid tickets to process:   {len(valid_tickets)}")

    # 2. Reset mock LLM call counters immediately before the full run
    reset()
    print("Called mock_llm.reset() successfully.")

    # 3. Process tickets with resilient retry logic
    results = []
    transient_errors_caught = 0
    permanent_errors = 0

    for row in valid_tickets:
        tid = row["ticket_id"]
        cid = row["customer_id"]
        oid = row["order_id"]
        raw_text = row["text"]
        true_cat = row["true_category"]

        # Redact PII (user message must be exactly the redacted ticket text)
        redacted_text, _ = redact_text(raw_text)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": redacted_text},
        ]

        # Call with retry wrapper and error counting
        response = None
        retries_needed = 0
        for n in range(MAX_RETRIES):
            try:
                response = chat(messages)
                break
            except TransientLLMError as err:
                transient_errors_caught += 1
                retries_needed += 1
                wait = BASE_DELAY * (2 ** n) * random.uniform(0.9, 1.15)
                print(f"[RETRY LOG] ticket_id={tid:<6} | retry_number={n} | wait={wait:.4f}s | reason='{err}'")
                time.sleep(wait)

        if response is None:
            try:
                response = chat(messages)
            except Exception as final_err:
                permanent_errors += 1
                print(f"[PERMANENT ERROR] Ticket {tid} failed after max retries: {final_err}")

        results.append({
            "ticket_id": tid,
            "customer_id": cid,
            "order_id": oid,
            "redacted_text": redacted_text,
            "true_category": true_cat,
            "retries_needed": retries_needed,
            "raw_response": response.choices[0].message.content if response else None,
            "succeeded": response is not None,
        })

    stats = {
        "total_valid_tickets": len(valid_tickets),
        "transient_errors_caught": transient_errors_caught,
        "permanent_errors": permanent_errors,
        "recovery_rate": (len(valid_tickets) - permanent_errors) / len(valid_tickets) if valid_tickets else 0.0,
    }

    return results, stats


def main():
    print("=" * 80)
    print("  RESILIENT MODEL CALLS EXECUTION AUDIT")
    print("=" * 80)
    print(f"Max retries per ticket: 3")
    print(f"Base delay: {BASE_DELAY}s (exponential backoff with jitter in [0.9, 1.15])")
    print("=" * 80)

    results, stats = run_resilient_batch()

    print("\n" + "=" * 80)
    print("  RUN SUMMARY & REFERENCE VERIFICATION")
    print("=" * 80)
    print(f"Valid Tickets Evaluated:         {stats['total_valid_tickets']}")
    print(f"TransientLLMErrors Encountered:  {stats['transient_errors_caught']}")
    print(f"Permanent Errors:                {stats['permanent_errors']}")
    print(f"Final Success Rate:              {stats['recovery_rate']*100:.2f}%")

    # Assert reference benchmarks
    assert stats["total_valid_tickets"] == 156, (
        f"Expected 156 valid tickets, got {stats['total_valid_tickets']}"
    )
    assert stats["transient_errors_caught"] == 19, (
        f"Expected exactly 19 TransientLLMErrors, got {stats['transient_errors_caught']}"
    )
    assert stats["permanent_errors"] == 0, (
        f"Expected 0 permanent errors, got {stats['permanent_errors']}"
    )

    print("\n--> CHECKPOINT PASSED: Exactly 19 TransientLLMErrors produced and 100% recovered via retries.")
    print("=" * 80)


if __name__ == "__main__":
    main()
