"""validate_answers.py -- Validate and parse LLM classification replies.

Tasks (4 marks):
1. Parse the reply as JSON:
   - Accept if data is JSON dict and category is one of the four allowed values
     (billing, delivery, technical, fraud).
   - Status: "classified_json"
2. If not valid JSON:
   - Extract lowercase words with re.findall(r"[a-z]+", text.lower())
     (e.g., "billing-related" -> ["billing", "related"]).
   - Accept if exactly one allowed category word is present among them.
   - Status: "classified_keyword"
3. In every other case:
   - Including valid JSON whose category is outside the allowed list (e.g., "shipping").
   - Non-JSON with 0 or >1 matching categories.
   - Status: "flagged_invalid_output" (routed to human review, never guessed).

Reference (over 156 eligible tickets):
- 135 classified_json
- 12 classified_keyword
- 9 flagged_invalid_output
(A plain whitespace split gives 135 / 0 / 21 because "billing-related" is not split).
"""

import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd

from input_guardrails import check_input_guardrail
from mock_llm import reset
from prompts import ZERO_SHOT_PROMPT
from redact_pii import redact_text
from resilient_calls import call_with_retry

ALLOWED_CATEGORIES: Set[str] = {"billing", "delivery", "technical", "fraud"}
RAW_TICKETS_PATH = "data/raw/tickets.csv"


def validate_and_classify_reply(
    reply_text: Optional[str],
    allowed_categories: Set[str] = ALLOWED_CATEGORIES,
    use_regex_extraction: bool = True,
) -> Tuple[str, Optional[str], Optional[str]]:
    """Validates and parses LLM output per specification.

    Returns:
        (status, predicted_category, reason)
        where status is one of:
            - 'classified_json'
            - 'classified_keyword'
            - 'flagged_invalid_output'
    """
    if not reply_text:
        return "flagged_invalid_output", None, "empty reply"

    clean_text = reply_text.strip()

    # Step 1: Attempt JSON parsing
    try:
        data = json.loads(clean_text)
        if isinstance(data, dict):
            cat = str(data.get("category", "")).strip().lower()
            reason = data.get("reason", "")
            if cat in allowed_categories:
                return "classified_json", cat, reason
            else:
                # Valid JSON, but category outside allowed list (e.g., "shipping")
                return "flagged_invalid_output", None, f"invalid category in json: '{cat}'"
    except (json.JSONDecodeError, TypeError):
        pass

    # Step 2: Fallback keyword extraction for non-JSON replies
    if use_regex_extraction:
        # Extract lowercase alphabetical tokens (splits hyphens like "billing-related" into "billing", "related")
        words = set(re.findall(r"[a-z]+", clean_text.lower()))
    else:
        # Naive whitespace split (fails on hyphenated words)
        words = set(clean_text.lower().split())

    matching_cats = words & allowed_categories

    if len(matching_cats) == 1:
        extracted_cat = list(matching_cats)[0]
        return "classified_keyword", extracted_cat, "extracted from non-json text"

    # Step 3: All other cases -> flagged_invalid_output (route to human review, never guess)
    return "flagged_invalid_output", None, "non-json without single category match"


def run_validation_pipeline(
    system_prompt: str = ZERO_SHOT_PROMPT,
    tickets_path: str = RAW_TICKETS_PATH,
) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, int]]:
    """Executes the full pipeline on 156 eligible tickets and evaluates validation results."""
    df = pd.read_csv(tickets_path, dtype=str, keep_default_na=False)

    # 1. Filter eligible tickets using guardrails
    eligible_rows = [row for _, row in df.iterrows() if check_input_guardrail(row["text"])[0]]
    assert len(eligible_rows) == 156, f"Expected 156 eligible tickets, got {len(eligible_rows)}"

    # 2. Reset mock LLM
    reset()

    records_regex = []
    records_naive = []

    for row in eligible_rows:
        tid = row["ticket_id"]
        cid = row["customer_id"]
        oid = row["order_id"]
        raw_text = row["text"]
        true_cat = row["true_category"]

        redacted_text, _ = redact_text(raw_text)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": redacted_text},
        ]

        resp = call_with_retry(messages, ticket_id=tid)
        reply = resp.choices[0].message.content

        # Validate using regex extraction (standard rule)
        status_regex, pred_cat_regex, reason_regex = validate_and_classify_reply(
            reply, use_regex_extraction=True
        )

        # Validate using naive split (for comparison demonstration)
        status_naive, _, _ = validate_and_classify_reply(
            reply, use_regex_extraction=False
        )

        records_regex.append({
            "ticket_id": tid,
            "customer_id": cid,
            "order_id": oid,
            "raw_reply": reply,
            "status": status_regex,
            "predicted_category": pred_cat_regex,
            "true_category": true_cat,
            "reason": reason_regex,
        })
        records_naive.append(status_naive)

    df_results = pd.DataFrame(records_regex)

    counts_regex = df_results["status"].value_counts().to_dict()
    counts_naive = pd.Series(records_naive).value_counts().to_dict()

    return df_results, counts_regex, counts_naive


def main():
    print("=" * 80)
    print("  VALIDATE EVERY ANSWER AUDIT (156 Eligible Tickets)")
    print("=" * 80)

    df_results, counts_regex, counts_naive = run_validation_pipeline()

    print("\n--- Validation Breakdown with re.findall(r'[a-z]+', text.lower()) ---")
    for status, count in sorted(counts_regex.items()):
        print(f"  - {status:<24}: {count:>3} tickets")

    print("\n--- Naive Whitespace Split Comparison (.split()) ---")
    for status, count in sorted(counts_naive.items()):
        print(f"  - {status:<24}: {count:>3} tickets")

    # Assert exact reference counts
    assert counts_regex.get("classified_json", 0) == 135, (
        f"Expected 135 classified_json, got {counts_regex.get('classified_json')}"
    )
    assert counts_regex.get("classified_keyword", 0) == 12, (
        f"Expected 12 classified_keyword, got {counts_regex.get('classified_keyword')}"
    )
    assert counts_regex.get("flagged_invalid_output", 0) == 9, (
        f"Expected 9 flagged_invalid_output, got {counts_regex.get('flagged_invalid_output')}"
    )

    # Assert naive counts matching the explanation in the prompt
    assert counts_naive.get("classified_json", 0) == 135
    assert counts_naive.get("classified_keyword", 0) == 0
    assert counts_naive.get("flagged_invalid_output", 0) == 21

    print("\n" + "=" * 80)
    print("  CHECKPOINT PASSED: Exactly 135 / 12 / 9 verified!")
    print("=" * 80)

    # Show examples of each status category
    print("\nSample 'classified_json' replies (Valid JSON + valid category):")
    sample_json = df_results[df_results["status"] == "classified_json"].head(2)
    for _, r in sample_json.iterrows():
        print(f"  Ticket {r['ticket_id']}: {r['raw_reply']} -> predicted={r['predicted_category']}")

    print("\nSample 'classified_keyword' replies (Non-JSON chatty replies):")
    sample_kw = df_results[df_results["status"] == "classified_keyword"].head(2)
    for _, r in sample_kw.iterrows():
        print(f"  Ticket {r['ticket_id']}: '{r['raw_reply']}' -> predicted={r['predicted_category']}")

    print("\nSample 'flagged_invalid_output' replies (JSON with invalid category 'shipping'):")
    sample_invalid = df_results[df_results["status"] == "flagged_invalid_output"].head(2)
    for _, r in sample_invalid.iterrows():
        print(f"  Ticket {r['ticket_id']}: '{r['raw_reply']}' -> routed to human review (reason: {r['reason']})")


if __name__ == "__main__":
    main()
