"""redact_pii.py -- Personal Identifiable Information (PII) Redaction for Support Tickets.

Tasks (3 marks):
Apply, in this exact order, these regular expressions and placeholders to ticket text:
1. email:    r"[\\w.+-]+@[\\w-]+\\.[\\w.]+"  -> [EMAIL]
2. phone:    r"\\b[6-9]\\d{9}\\b"            -> [PHONE]
3. order id: r"ORD-\\d{6}"                   -> [ORDER_ID]

Reference:
- 55 emails
- 91 phone numbers
- 132 order ids
- 278 substitutions in total
- 0 matches remaining afterwards
- The user message sent to the LLM must be exactly the redacted ticket text and nothing else.
"""

import re
from typing import Dict, List, Tuple
import pandas as pd

RAW_TICKETS_PATH = "data/raw/tickets.csv"

# Ordered list of (name, pattern, placeholder)
REDACTION_RULES = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
    ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]"),
]


def redact_text(text: str) -> Tuple[str, Dict[str, int]]:
    """Applies the 3 redaction rules in order and returns (redacted_text, counts_dict)."""
    if text is None:
        return "", {name: 0 for name, _, _ in REDACTION_RULES}

    cur = str(text)
    counts = {}

    for name, pattern, repl in REDACTION_RULES:
        # Count occurrences before substitution
        matches = len(pattern.findall(cur))
        counts[name] = matches
        cur = pattern.sub(repl, cur)

    return cur, counts


def verify_no_remaining_pii(text: str) -> Dict[str, int]:
    """Re-applies the 3 patterns to verify 0 matches remain."""
    remaining = {}
    for name, pattern, _ in REDACTION_RULES:
        remaining[name] = len(pattern.findall(text))
    return remaining


def audit_dataset_redaction(csv_path: str = RAW_TICKETS_PATH) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, int]]:
    """Applies PII redaction across all tickets in dataset and validates clean state."""
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

    total_substitutions = {"email": 0, "phone": 0, "order_id": 0}
    total_remaining = {"email": 0, "phone": 0, "order_id": 0}

    redacted_texts = []
    for _, row in df.iterrows():
        orig_text = row["text"]
        redacted, counts = redact_text(orig_text)
        rem = verify_no_remaining_pii(redacted)

        for k in total_substitutions:
            total_substitutions[k] += counts[k]
            total_remaining[k] += rem[k]

        redacted_texts.append(redacted)

    df["redacted_text"] = redacted_texts
    return df, total_substitutions, total_remaining


def main():
    print("=" * 80)
    print("  PERSONAL IDENTIFIABLE INFORMATION (PII) REDACTION AUDIT")
    print("=" * 80)
    print("Rules applied in exact order:")
    print("  1. email:    [\\w.+-]+@[\\w-]+\\.[\\w.]+ -> [EMAIL]")
    print("  2. phone:    \\b[6-9]\\d{9}\\b           -> [PHONE]")
    print("  3. order id: ORD-\\d{6}                  -> [ORDER_ID]")

    df, subs, rems = audit_dataset_redaction(RAW_TICKETS_PATH)

    total_subs = sum(subs.values())
    total_rems = sum(rems.values())

    print(f"\nAudit across all {len(df)} tickets:")
    print(f"  - Emails redacted:       {subs['email']:>3} substitutions")
    print(f"  - Phone numbers redacted:{subs['phone']:>3} substitutions")
    print(f"  - Order IDs redacted:    {subs['order_id']:>3} substitutions")
    print(f"  -------------------------------------------")
    print(f"  - Total Substitutions:   {total_subs:>3}")

    print(f"\nVerification (Re-applying patterns to redacted texts):")
    print(f"  - Remaining emails:      {rems['email']:>3}")
    print(f"  - Remaining phones:      {rems['phone']:>3}")
    print(f"  - Remaining order IDs:   {rems['order_id']:>3}")
    print(f"  -------------------------------------------")
    print(f"  - Total Remaining:       {total_rems:>3}")

    # Reference assertions
    assert subs["email"] == 55, f"Expected 55 emails, found {subs['email']}"
    assert subs["phone"] == 91, f"Expected 91 phone numbers, found {subs['phone']}"
    assert subs["order_id"] == 132, f"Expected 132 order IDs, found {subs['order_id']}"
    assert total_subs == 278, f"Expected 278 total substitutions, found {total_subs}"
    assert total_rems == 0, f"Expected 0 remaining matches, found {total_rems}"

    print("\n" + "=" * 80)
    print("  CHECKPOINT PASSED: Exactly 278 substitutions made; 0 left afterwards.")
    print("=" * 80)

    # Show a few before & after examples
    print("\nSample Redacted Tickets:")
    samples = df[df["text"].str.contains("@") | df["text"].str.contains("ORD-")].head(3)
    for _, row in samples.iterrows():
        print(f"\n[Ticket {row['ticket_id']}]")
        print(f"  Original: {row['text']}")
        print(f"  Redacted: {row['redacted_text']}")


if __name__ == "__main__":
    main()
