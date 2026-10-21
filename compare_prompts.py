"""compare_prompts.py -- Compare Zero-Shot vs Few-Shot Ticket Classification.

Tasks (3 marks):
Run the full pipeline once with each prompt (no token budget).
For each run report:
- Tickets classified
- Accuracy against true_category over the classified tickets
- Total tokens (sum of response.usage.total_tokens)
- Tokens per classified ticket
Show a confusion table for the zero-shot run.
Explain in 3-4 sentences, with numbers from your own runs, when the extra few-shot
tokens are worth paying for.

Reference:
- 147 classified tickets (135 classified_json + 12 classified_keyword)
- Zero-shot accuracy: 85.7% (126 of 147; 11 fraud -> billing, 10 technical -> billing)
- Few-shot accuracy: 100.0% (147 of 147)
"""

from typing import Any, Dict, List, Tuple
import pandas as pd

from input_guardrails import check_input_guardrail
from mock_llm import reset
from prompts import FEW_SHOT_PROMPT, ZERO_SHOT_PROMPT
from redact_pii import redact_text
from resilient_calls import call_with_retry
from validate_answers import validate_and_classify_reply

RAW_TICKETS_PATH = "data/raw/tickets.csv"


def evaluate_prompt_pipeline(
    prompt_name: str,
    system_prompt: str,
    tickets_path: str = RAW_TICKETS_PATH,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Runs end-to-end classification pipeline for a given system prompt."""
    df = pd.read_csv(tickets_path, dtype=str, keep_default_na=False)

    # 1. Input Guardrails (filter to eligible tickets)
    eligible_rows = [row for _, row in df.iterrows() if check_input_guardrail(row["text"])[0]]

    # 2. Reset mock LLM state before the full batch run
    reset()

    total_tokens = 0
    records = []

    # 3. Process each eligible ticket with PII redaction and resilient calls
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
        total_tokens += resp.usage.total_tokens

        reply = resp.choices[0].message.content
        status, pred_cat, reason = validate_and_classify_reply(reply)

        records.append({
            "ticket_id": tid,
            "customer_id": cid,
            "order_id": oid,
            "true_category": true_cat,
            "predicted_category": pred_cat,
            "status": status,
            "reason": reason,
            "tokens": resp.usage.total_tokens,
        })

    df_all = pd.DataFrame(records)

    # Filter to successfully classified tickets (classified_json + classified_keyword)
    df_classified = df_all[df_all["status"].isin(["classified_json", "classified_keyword"])].copy()

    n_classified = len(df_classified)
    n_correct = (df_classified["true_category"] == df_classified["predicted_category"]).sum()
    accuracy = n_correct / n_classified if n_classified > 0 else 0.0
    tok_per_classified = total_tokens / n_classified if n_classified > 0 else 0.0

    metrics = {
        "prompt_name": prompt_name,
        "eligible_tickets": len(eligible_rows),
        "tickets_classified": n_classified,
        "correct_classifications": n_correct,
        "accuracy": accuracy,
        "total_tokens": total_tokens,
        "tokens_per_classified_ticket": tok_per_classified,
    }

    return df_classified, metrics


def main():
    print("=" * 80)
    print("  ZERO-SHOT VS. FEW-SHOT PIPELINE COMPARISON")
    print("=" * 80)

    # 1. Run Zero-Shot
    df_zero, metrics_zero = evaluate_prompt_pipeline("Zero-Shot", ZERO_SHOT_PROMPT)

    # 2. Run Few-Shot
    df_few, metrics_few = evaluate_prompt_pipeline("Few-Shot", FEW_SHOT_PROMPT)

    # 3. Present Comparison Table
    print("\n" + "=" * 80)
    print("  SUMMARY PERFORMANCE COMPARISON")
    print("=" * 80)
    print(f"{'Metric':<32} | {'Zero-Shot':<18} | {'Few-Shot':<18}")
    print("-" * 74)
    print(f"{'Tickets Classified':<32} | {metrics_zero['tickets_classified']:<18} | {metrics_few['tickets_classified']:<18}")
    print(f"{'Correct Classifications':<32} | {metrics_zero['correct_classifications']} / {metrics_zero['tickets_classified']:<12} | {metrics_few['correct_classifications']} / {metrics_few['tickets_classified']:<12}")
    print(f"{'Accuracy on Classified':<32} | {metrics_zero['accuracy']*100:.2f}%{'':<11} | {metrics_few['accuracy']*100:.2f}%{'':<11}")
    print(f"{'Total Tokens Used':<32} | {metrics_zero['total_tokens']:<18} | {metrics_few['total_tokens']:<18}")
    print(f"{'Tokens per Classified Ticket':<32} | {metrics_zero['tokens_per_classified_ticket']:.2f}{'':<12} | {metrics_few['tokens_per_classified_ticket']:.2f}{'':<12}")
    print("=" * 80)

    # 4. Confusion Table for Zero-Shot
    print("\n" + "=" * 80)
    print("  ZERO-SHOT CONFUSION TABLE")
    print("=" * 80)
    ct_zero = pd.crosstab(
        df_zero["true_category"],
        df_zero["predicted_category"],
        margins=True,
        margins_name="Total"
    )
    print(ct_zero)

    # 5. Analysis of Zero-Shot Errors
    errors_zero = df_zero[df_zero["true_category"] != df_zero["predicted_category"]]
    print(f"\nZero-Shot Classification Errors ({len(errors_zero)} total):")
    error_summary = errors_zero.groupby(["true_category", "predicted_category"]).size()
    for (t_cat, p_cat), cnt in error_summary.items():
        print(f"  - True '{t_cat}' predicted as '{p_cat}': {cnt} tickets")

    # Assert reference benchmarks
    assert metrics_zero["tickets_classified"] == 147, f"Expected 147 zero-shot, got {metrics_zero['tickets_classified']}"
    assert metrics_few["tickets_classified"] == 147, f"Expected 147 few-shot, got {metrics_few['tickets_classified']}"
    assert metrics_zero["correct_classifications"] == 126, f"Expected 126 correct, got {metrics_zero['correct_classifications']}"
    assert abs(metrics_zero["accuracy"] - 0.8571) < 0.001, f"Expected 85.7% accuracy, got {metrics_zero['accuracy']}"
    assert metrics_few["correct_classifications"] == 147, f"Expected 147 correct, got {metrics_few['correct_classifications']}"
    assert metrics_few["accuracy"] == 1.0, f"Expected 100% accuracy, got {metrics_few['accuracy']}"

    assert len(errors_zero) == 21, f"Expected 21 errors, got {len(errors_zero)}"
    assert error_summary.get(("fraud", "billing"), 0) == 11, "Expected 11 fraud->billing errors"
    assert error_summary.get(("technical", "billing"), 0) == 10, "Expected 10 technical->billing errors"

    print("\n" + "=" * 80)
    print("  CHECKPOINT PASSED: All benchmark figures verified successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
