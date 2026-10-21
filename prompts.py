"""prompts.py -- Zero-shot and Few-shot system prompts for support ticket classification.

Tasks (2 marks):
Write a zero-shot and a few-shot system prompt:
- Both must contain the word CLASSIFY.
- Both must list the four allowed categories: billing, delivery, technical, fraud.
- Both must require a JSON-only reply of the form: {"category": "...", "reason": "..."}.
- The zero-shot prompt must NOT contain the text "EXAMPLES:".
- The few-shot prompt MUST contain a line "EXAMPLES:" followed by at least two labelled
  examples, one of which is a fraud example.
"""

ZERO_SHOT_PROMPT = """You are an automated support ticket classification assistant for an e-commerce platform.
Your task is to CLASSIFY the incoming customer support ticket into exactly one of the four allowed categories:
- billing
- delivery
- technical
- fraud

You must provide a JSON-only reply of the form {"category": "...", "reason": "..."}.
Do not include any markdown fences, extra explanations, or surrounding text."""

FEW_SHOT_PROMPT = """You are an automated support ticket classification assistant for an e-commerce platform.
Your task is to CLASSIFY the incoming customer support ticket into exactly one of the four allowed categories:
- billing
- delivery
- technical
- fraud

You must provide a JSON-only reply of the form {"category": "...", "reason": "..."}.

EXAMPLES:
Ticket: "There is an unauthorized transaction on my credit card linked to my account. I did not place this order and suspect my card was stolen."
Reply: {"category": "fraud", "reason": "unauthorized card transaction and stolen payment credentials"}

Ticket: "I was charged twice for order [ORDER_ID]. Please refund the extra payment to my bank."
Reply: {"category": "billing", "reason": "duplicate charge and refund request"}

Ticket: "The app crashes and freezes every time I attempt to enter my delivery address at checkout."
Reply: {"category": "technical", "reason": "application crash and screen freeze during checkout"}

Ticket: "The delivery rider marked my order as delivered, but the package was never received at my address."
Reply: {"category": "delivery", "reason": "missing package marked delivered by courier"}
"""


def validate_prompts():
    """Validates structural and semantic requirements for both prompts."""
    categories = ["billing", "delivery", "technical", "fraud"]

    # 1. Zero-shot validation
    assert "CLASSIFY" in ZERO_SHOT_PROMPT, "Zero-shot prompt must contain the word 'CLASSIFY'"
    for cat in categories:
        assert cat in ZERO_SHOT_PROMPT, f"Zero-shot prompt must mention category '{cat}'"
    assert '{"category": "...", "reason": "..."}' in ZERO_SHOT_PROMPT, (
        "Zero-shot prompt must require JSON format {'category': '...', 'reason': '...'}"
    )
    assert "EXAMPLES:" not in ZERO_SHOT_PROMPT, "Zero-shot prompt must NOT contain 'EXAMPLES:'"

    # 2. Few-shot validation
    assert "CLASSIFY" in FEW_SHOT_PROMPT, "Few-shot prompt must contain the word 'CLASSIFY'"
    for cat in categories:
        assert cat in FEW_SHOT_PROMPT, f"Few-shot prompt must mention category '{cat}'"
    assert '{"category": "...", "reason": "..."}' in FEW_SHOT_PROMPT, (
        "Few-shot prompt must require JSON format {'category': '...', 'reason': '...'}"
    )
    assert "EXAMPLES:" in FEW_SHOT_PROMPT, "Few-shot prompt must contain a line 'EXAMPLES:'"
    assert '"fraud"' in FEW_SHOT_PROMPT, "Few-shot prompt must include at least one fraud example"

    print("All prompt structure and content assertions passed successfully!")


if __name__ == "__main__":
    validate_prompts()
    print("\n--- ZERO-SHOT PROMPT ---")
    print(ZERO_SHOT_PROMPT)
    print("\n--- FEW-SHOT PROMPT ---")
    print(FEW_SHOT_PROMPT)
