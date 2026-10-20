# 🛒 Instamart Support Copilot

A capstone project that builds an **offline-first, deterministic AI support pipeline** for a fictional quick-commerce platform — Instamart. It covers synthetic data generation, a mock LLM backend, ticket classification, policy Q&A, and a ReAct-style refund agent — all without any API keys or network calls.

---

## 📁 Project Structure

```
support-copilot/
├── generate_data.py      # Synthetic data generator (run once)
├── mock_llm.py           # Offline LLM stand-in (classifier · policy · agent)
├── requirements.txt      # Python dependencies
├── data/
│   ├── raw/              # Generated CSV files
│   │   ├── customers.csv
│   │   ├── products.csv
│   │   ├── orders.csv
│   │   ├── tickets.csv
│   │   ├── policy_eval.csv
│   │   └── agent_scenarios.csv
│   └── policies/         # Plain-text policy documents
│       ├── refund_policy.txt
│       ├── delivery_policy.txt
│       ├── payment_policy.txt
│       ├── privacy_policy.txt
│       └── escalation_policy.txt
```

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/PrabhuST-AI/support-copilot.git
cd support-copilot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> The core scripts (`generate_data.py`, `mock_llm.py`) use **only the Python standard library** — no extra packages required.

### 3. Generate synthetic data

```bash
python3 generate_data.py
```

Expected output:

```
generated: customers=1200 products=40 orders_raw=13084 tickets=164
```

This writes all CSV files into `data/raw/` and policy text files into `data/policies/`.

---

## 📦 Modules

### `generate_data.py`

Deterministic synthetic data generator (seeded `random.Random(42)`). Produces:

| File | Description |
|------|-------------|
| `customers.csv` | 1 200 customers across 6 Indian cities with intentional data-quality issues (missing ages, mangled city/gender values) |
| `products.csv` | 40 SKUs across Grocery, Dairy, Snacks, Personal Care, and Electronics |
| `orders.csv` | ~13 000 raw order rows with duplicates, pending records, impossible delivery times, and unparseable amounts planted for cleaning exercises |
| `tickets.csv` | 164 support tickets across billing / delivery / technical / fraud categories, including empty and very-long edge cases |
| `policy_eval.csv` | 14 Q&A pairs for evaluating the policy-assistant mode |
| `agent_scenarios.csv` | 6 labelled refund scenarios (REFUND · ESCALATE · DENIED · NOT\_FOUND · FALLBACK) |

### `mock_llm.py`

A drop-in offline replacement for a hosted chat API. Import and call it like a real LLM client:

```python
from mock_llm import chat, reset, TransientLLMError

response = chat(messages)
print(response.choices[0].message.content)
print(response.usage.total_tokens)
```

**Three modes**, selected by a keyword in the first (system) message:

| Keyword | Behaviour |
|---------|-----------|
| `CLASSIFY` | Keyword-based support-ticket classifier → returns JSON `{"category": ..., "reason": ...}` |
| `POLICY ASSISTANT` | TF-IDF-style grounded answerer — finds the best-matching sentence from a `CONTEXT:` block |
| `AGENT` | ReAct-style refund agent that emits tool calls (`get_order`, `search_policy`, `calculate_refund`, `issue_refund`, `escalate_to_human`) |

**Simulated misbehaviour** (built in for robustness testing):

- Every **9th** call raises `TransientLLMError("503 server busy")` — tests retry logic.
- Every **13th** successful call returns a chatty non-JSON response — tests parse-error handling.
- Every **17th** successful call returns an out-of-vocabulary category — tests validation logic.

---

## 🏗️ Architecture Overview

```
generate_data.py
      │
      ▼
  data/raw/  ◄────────────────────────────────────┐
  data/policies/                                  │
                                                  │
mock_llm.chat()  ──► CLASSIFY  ──► ticket label   │
                 ──► POLICY    ──► grounded answer │
                 ──► AGENT     ──► tool calls ─────┘
                                   (ReAct loop)
```

---

## 🧪 Data Quality Issues (Intentional)

The synthetic data contains deliberate "real-world" messiness for cleaning exercises:

- **Age outliers**: ages like 280, -4, 150, 999 planted at fixed customer indices.
- **Missing values**: ~5 % of ages and ~3 % of genders are blank.
- **Mangled cities/genders**: random casing and whitespace variants.
- **Duplicate orders**: ~5 % partial duplicates (`pending` state) + ~1 % exact duplicates.
- **Impossible delivery times**: negative durations planted at every 100th row.
- **Unparseable amounts**: `"N/A"` and `"error"` injected in ~1 % of completed orders.

---

## 📋 Policy Documents

Five plain-text policies cover:

- **Refund** — eligibility, thresholds (INR 1 000 auto / human approval), SLA (5 working days), 7-day window.
- **Delivery** — 30-min promise, INR 25 fee below INR 199, rain extension, electronics OTP.
- **Payment** — UPI / card / COD (up to INR 2 500), 48 h failed-payment return, wallet rules.
- **Privacy** — data retention, masking, address deletion (180 days), account deletion (30 days).
- **Escalation** — fraud priority (2 h), premium SLA (1 h), 6-step handover rule.

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

MIT
