# 🛒 Instamart Support Copilot

A capstone project that builds an **offline-first, deterministic AI support pipeline** for a fictional quick-commerce platform — Instamart. It covers synthetic data generation, a mock LLM backend, ticket classification, policy Q&A, and a ReAct-style refund agent — all without any API keys or network calls.

---

## 📁 Project Structure

```
support-copilot/
├── .venv/                # Isolated virtual environment (not committed)
├── .gitignore            # Excludes .env, .venv/, __pycache__/, generated data
├── generate_data.py      # Synthetic data generator (run once)
├── mock_llm.py           # Offline LLM stand-in (classifier · policy · agent)
├── requirements.txt      # Pinned deps from pip freeze
├── data/
│   ├── raw/              # Generated CSV files (git-ignored, recreate with step 4)
│   │   ├── customers.csv
│   │   ├── products.csv
│   │   ├── orders.csv
│   │   ├── tickets.csv
│   │   ├── policy_eval.csv
│   │   └── agent_scenarios.csv
│   ├── clean/            # Cleaned CSVs (produced by cleaning notebook/script)
│   ├── policies/         # Plain-text policy documents (git-ignored, recreate with step 4)
│   │   ├── refund_policy.txt
│   │   ├── delivery_policy.txt
│   │   ├── payment_policy.txt
│   │   ├── privacy_policy.txt
│   │   └── escalation_policy.txt
│   └── sql/              # SQL queries against instamart.db
│       └── queries.sql
```

---

## 🔁 Reproduce

Follow these numbered steps exactly to go from a fresh clone to a fully working environment.

**1. Clone the repository and switch to `main`**

```bash
git clone https://github.com/PrabhuST-AI/support-copilot.git
cd support-copilot
git checkout main
```

**2. Create an isolated virtual environment**

```bash
python3 -m venv .venv
```

This creates a self-contained Python environment in `.venv/` (excluded from git via `.gitignore`).

**3. Activate the environment and install dependencies**

```bash
# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

`requirements.txt` was produced by `pip freeze` and contains exact pinned versions:

```
numpy==2.5.4
pandas==3.0.6
python-dateutil==2.9.0.post0
six==1.17.0
```

**4. Generate synthetic data**

```bash
python generate_data.py
```

Expected last line of output (must match exactly):

```
generated: customers=1200 products=40 orders_raw=13084 tickets=164
```

This writes:
- `data/raw/` — six CSV files (customers, products, orders, tickets, policy_eval, agent_scenarios)
- `data/policies/` — five plain-text policy files

Both directories are git-ignored and must be recreated via this step after a fresh clone.

**5. Verify the environment (optional smoke test)**

```bash
python -c "from mock_llm import chat; print('mock_llm OK')"
```

Expected output: `mock_llm OK`

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

## 🧹 Data Cleaning Rules

### 1. Customers (`data/clean/customers.csv`)
1. **City standardisation**: Leading and trailing spaces are removed and title-casing is applied (`.str.strip().str.title()`), standardising raw variations to exactly 6 canonical cities (Pune: 227, Delhi: 210, Bengaluru: 197, Chennai: 191, Hyderabad: 191, Mumbai: 184).
2. **Gender mapping**: Raw spellings are stripped, lowercased, and mapped (`m`/`male` → `male`, `f`/`female` → `female`, blank/unmapped → `unknown`), resulting in female: 610, male: 560, unknown: 30.
3. **Age validation & imputation**: Ages outside the range `[10, 100]` are treated as missing (`NaN`). Missing ages are imputed with the median age of the customer's city using `groupby('city')['age'].transform('median')`. Resulting dataset has 0 missing ages, and all customer ages lie between 18 and 60.
4. **Signup date conversion**: Converted `signup_date` to pandas datetime (`datetime64[us]`).

### 2. Orders (`data/clean/orders.csv`)
Executed sequentially in this exact order:
1. **Convert `order_ts` to datetime**: Converted using `pd.to_datetime(orders['order_ts'])`.
2. **Remove exact duplicate rows**: Dropped 119 exact duplicate rows using `orders.drop_duplicates()`, reducing 13,084 raw rows to 12,965 rows.
3. **Deduplicate `order_id` keeping latest `order_ts`**: Resolved multi-status orders (earlier `pending` records and later final records) by sorting by `['order_id', 'order_ts']` and keeping the last record per `order_id`. Yields exactly 12,309 unique orders (11,110 completed, 1,199 cancelled, and 0 pending).
4. **Numeric `order_amount` & price formula imputation**: Converted `order_amount` to numeric, coercing unparseable entries (`"N/A"`, `"error"`) to `NaN`. Filled missing values using: `round(price_inr * quantity * (1 - discount_pct / 100), 2)`. Results in 0 missing values and a completed orders total sum of exactly INR 2,998,668.80.
5. **Handle `delivery_minutes` (completed vs cancelled)**: Set negative durations (< 0) to `NaN`. For completed orders, filled missing delivery minutes with the median delivery time of that product's category among completed orders using `groupby('category')['delivery_minutes'].transform('median')`. For cancelled orders, left `delivery_minutes` missing (`NaN`), reflecting that cancelled orders were never delivered.

---

## 📊 SQL Analysis (`sql/queries.sql`)

Analytical queries executed against `data/instamart.db` using `pd.read_sql_query`:

- **Q1 (WHERE then HAVING)**: Filtered completed orders with `order_amount > 500` and kept customers whose total spend exceeds INR 8,000 (30 customers).
- **Q2 (Three-Table Join)**: Aggregated completed revenue across city and product category across 30 cells totaling INR 2,998,668.80 (largest cell: Bengaluru / Electronics at INR 233,909.45).
- **Q3 (Scalar Subquery)**: Identified 11 catalogue products priced strictly above the catalogue average (INR 289.32).
- **Q4 (LEFT JOIN)**: Found 25 customers who have placed zero completed orders.
- **Q5 (Common Table Expression)**: Computed benchmark customer average spend (INR 2,552.06) via two CTEs and returned 550 customers exceeding that benchmark.
- **Q6 (Ranking Window Functions)**: Computed city-partitioned spend ranks; ties cause `RANK()` to differ from `ROW_NUMBER()` because `RANK()` assigns identical ranks to tied rows and skips subsequent ranks, whereas `ROW_NUMBER()` strictly assigns unique, incrementing integers regardless of ties.
- **Q7 (Running Total)**: Evaluated 183 daily revenue rows from 2025-04-01 to 2025-09-30, achieving a cumulative total of exactly INR 2,998,668.80.

---

## 📋 Policy Documents

Five plain-text policies cover:

- **Refund** — eligibility, thresholds (INR 1 000 auto / human approval), SLA (5 working days), 7-day window.
- **Delivery** — 30-min promise, INR 25 fee below INR 199, rain extension, electronics OTP.
- **Payment** — UPI / card / COD (up to INR 2 500), 48 h failed-payment return, wallet rules.
- **Privacy** — data retention, masking, address deletion (180 days), account deletion (30 days).
- **Escalation** — fraud priority (2 h), premium SLA (1 h), 6-step handover rule.

---

## 🤖 Part D: Autonomous ReAct Refund Agent

The autonomous customer support agent operates via a ReAct (Reasoning + Acting) execution loop in [`react_agent.py`](file:///Users/prabhust/Documents/support-copilot/react_agent.py) (`run_agent`), interfacing with [`mock_llm.chat`](file:///Users/prabhust/Documents/support-copilot/mock_llm.py) and five deterministic domain tools defined in [`agent_tools.py`](file:///Users/prabhust/Documents/support-copilot/agent_tools.py):
- `get_order(order_id)`: Fetches database order status, amount, and delivery minutes.
- `search_policy(query)`: Cosine TF-IDF retrieval across policy chunks.
- `calculate_refund(order_amount, delivery_minutes)`: Validates delivery delay (> 45 minutes).
- `issue_refund(order_id, refund_inr)`: Automatically executes and logs refunds up to ₹1,000 to [`outputs/refund_ledger.jsonl`](file:///Users/prabhust/Documents/support-copilot/outputs/refund_ledger.jsonl).
- `escalate_to_human(order_id, refund_inr)`: Escalates refunds exceeding ₹1,000 to human leads with audit trail.

### Scenario Evaluation Results

All 7 evaluation rows from [`data/raw/agent_scenarios.csv`](file:///Users/prabhust/Documents/support-copilot/data/raw/agent_scenarios.csv) were processed and recorded to [`outputs/agent_runs.jsonl`](file:///Users/prabhust/Documents/support-copilot/outputs/agent_runs.jsonl):

| Scenario ID | Status | Refund Amount | Model Calls | Tokens Used |
|---|---|---|---|---|
| `S1` | **REFUND** | 128.25 | 5 | 897 |
| `S2` | **REFUND** | 135.00 | 5 | 897 |
| `S3` | **ESCALATE** | 1999.00 | 5 | 884 |
| `S4` | **DENIED** | 0.00 | 4 | 605 |
| `S5` | **DENIED** | 0.00 | 3 | 337 |
| `S6` | **NOT_FOUND** | 0.00 | 2 | 72 |
| `S7` | **FALLBACK_MAX_ITERATIONS** | 0.00 | 6 | 3000 |

### Safety Limits & Graceful Fallback Demonstrations

To guarantee safe and bounded execution in production, the agent enforces three independent guardrails:
1. **Iteration Limit (`max_iterations=6`)**: Terminate after 6 model calls without a final answer (`FALLBACK_MAX_ITERATIONS`). Prevents infinite loops caused by cyclical model queries.
2. **Cumulative Token Budget (`token_limit=6000`)**: After each model call, add `usage.total_tokens` to `tokens_used`. If `tokens_used > token_limit`, stop immediately without executing the requested tool (`FALLBACK_TOKEN_LIMIT`).
3. **Wall-Clock Timeout (`time_limit_seconds=10.0`)**: Evaluated at the top of every iteration using `time.monotonic()` (`elapsed >= time_limit_seconds`). If exceeded, stop immediately (`FALLBACK_TIMEOUT`).

Every fallback returns all completed steps so far, zero unauthorized refund disbursement, and a short human-handoff summary string in `final_answer`.

| Demo Run | Scenario | Condition / Guardrail | Resulting Status | Model Calls | Steps Completed | Tokens Used | Final Answer / Human Handoff |
|---|---|---|---|---|---|---|---|
| **Max Iterations** | `S7` | `max_iterations=6` | `FALLBACK_MAX_ITERATIONS` | 6 | 6 | 3000 | `FALLBACK_MAX_ITERATIONS: Maximum iterations (6) reached without final answer; handing off to human support lead.` |
| **Token Limit** | `S1` | `token_limit=200` | `FALLBACK_TOKEN_LIMIT` | 3 | 2 | 354 | `FALLBACK_TOKEN_LIMIT: Cumulative token limit of 200 exceeded (accumulated 354); handing off to human support lead.` |
| **Timeout Limit** | `S1` | `time_limit_seconds=0` | `FALLBACK_TIMEOUT` | 0 | 0 | 0 | `FALLBACK_TIMEOUT: Wall-clock limit of 0s reached (elapsed 0.0000s); handing off to human support lead.` |

- **Token Limit Verification**: In the `token_limit=200` run on `S1`, call 1 (`get_order`) used 38 tokens, call 2 (`search_policy`) used 62 tokens (cumulative 100 tokens), and call 3 returned a request for `calculate_refund` adding 254 tokens (cumulative 354 > 200). The agent immediately aborted without executing `calculate_refund`, preserving safe state with only 2 executed steps.
- **Timeout Verification**: When `time_limit_seconds=0`, `elapsed >= 0` evaluates to `True` at the top of iteration 1 before the first model call is dispatched, guaranteeing zero LLM spend across all platforms.

### Resumable Execution & Crash Recovery (Scenario S1 Demo)

To ensure fault tolerance against process crashes, network dropouts, or worker preemptions, the agent implements checkpointed state persistence:
- **Checkpoint Persistence**: After every tool observation is appended to the message list, a persistent snapshot is saved to `state/<run_id>.json`.
- **Checkpoint Schema**: Contains `run_id`, `scenario_id`, the full conversation `messages`, `tokens_used`, `step_number`, `tools_executed`, `steps`, `llm_calls`, `max_iterations`, `token_limit`, and `time_limit_seconds`.
- **Resumption (`resume_agent(run_id)`)**: Loads the checkpoint from disk and resumes the ReAct deliberation loop from the exact point of interruption without duplicating previously executed tools.

#### Crash & Recovery Demonstration (Scenario S1)
A simulated crash (`SimulatedCrashError`) was injected immediately after the second tool execution and after its checkpoint was safely written to `state/s1_crash_demo.json`. Resumption was then triggered via `resume_agent('s1_crash_demo')`.

**Tools executed across the crash and resume, in exact order:**
1. `get_order` *(Pre-crash execution, step 1: retrieved order details for `ORD-100048`)*
2. `search_policy` *(Pre-crash execution, step 2: retrieved late-delivery refund terms)*
   - *💥 SIMULATED CRASH: Process killed right after step 2 checkpoint persisted to disk.*
   - *🔄 RESUMED: `resume_agent('s1_crash_demo')` reloads checkpoint and continues loop.*
3. `calculate_refund` *(Post-resume execution, step 3: computed refund amount ₹128.25)*
4. `issue_refund` *(Post-resume execution, step 4: approved and logged refund to `outputs/refund_ledger.jsonl`)*

| Phase | Step | Tool Executed | State / Outcome |
|---|---|---|---|
| **Pre-Crash** | Step 1 | `get_order` | Checkpoint written to `state/s1_crash_demo.json` |
| **Pre-Crash** | Step 2 | `search_policy` | Checkpoint written to `state/s1_crash_demo.json` (`step_number=2`, `tools_executed=['get_order', 'search_policy']`) |
| **Crash Trigger** | - | - | Simulated crash raised immediately after step 2 checkpoint |
| **Resume Call** | - | - | `resume_agent('s1_crash_demo')` called from persisted state |
| **Post-Resume** | Step 3 | `calculate_refund` | Checkpoint updated (`step_number=3`, `refund_inr=128.25`) |
| **Post-Resume** | Step 4 | `issue_refund` | Checkpoint updated (`step_number=4`, `refund_inr=128.25`) |
| **Final State** | - | - | Status **REFUND**, Amount **₹128.25**, LLM Calls: 5 |

**Outcome**: Exactly 4 tool executions occurred across both halves of the run (`get_order` → `search_policy` → `calculate_refund` → `issue_refund`), zero tools were repeated or skipped, and the final state concluded with status **REFUND** and ₹128.25.

---

## ⚠️ Limitations and Next Steps

- **Post-Hoc Segment Evaluation**: The retention list is illustrative and not a leakage-free evaluation because the Premium customer segment flag is a post-hoc whole-window label derived from RFM clustering across the entire observation window (including the September window that defines the lapse target label). (The model's own features strictly use only orders before the cutoff, as required.)
- **Next Steps**: Future iterations should compute RFM customer segments dynamically using pre-cutoff order history only, test cost-sensitive loss functions, and evaluate retention interventions through randomized A/B holdout testing.

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

MIT
