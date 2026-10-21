# 🛒 Instamart Support Copilot

An end-to-end, offline-first, deterministic AI support and customer retention pipeline for a fictional quick-commerce platform — **Instamart**. It covers data engineering, customer churn/lapse modeling, support ticket triage with RAG, and an autonomous ReAct refund agent — completely free of external API keys, network dependencies, or cloud costs.

---

## Overview

The Instamart Support Copilot demonstrates a production-grade customer support and intelligence engine structured across four interconnected parts:

1. **Part A (Data Engineering & SQL Analytics)**: Cleaning, imputing, and standardising raw customer and order data; modeling them into SQLite (`data/instamart.db`); and executing analytical SQL queries (window functions, CTEs, multi-table joins).
2. **Part B (Customer Segmentation & Lapse Prediction)**: RFM behavioral clustering, leakage-free churn cohort definition, logistic regression and random forest classification, class-balance oversampling, model explainability, and proactive high-value retention lists.
3. **Part C (Support Triage, Guardrails & Policy RAG)**: PII redaction (regex sanitisation), input length/empty guardrails, resilient exponential backoff calls, JSON/keyword validation, zero-shot vs. few-shot prompt comparison, token-budgeted queue routing, TF-IDF cosine document retrieval, and grounded policy question-answering with overlap sensitivity audits.
4. **Part D (Autonomous ReAct Refund Agent)**: Multi-turn ReAct reasoning and tool invocation across 5 domain tools, bounded by 3 execution guardrails (`FALLBACK_MAX_ITERATIONS`, `FALLBACK_TOKEN_LIMIT`, `FALLBACK_TIMEOUT`), persistent state checkpointing (`state/`), and seamless crash recovery (`resume_agent`).

---

## Reproduce

Follow these numbered shell commands sequentially to reproduce every artifact, model, database table, and evaluation report from scratch:

```bash
# 1. Create isolated virtual environment
python3 -m venv .venv

# 2. Activate virtual environment
source .venv/bin/activate

# 3. Install pinned dependencies
pip install -r requirements.txt

# 4. Generate raw synthetic datasets and policy documents
python generate_data.py

# 5. Clean customer records (city standardisation, gender mapping, median age imputation)
python clean_customers.py

# 6. Clean order records (deduplication, revenue formula imputation, delivery time handling)
python clean_orders.py

# 7. Store clean data into SQLite (data/instamart.db) and run validation assertions
python store_and_validate.py

# 8. Execute analytical SQL queries against SQLite database
python run_sql_analysis.py

# 9. Perform Exploratory Data Analysis (EDA) on distributions, categories, and delivery times
python eda_analysis.py

# 10. Compute RFM metrics and cluster customers into 4 behavioral segments (Part B)
python build_rfm.py
python cluster_rfm.py
python segmentation.py

# 11. Construct leakage-free lapse cohort and preprocess features with train/test isolation
python build_lapse_cohort.py
python preprocess_lapse.py

# 12. Train, tune, and evaluate Logistic Regression and Random Forest models
python train_evaluate_models.py

# 13. Run class-balance oversampling experiment on the training split
python class_balance_experiment.py

# 14. Interpret model feature weights, persist joblib pipeline, and generate retention list
python interpret_persist_and_use.py

# 15. Audit regex-based PII redaction and apply input guardrails (Part C)
python redact_pii.py
python input_guardrails.py

# 16. Test resilient calls with exponential backoff and jitter under simulated 503 errors
python resilient_calls.py

# 17. Validate and parse model responses (JSON and keyword fallback extraction)
python validate_answers.py

# 18. Compare zero-shot vs few-shot prompt performance and generate confusion table
python compare_prompts.py

# 19. Run token budget exhaustion demo and execute triage routing to department queues
python triage_routing.py

# 20. Chunk policy documents (40 words / 10 words overlap) and fit TF-IDF vectorizer
python chunk_documents.py
python vectorize_and_retrieve.py

# 21. Query grounded policy assistant using TF-IDF context injection
python augment_and_answer.py

# 22. Evaluate policy retrieval, grounded fact matching, and chunk overlap sensitivity
python evaluate_rag.py

# 23. Run autonomous ReAct refund agent, safety limit demos, and crash-resumption (Part D)
python react_agent.py
```

---

## Part A

### 1. Data Cleaning Rules & Metrics

| Dataset | Raw Count | Cleaning Operations Applied | Final Clean Count | Key Validation Metric |
|---|---|---|---|---|
| **Customers** (`data/clean/customers.csv`) | 1,200 | • Standardised cities to 6 canonical names (`Title()`).<br>• Mapped raw gender spellings (`female`: 610, `male`: 560, `unknown`: 30).<br>• Replaced ages outside `[10, 100]` with `NaN` and imputed using city median. | 1,200 | 0 missing ages; customer ages strictly between 18 and 60. |
| **Orders** (`data/clean/orders.csv`) | 13,084 | • Converted `order_ts` to datetime.<br>• Dropped 119 exact duplicate rows (reduced to 12,965).<br>• Deduplicated `order_id` keeping latest `order_ts` (resolved pending vs final).<br>• Imputed missing/corrupt amounts via formula: `round(price_inr * quantity * (1 - discount_pct / 100), 2)`.<br>• Negative delivery minutes set to `NaN`; completed orders imputed with product category median; cancelled orders left `NaN`. | 12,309 | 11,110 completed, 1,199 cancelled, 0 pending.<br>Completed orders revenue sum = **₹2,998,668.80**. |

### 2. SQLite Database Modeling (`data/instamart.db`)
The cleaned datasets are stored as relational tables in `data/instamart.db`:
- `customers`: 1,200 rows (`customer_id`, `city`, `gender`, `age`, `signup_date`)
- `products`: 40 rows across 5 product categories (`product_id`, `category`, `price_inr`)
- `orders`: 12,309 rows (`order_id`, `customer_id`, `order_amount`, `status`, `delivery_minutes`, `order_ts`)

### 3. Analytical SQL Queries (`sql/queries.sql`)

| Query ID | Objective / Technique | Key Reported Result |
|---|---|---|
| **Q1** | High-value customer filter: `WHERE status='completed' AND order_amount > 500 GROUP BY customer_id HAVING SUM(order_amount) > 8000` | **30 customers** qualified with high-ticket orders exceeding ₹8,000 spend. |
| **Q2** | Multi-table join (`orders` $\Join$ `customers` $\Join$ `products`): Aggregate completed revenue across City $\times$ Category. | **30 cells** totaling exactly **₹2,998,668.80** (top cell: Bengaluru / Electronics at ₹233,909.45). |
| **Q3** | Scalar subquery filter: Products priced strictly above catalogue mean. | **11 products** priced above catalogue average of **₹289.32** (top: Noise-Cancelling Earbuds at ₹1,999.00). |
| **Q4** | Left join identifying dormant accounts: Customers with zero completed orders. | **25 customers** have placed zero completed orders. |
| **Q5** | Common Table Expression (CTE) benchmark comparison: Average customer spend benchmark. | Benchmark average spend = **₹2,552.06**; **550 customers** spend strictly above this average. |
| **Q6** | City-partitioned spend ranks comparing `RANK()` vs `ROW_NUMBER()`. | `RANK()` assigns identical ranks to tied rows and skips ranks, whereas `ROW_NUMBER()` strictly assigns sequential integers. |
| **Q7** | Daily running total window query across the full 6-month observation window. | **183 daily revenue rows** (2025-04-01 to 2025-09-30) totaling exactly **₹2,998,668.80**. |

---

## Part B

### 1. RFM Customer Segmentation (`outputs/customer_segments.csv`)
Using order history across 1,175 active customers, RFM features (Recency, Frequency, Monetary) were clustered into 4 behavioral segments:

| Segment ID | Segment Name | Customer Count | Total Spend (INR) | Mean Spend | Description / Action |
|---|---|---|---|---|---|
| `0` | **Occasional Buyers** | 436 (37.1%) | ₹493,529.10 | ₹1,131.95 | Low frequency, modest spend; nurture with coupons. |
| `1` | **Premium Customers** (`is_premium=True`) | 138 (11.7%) | ₹1,032,689.85 | ₹7,483.26 | Highest monetary spend; eligible for priority SLA desk. |
| `2` | **Loyal / Frequent Shoppers** | 419 (35.7%) | ₹1,299,661.10 | ₹3,101.82 | Steady recurring volume; cross-sell high-margin categories. |
| `3` | **At-Risk / Churned** | 182 (15.5%) | ₹172,788.75 | ₹949.39 | Stale recency; target with win-back promotions. |

### 2. Leakage-Free Churn / Lapse Cohort (`outputs/lapse_cohort.csv`)
- **Cutoff Date**: `2025-08-31` (orders strictly before this cutoff were used for feature engineering).
- **Target Definition (`is_lapsed`)**: Evaluated in September 2025 (30-day window); `1` if customer placed 0 orders in September, else `0`.
- **Cohort Size**: 1,175 customers (524 lapsed = 44.6%, 651 retained = 55.4%).
- **Data Splitting**: Stratified 80/20 train/test split (940 train, 235 test) with standard scaling and one-hot encoding fit strictly on training data.

### 3. Model Comparison Table (Test Set Evaluation)

| Model | Hyperparameters | 5-Fold CV F1 | Test Accuracy | Test Precision | Test Recall | Test F1-Score |
|---|---|---|---|---|---|---|
| **Majority Baseline** | Predict Class 0 (`not lapsed`) | — | 61.6% | 0.00% | 0.00% | 0.0000 |
| **Logistic Regression** | $C=100.0$, `l2` penalty | 0.6506 | 72.84% | 64.44% | 65.17% | 0.6480 |
| **Random Forest** | `max_depth=8`, `min_samples_leaf=10` | **0.6829** | **73.71%** | **65.91%** | **65.17%** | **0.6554** |

### 4. Class-Balance Oversampling Experiment

The training set was oversampled from 924 rows (570 non-lapsed, 354 lapsed) to 1,140 rows (570 per class) by duplicating lapsed examples with replacement (`random_state=42`). The test set (232 rows) remained untouched.

| Metric | Un-balanced Model ($C=100$) | Balanced Model ($C=100$) | Delta ($\Delta$) | Impact on Retention Team |
|---|---|---|---|---|
| **Precision** | 64.44% | 58.91% | -5.53% | Slight increase in false alarms (contacting customers who wouldn't lapse). |
| **Recall** | **65.17%** | **85.39%** | **+20.22%** | **Highest mover.** Catches 76 of 89 churning customers (up from 58). |
| **F1-Score** | 0.6480 | 0.6972 | +0.0492 | Overall harmonic balance improves significantly. |
| **Accuracy** | 72.84% | 71.55% | -1.29% | Minor trade-off in accuracy to prioritize churn recall. |

*Strategic Decision*: For retention teams, recall is paramount. Gaining +20.22% recall far outweighs a 5.53% drop in precision, protecting vital customer lifetime value.

### 5. Model Interpretation & Top Feature Drivers
- **Random Forest Gini Importances**:
  1. `frequency`: **27.96%** (strongest predictor of loyalty)
  2. `monetary`: **27.19%** (historical spend volume)
  3. `recency`: **11.49%** (days since last purchase)
  4. `avg_delivery_minutes`: **6.14%** (fulfillment friction)
  5. `cancellation_rate`: **5.27%** (service dissatisfaction)
- **Logistic Regression Coefficients**:
  - `frequency`: **-1.70** (higher frequency sharply reduces lapse likelihood)
  - `recency`: **+0.04** (higher inactivity increases lapse likelihood)
  - `avg_delivery_minutes`: **+0.08** (slower delivery increases churn risk)

### 6. Proactive Retention List (`outputs/retention_list.csv`)
The trained pipeline was persisted to [`outputs/lapse_model.joblib`](file:///Users/prabhust/Documents/support-copilot/outputs/lapse_model.joblib) (577.0 KB) and used to score test set customers. The top 10 highest-risk customers within the Premium segment were extracted:

| Customer ID | Predicted Lapse Probability | True Outcome (`actually_lapsed`) | Priority Action |
|---|---|---|---|
| `C0862` | 49.33% | 0 | Dedicated account manager check-in |
| `C1112` | 48.41% | 0 | Exclusive VIP anniversary discount |
| `C0060` | 48.03% | 0 | Free delivery pass for 30 days |
| `C0802` | 47.96% | 0 | Proactive service quality audit |
| `C1008` | 47.76% | 0 | Personalised category voucher |
| `C1000` | 46.24% | 0 | Re-engagement SMS with instant credit |
| `C0525` | 45.97% | 1 (Truly Lapsed) | Churn recovery survey & apology coupon |
| `C0267` | 45.76% | 0 | VIP concierge outreach |
| `C0133` | 45.63% | 0 | Personalized product recommendations |
| `C0167` | 45.22% | 0 | Tailored subscription offer |

---

## Part C

### 1. PII Redaction Audit (`redact_pii.py`)
Applied sequential regex substitutions to sanitize support tickets:
- Emails (`[\w.+-]+@[\w-]+\.[\w.]+]`): **55 substitutions**
- Phone numbers (`\b[6-9]\d{9}\b`): **91 substitutions**
- Order IDs (`ORD-\d{6}`): **132 substitutions**
- **Audit Outcome**: Exactly **278 total substitutions** across 164 tickets; **0 remaining PII instances** confirmed via second-pass verification.

### 2. Input Guardrails & Resilient Calls
- **Input Guardrails** (`data/raw/tickets.csv`, 164 tickets):
  - `skipped_empty`: **1 ticket** (length 0)
  - `flagged_long`: **7 tickets** (length > 500 chars, routed to human review)
  - `eligible`: **156 tickets** passed to model classification
- **Resilient Calls** (`resilient_calls.py`):
  - Every 9th mock call raises `TransientLLMError("503 server busy")` (19 simulated errors).
  - Wrapped with exponential backoff & jitter ($0.05 \times 2^n \times [0.9, 1.15]$).
  - **Outcome**: **19 / 19 errors recovered** (100% recovery rate); **0 permanent failures**.

### 3. Response Validation & Parsing (`validate_answers.py`)
- Evaluated across the 156 eligible tickets:
  - `classified_json`: **135 tickets** (valid JSON matching allowed categories)
  - `classified_keyword`: **12 tickets** (chatty answers extracted via `re.findall(r"[a-z]+", text.lower())`)
  - `flagged_invalid_output`: **9 tickets** (unrecognized category or parse failure $\to$ routed to human review)

### 4. Zero-Shot vs Few-Shot Prompt Comparison (`compare_prompts.py`)

| Metric | Zero-Shot Pipeline | Few-Shot Pipeline | Delta ($\Delta$) |
|---|---|---|---|
| **Classified Tickets** | 147 | 147 | 0 |
| **Correct Classifications** | 126 / 147 | 147 / 147 | +21 tickets |
| **Accuracy on Classified** | **85.71%** | **100.00%** | **+14.29%** |
| **Total Tokens Consumed** | 17,044 | 38,884 | +21,840 tokens |
| **Tokens per Classified Ticket** | 115.95 | 264.52 | +148.57 tokens (2.28x) |

#### Zero-Shot Confusion Matrix (147 classified tickets):
```
predicted_category  billing  delivery  fraud  technical  Total
true_category                                                 
billing                  38         0      0          0     38
delivery                  0        36      0          0     36
fraud                    11         0     23          0     34
technical                10         0      0         29     39
Total                    59        36     23         29    147
```
*Error Breakdown*: Every zero-shot error misclassified fraud (11) or technical issues (10) as billing.
*Token Value Justification*: The extra 148.57 tokens/ticket for few-shot prompting are essential for fraud and technical queues, where misrouting a stolen credit card or app freeze as a billing inquiry violates critical 2-hour SLAs.

### 5. Token Budgeting & Triage Routing
- **Budget Demonstration** (`TOKEN_BUDGET=6000`, [`outputs/triage_budget_demo.csv`](file:///Users/prabhust/Documents/support-copilot/outputs/triage_budget_demo.csv)):
  - **101 tickets** ended `budget_exhausted` and safely fell back to human review.
- **Production Triage Routing** (`TOKEN_BUDGET=100000`, [`outputs/triage_results.csv`](file:///Users/prabhust/Documents/support-copilot/outputs/triage_results.csv)):
  - **Queue Breakdown**: `support_queue`: 102 | `security_team`: 34 | `human_review`: 17 | `premium_desk`: 11
  - **Priority Breakdown**: `NORMAL`: 119 | `HIGH`: 45

### 6. Policy Document Chunking & TF-IDF Retrieval
- Split 5 policy texts into 40-word chunks with 10-word overlap:
  - Total chunks: **15** (`refund_policy.txt`: 4, `delivery_policy.txt`: 3, `escalation_policy.txt`: 3, `payment_policy.txt`: 3, `privacy_policy.txt`: 2).
- **TF-IDF Grounding**: Term Frequency (TF) weights localized word importance, while Inverse Document Frequency (IDF) discounts ubiquitous words across chunks. Limitation: fails on semantic synonyms with zero lexical overlap (e.g., "reimbursement" vs "refund").

### 7. RAG Evaluation: Overlap 10 vs Overlap 0 Sensitivity (`evaluate_rag.py`)

| Evaluation Metric | Overlap = 10 Chunks | Overlap = 0 Chunks | Analysis |
|---|---|---|---|
| **Indexed Chunks** | 15 chunks | 13 chunks | Overlap creates 2 boundary-bridging chunks. |
| **Top-3 Retrieval Accuracy** | 12 / 12 (100.0%) | 12 / 12 (100.0%) | TF-IDF finds relevant documents consistently. |
| **Grounded Answer Correctness** | **12 / 12 (100.0%)** | **10 / 12 (83.33%)** | **2 questions fail under Overlap 0.** |
| **Fallback on Unanswerable** | 2 / 2 (100.0%) | 2 / 2 (100.0%) | Correctly triggers "I could not find..." |

#### Breakage Analysis (Overlap = 0 Failures):
1. *"What delivery time makes a completed order eligible for a full refund?"*: The qualifying clause (*"more than 45 minutes"*) was severed at the chunk boundary, causing the model to answer incomplete sentence fragments.
2. *"How soon is money returned when a payment fails but is deducted?"*: The threshold (*"48 hours"*) was split across boundaries, causing the answerer to synthesize unrelated lead approval clauses.

---

## Part D

### 1. Seven-Scenario ReAct Agent Results Table

Processed through [`react_agent.py`](file:///Users/prabhust/Documents/support-copilot/react_agent.py) (`run_agent`) and logged to [`outputs/agent_runs.jsonl`](file:///Users/prabhust/Documents/support-copilot/outputs/agent_runs.jsonl):

| Scenario ID | Status | Refund Amount | Model Calls | Tokens Used | Trajectory Summary |
|---|---|---|---|---|---|
| `S1` | **REFUND** | 128.25 | 5 | 897 | `get_order` → `search_policy` → `calculate_refund` → `issue_refund` → final answer |
| `S2` | **REFUND** | 135.00 | 5 | 897 | Automated late delivery refund under ₹1,000 threshold |
| `S3` | **ESCALATE** | 1999.00 | 5 | 884 | Refund > ₹1,000 escalated to human lead via `escalate_to_human` |
| `S4` | **DENIED** | 0.00 | 4 | 605 | Delivery was completed within 20 mins (< 45 min policy cutoff) |
| `S5` | **DENIED** | 0.00 | 3 | 337 | Order status was cancelled; denied after policy verification |
| `S6` | **NOT_FOUND** | 0.00 | 2 | 72 | Order `ORD-999999` not in DB; early error return from `get_order` |
| `S7` | **FALLBACK_MAX_ITERATIONS** | 0.00 | 6 | 3000 | Adversarial loop prompt safely terminated after 6 model calls |

*Verification*: Every refund amount matches `expected_refund_inr` within ₹0.01; `S1` takes exactly 5 model calls.

### 2. Safety Limit Fallback Demonstrations

| Demo Run | Scenario | Condition / Guardrail | Resulting Status | Model Calls | Steps Completed | Tokens Used | Human Handoff Summary |
|---|---|---|---|---|---|---|---|
| **Max Iterations** | `S7` | `max_iterations=6` | `FALLBACK_MAX_ITERATIONS` | 6 | 6 | 3000 | `FALLBACK_MAX_ITERATIONS: Maximum iterations (6) reached without final answer; handing off to human support lead.` |
| **Token Limit** | `S1` | `token_limit=200` | `FALLBACK_TOKEN_LIMIT` | 3 | 2 | 354 | `FALLBACK_TOKEN_LIMIT: Cumulative token limit of 200 exceeded (accumulated 354); handing off to human support lead.` |
| **Timeout Limit** | `S1` | `time_limit_seconds=0` | `FALLBACK_TIMEOUT` | 0 | 0 | 0 | `FALLBACK_TIMEOUT: Wall-clock limit of 0s reached (elapsed 0.0000s); handing off to human support lead.` |

- **Token Limit Abort**: In the `token_limit=200` demo, Call 1 used 38 tokens, Call 2 used 62 tokens (cumulative 100). Call 3 emitted a request for `calculate_refund` using 254 tokens (cumulative 354 > 200). The agent **aborted immediately without executing `calculate_refund`**, keeping state safe at 2 completed steps.
- **Immediate Timeout**: With `time_limit_seconds=0`, `elapsed >= 0` evaluates to `True` at the top of iteration 1 before any model call is dispatched, guaranteeing zero compute expenditure.

### 3. Resumable Crash & Recovery Demonstration (Scenario S1)
A simulated crash was triggered immediately after the second tool execution (after its checkpoint was written to `state/s1_crash_demo.json`), followed by recovery via `resume_agent('s1_crash_demo')`.

**Tools executed across the crash and resume, in exact chronological order:**
1. `get_order` *(Pre-crash execution, step 1: fetched details for `ORD-100048`)*
2. `search_policy` *(Pre-crash execution, step 2: retrieved late-delivery guidelines)*
   - *💥 SIMULATED CRASH: Process killed right after step 2 checkpoint persisted to disk.*
   - *🔄 RESUMED: `resume_agent('s1_crash_demo')` reloaded state from disk.*
3. `calculate_refund` *(Post-resume execution, step 3: verified ₹128.25 refund eligibility)*
4. `issue_refund` *(Post-resume execution, step 4: approved and logged refund to `outputs/refund_ledger.jsonl`)*

**Outcome**: Exactly 4 tool executions occurred across both halves of the run, zero tools were duplicated, and the agent completed with status **REFUND** (₹128.25) across 5 total model calls.

### 4. ReAct versus Plan-and-Execute Comparative Analysis

In our ReAct agent, each step is evaluated sequentially against intermediate tool observations. For happy paths like **S1**, ReAct takes **5 model calls (897 tokens)** because every action (`get_order`, `search_policy`, `calculate_refund`, `issue_refund`) requires a separate LLM round-trip. In contrast, **Plan-and-Execute** generates the entire pipeline upfront, needing only **1 planning call** (plus a final synthesis call), significantly reducing token overhead by not repeatedly resending the growing context.

However, Plan-and-Execute struggles with conditional branching. When faced with missing or invalid orders (**S6** `NOT_FOUND`), a static plan (`get_order` → `calculate_refund` → `issue_refund`) immediately fails at step 1, discarding subsequent planned actions and forcing an expensive replanning call. Similarly, for a cancelled order (**S5**), pre-planned refund calculations become dead weight. ReAct handles these cleanly by adapting immediately: halting at **2 calls (72 tokens)** for S6 and **3 calls (337 tokens)** for S5 without executing downstream refund tools.

**Design Choice**: I would choose **ReAct**. Financial workflows require strict grounding and safety guarantees before disbursing funds. ReAct’s dynamic early-stopping prevents speculative execution of sensitive financial tools (`issue_refund`, `escalate_to_human`) on invalid orders. While Plan-and-Execute saves tokens on predictable paths, ReAct's reactive guardrails, lower error-recovery latency on failure branches (S5/S6), and tight verification loop make it far safer for real-world automated refunds.

---

## Limitations and Next Steps

- **Post-Hoc Segment Evaluation**: The customer retention list is illustrative and not a leakage-free evaluation because the Premium customer segment flag is a post-hoc whole-window label derived from RFM clustering across the entire observation window (including the September window that defines the lapse target label). (The model's own features strictly use only orders before the cutoff, as required.)
- **Next Steps**:
  1. *Dynamic Pre-Cutoff Segmentation*: Refactor RFM clustering to compute segment features exclusively on pre-cutoff historical orders to prevent forward information leakage.
  2. *Cost-Sensitive Objective Functions*: Incorporate business costs (cost of outreach vs. customer lifetime value) directly into the classifier thresholding to optimize financial return rather than raw F1 score.
  3. *Randomized Holdout Testing*: Deploy retention campaigns in randomized A/B test cohorts to measure causal lift and incremental retention.
  4. *Hybrid Planning Engine*: Combine ReAct's step-level safety verification with deterministic state-machine routing to eliminate redundant model calls on static sub-tasks.

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## License

MIT
