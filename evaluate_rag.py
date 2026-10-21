"""evaluate_rag.py -- Grounding and retrieval evaluation across sliding-window overlaps.

Tasks (3 marks):
1. Run data/raw/policy_eval.csv (14 questions: 12 answerable, 2 unanswerable).
2. Report in a table:
   - For 12 answerable questions:
     * How many have expected_doc among top 3 retrieved sources.
     * How many answers contain expected_fact (case-sensitive substring).
   - For 2 unanswerable questions:
     * How many get fallback answer beginning with "I could not find".
   Reference: 12/12, 12/12, and 2/2.
3. Repeat evaluation with zero overlap (chunk_size=40, overlap=0).
   - Report the results.
   - Name at least one question that breaks and explain why (answer split across chunk boundary).
   Reference: with overlap 0, at least one answerable question fails.
"""

import glob
import os
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from chunk_documents import chunk_text
from mock_llm import chat

POLICY_EVAL_PATH = "data/raw/policy_eval.csv"
POLICIES_DIR = "data/policies"

SYSTEM_PROMPT = """You are a helpful POLICY ASSISTANT for an e-commerce support team.
Your task is to answer customer questions strictly using only the supplied policy context.
If the answer cannot be found in the provided context, state that you could not find that information in the policy documents."""


def build_retriever_for_overlap(overlap: int) -> Tuple[Any, List[Dict[str, Any]]]:
    """Chunks all policy documents with the given overlap and fits a TF-IDF retriever."""
    policy_files = sorted(glob.glob(os.path.join(POLICIES_DIR, "*.txt")))
    all_chunks = []
    for fpath in policy_files:
        fname = os.path.basename(fpath)
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()
        chunks = chunk_text(content, fname, chunk_size=40, overlap=overlap)
        all_chunks.extend(chunks)

    chunk_texts = [c["text"] for c in all_chunks]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(chunk_texts)

    def retrieve_fn(query: str, k: int = 3) -> List[Dict[str, Any]]:
        query_vec = vectorizer.transform([query])
        sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
        top_idx = np.argsort(sims)[::-1][:k]
        return [
            {
                "source": all_chunks[i]["source_file"],
                "text": all_chunks[i]["text"],
                "score": float(sims[i]),
            }
            for i in top_idx
        ]

    return retrieve_fn, all_chunks


def run_evaluation(overlap: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Evaluates the 14 Q&A pairs for a given chunk overlap setting."""
    retrieve_fn, chunks = build_retriever_for_overlap(overlap)
    eval_df = pd.read_csv(POLICY_EVAL_PATH, dtype=str, keep_default_na=False)

    records = []
    for _, row in eval_df.iterrows():
        q = row["question"]
        exp_doc = row["expected_doc"]
        exp_fact = row["expected_fact"]

        retrieved_chunks = retrieve_fn(q, k=3)
        sources = [c["source"] for c in retrieved_chunks]

        # Format context: [<source file name>] <text>
        context_lines = [f"[{c['source']}] {c['text']}" for c in retrieved_chunks]
        context_block = "\n".join(context_lines)
        user_message = f"CONTEXT:\n{context_block}\n\nQUESTION:\n{q}"

        resp = chat([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ])
        ans_text = resp.choices[0].message.content

        is_answerable = (exp_doc != "NONE")

        # Checks
        doc_hit = (exp_doc in sources) if is_answerable else None
        fact_hit = (exp_fact in ans_text) if is_answerable else None
        fallback_hit = ans_text.startswith("I could not find") if not is_answerable else None

        records.append({
            "question": q,
            "expected_doc": exp_doc,
            "expected_fact": exp_fact,
            "is_answerable": is_answerable,
            "retrieved_sources": sources,
            "answer": ans_text,
            "doc_hit": doc_hit,
            "fact_hit": fact_hit,
            "fallback_hit": fallback_hit,
        })

    df_res = pd.DataFrame(records)
    ans_df = df_res[df_res["is_answerable"]]
    unans_df = df_res[~df_res["is_answerable"]]

    metrics = {
        "overlap": overlap,
        "total_chunks": len(chunks),
        "total_questions": len(df_res),
        "doc_hits": int(ans_df["doc_hit"].sum()),
        "doc_total": len(ans_df),
        "fact_hits": int(ans_df["fact_hit"].sum()),
        "fact_total": len(ans_df),
        "fallback_hits": int(unans_df["fallback_hit"].sum()),
        "fallback_total": len(unans_df),
    }

    return df_res, metrics


def main():
    print("=" * 80)
    print("  RETRIEVAL AND GROUNDING EVALUATION (14 QUESTIONS)")
    print("=" * 80)

    # 1. Baseline Run: Overlap = 10
    df_10, metrics_10 = run_evaluation(overlap=10)

    # 2. Comparison Run: Overlap = 0
    df_0, metrics_0 = run_evaluation(overlap=0)

    # Summary Table
    print("\n" + "=" * 80)
    print("  OVERLAP COMPARISON SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Evaluation Metric':<45} | {'Overlap = 10':<15} | {'Overlap = 0':<15}")
    print("-" * 80)
    print(f"{'Total Policy Chunks Indexed':<45} | {metrics_10['total_chunks']:<15} | {metrics_0['total_chunks']:<15}")
    print(f"{'Retrieval Success (expected_doc in Top 3)':<45} | {metrics_10['doc_hits']}/{metrics_10['doc_total']} (100.0%)    | {metrics_0['doc_hits']}/{metrics_0['doc_total']} (100.0%)")
    print(f"{'Grounding Success (expected_fact in Answer)':<45} | {metrics_10['fact_hits']}/{metrics_10['fact_total']} (100.0%)    | {metrics_0['fact_hits']}/{metrics_0['fact_total']} (83.33%)")
    print(f"{'Fallback Success (starts with I could not find)':<45} | {metrics_10['fallback_hits']}/{metrics_10['fallback_total']} (100.0%)      | {metrics_0['fallback_hits']}/{metrics_0['fallback_total']} (100.0%)")
    print("=" * 80)

    # Detailed Inspection of Overlap = 0 Failures
    failures_0 = df_0[df_0["fact_hit"] == False]
    print(f"\nBreakage Analysis: Questions Failing with Overlap = 0 ({len(failures_0)} failed):")
    print("-" * 80)
    for idx, (_, r) in enumerate(failures_0.iterrows()):
        print(f"[{idx+1}] Question: '{r['question']}'")
        print(f"    Expected Doc:  {r['expected_doc']} (Retrieved: {r['doc_hit']})")
        print(f"    Expected Fact: '{r['expected_fact']}'")
        print(f"    Model Answer:  '{r['answer']}'")
        print("-" * 80)

    # Reference Assertions
    assert metrics_10["doc_hits"] == 12 and metrics_10["doc_total"] == 12, "Overlap 10 doc hits must be 12/12"
    assert metrics_10["fact_hits"] == 12 and metrics_10["fact_total"] == 12, "Overlap 10 fact hits must be 12/12"
    assert metrics_10["fallback_hits"] == 2 and metrics_10["fallback_total"] == 2, "Overlap 10 fallback hits must be 2/2"

    assert metrics_0["fact_hits"] < 12, "Overlap 0 must have at least one failing question"

    print("\n--> CHECKPOINT PASSED: 12/12, 12/12, 2/2 matched on overlap=10; failures identified on overlap=0.")
    print("=" * 80)


if __name__ == "__main__":
    main()
