"""chunk_documents.py -- Document chunking for policy Q&A and retrieval.

Tasks (2 marks):
- Read the five .txt files in data/policies/.
- Split each into chunks of 40 words with an overlap of 10 words:
  * Chunk 1 covers words 0-39 (indices 0:40).
  * Next chunk starts 30 words later (step = 30), covering words 30-69 (indices 30:70).
  * Continue until the first chunk whose end reaches or passes the last word of the document.
- Record the source file name with every chunk.

Reference:
- 15 chunks in total:
  * refund: 4
  * delivery: 3
  * escalation: 3
  * payment: 3
  * privacy: 2
"""

import glob
import json
import os
from typing import Any, Dict, List
import pandas as pd

POLICIES_DIR = "data/policies"
CHUNKS_OUTPUT_CSV = "outputs/policy_chunks.csv"
CHUNKS_OUTPUT_JSON = "outputs/policy_chunks.json"

CHUNK_SIZE = 40
OVERLAP = 10
STEP = CHUNK_SIZE - OVERLAP  # 30 words


def chunk_text(
    text: str,
    source_file: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
) -> List[Dict[str, Any]]:
    """Splits text into sliding windows of words with specified overlap.

    Stops after the first chunk whose end reaches or passes the last word.
    """
    words = text.split()
    total_words = len(words)
    step = chunk_size - overlap
    chunks = []

    start = 0
    chunk_num = 1
    while start < total_words:
        end = min(start + chunk_size, total_words)
        chunk_words = words[start:end]
        chunk_text_str = " ".join(chunk_words)

        chunks.append({
            "source_file": source_file,
            "chunk_id": f"{source_file}#chunk_{chunk_num}",
            "chunk_index": chunk_num,
            "start_word": start,
            "end_word": end - 1,  # 0-indexed inclusive (e.g., 0-39)
            "word_count": len(chunk_words),
            "text": chunk_text_str,
        })

        if end >= total_words:
            break

        start += step
        chunk_num += 1

    return chunks


def chunk_all_policies(policies_dir: str = POLICIES_DIR) -> List[Dict[str, Any]]:
    """Reads all .txt files from policies directory and chunks them."""
    file_paths = sorted(glob.glob(os.path.join(policies_dir, "*.txt")))
    all_chunks = []

    for fpath in file_paths:
        fname = os.path.basename(fpath)
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()
        file_chunks = chunk_text(content, fname)
        all_chunks.extend(file_chunks)

    return all_chunks


def main():
    print("=" * 80)
    print("  POLICY DOCUMENTS CHUNKING")
    print("=" * 80)
    print(f"Directory: {POLICIES_DIR}")
    print(f"Chunk size: {CHUNK_SIZE} words | Overlap: {OVERLAP} words | Step: {STEP} words")
    print("=" * 80)

    chunks = chunk_all_policies(POLICIES_DIR)
    df_chunks = pd.DataFrame(chunks)

    # Save to outputs/
    os.makedirs("outputs", exist_ok=True)
    df_chunks.to_csv(CHUNKS_OUTPUT_CSV, index=False)
    with open(CHUNKS_OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)
    print(f"Saved {len(chunks)} chunks to {CHUNKS_OUTPUT_CSV} and {CHUNKS_OUTPUT_JSON}\n")

    # Group summary by policy file
    summary = df_chunks.groupby("source_file").size().to_dict()
    print("Summary of chunks per document:")
    for fname, count in summary.items():
        print(f"  - {fname:<25}: {count} chunks")
    print("-" * 50)
    print(f"  Total Chunks:              {len(chunks)} chunks")

    # Detailed chunk verification
    print("\n" + "=" * 80)
    print("  DETAILED CHUNK METRICS")
    print("=" * 80)
    for c in chunks:
        print(f"[{c['source_file']:<22}] Chunk #{c['chunk_index']:<2} | Words {c['start_word']:>3} to {c['end_word']:>3} ({c['word_count']:>2} words): '{c['text'][:60]}...'")

    # Assertions matching prompt requirements exactly
    assert len(chunks) == 15, f"Expected 15 chunks in total, got {len(chunks)}"
    assert summary.get("refund_policy.txt") == 4, f"Expected 4 refund chunks, got {summary.get('refund_policy.txt')}"
    assert summary.get("delivery_policy.txt") == 3, f"Expected 3 delivery chunks, got {summary.get('delivery_policy.txt')}"
    assert summary.get("escalation_policy.txt") == 3, f"Expected 3 escalation chunks, got {summary.get('escalation_policy.txt')}"
    assert summary.get("payment_policy.txt") == 3, f"Expected 3 payment chunks, got {summary.get('payment_policy.txt')}"
    assert summary.get("privacy_policy.txt") == 2, f"Expected 2 privacy chunks, got {summary.get('privacy_policy.txt')}"

    print("\n" + "=" * 80)
    print("  ALL CHUNKING SPECIFICATIONS & REFERENCE BENCHMARKS VERIFIED")
    print("=" * 80)


if __name__ == "__main__":
    main()
