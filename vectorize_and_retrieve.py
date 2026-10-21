"""vectorize_and_retrieve.py -- TF-IDF Vectorization and Cosine Similarity Retrieval.

Tasks (3 marks):
1. Fit scikit-learn's TfidfVectorizer() on the policy chunk texts.
2. Transform a query with the same fitted vectorizer and rank chunks by cosine_similarity.
3. Implement retrieve(query, k=3) returning a list of dictionaries with:
   {"source": ..., "score": ..., "text": ...}
4. Explain in two sentences what TF and IDF each measure and one limitation of TF-IDF for this task.

Reference:
- For the query "refund policy late delivery", the top three chunks all come from refund_policy.txt.
"""

from typing import Any, Dict, List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from chunk_documents import chunk_all_policies


class PolicyRetriever:
    """TF-IDF Policy Chunk Retriever using Cosine Similarity."""

    def __init__(self, policies_dir: str = "data/policies"):
        self.chunks = chunk_all_policies(policies_dir)
        self.chunk_texts = [c["text"] for c in self.chunks]
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.chunk_texts)

    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Transforms query and returns top-k chunks ranked by cosine similarity.

        Returns list of dicts with keys: 'source', 'score', 'text'
        """
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Rank in descending order of similarity score
        top_indices = np.argsort(similarities)[::-1][:k]

        results = []
        for idx in top_indices:
            results.append({
                "source": self.chunks[idx]["source_file"],
                "score": float(similarities[idx]),
                "text": self.chunks[idx]["text"],
            })
        return results


# Module-level convenience instance and function
_retriever_instance = None


def get_default_retriever() -> PolicyRetriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = PolicyRetriever()
    return _retriever_instance


def retrieve(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """Convenience retrieve function matching specification."""
    retriever = get_default_retriever()
    return retriever.retrieve(query, k=k)


def main():
    print("=" * 80)
    print("  TF-IDF POLICY RETRIEVAL AUDIT")
    print("=" * 80)

    query = "refund policy late delivery"
    print(f"Test Query: '{query}' (k=3)")

    top_3 = retrieve(query, k=3)

    print("\nTop 3 Retrieved Chunks:")
    print("-" * 80)
    for idx, item in enumerate(top_3):
        print(f"Rank {idx+1}:")
        print(f"  Source: {item['source']}")
        print(f"  Score:  {item['score']:.4f}")
        print(f"  Text:   {item['text']}")
        print("-" * 80)

    sources = [item["source"] for item in top_3]
    print(f"\nRetrieved Sources: {sources}")

    assert len(top_3) == 3, f"Expected 3 results, got {len(top_3)}"
    assert all(s == "refund_policy.txt" for s in sources), (
        f"Expected all 3 chunks from refund_policy.txt, got {sources}"
    )

    print("\n--> CHECKPOINT PASSED: All top 3 chunks originate from refund_policy.txt.")
    print("=" * 80)


if __name__ == "__main__":
    main()
