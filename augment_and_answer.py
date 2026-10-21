"""augment_and_answer.py -- Retrieval-Augmented Generation (RAG) Policy Assistant.

Tasks (3 marks):
Write answer(question):
1. Retrieve three chunks: retrieve(question, k=3).
2. Send two messages to mock_llm.chat:
   - System message: contains the words "POLICY ASSISTANT", instructs model to answer
     only from supplied context and to say it could not find the information otherwise.
   - User message: exactly of the form:
     CONTEXT:
     [<source file name>] <chunk text>
     [<source file name>] <chunk text>
     [<source file name>] <chunk text>

     QUESTION:
     <question>
3. Return the model's text and the sources used.
"""

from typing import List, Tuple
from mock_llm import chat
from vectorize_and_retrieve import retrieve

SYSTEM_PROMPT = """You are a helpful POLICY ASSISTANT for an e-commerce support team.
Your task is to answer customer questions strictly using only the supplied policy context.
If the answer cannot be found in the provided context, state that you could not find that information in the policy documents."""


def answer(question: str) -> Tuple[str, List[str]]:
    """Retrieves 3 policy chunks and queries mock_llm in POLICY ASSISTANT mode.

    Parameters:
        question: Customer query string.

    Returns:
        (model_text, sources_used)
    """
    # 1. Retrieve top 3 relevant policy chunks
    chunks = retrieve(question, k=3)
    sources = [c["source"] for c in chunks]

    # 2. Format context block: each chunk prefixed with [<source file name>] on its own line
    context_lines = [f"[{c['source']}] {c['text']}" for c in chunks]
    context_block = "\n".join(context_lines)

    # User message of exact form specified:
    # CONTEXT:
    # <chunks>
    #
    # QUESTION:
    # <question>
    user_message = f"CONTEXT:\n{context_block}\n\nQUESTION:\n{question}"

    # 3. Two messages to mock_llm.chat
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    response = chat(messages)
    model_text = response.choices[0].message.content

    return model_text, sources


def main():
    print("=" * 80)
    print("  POLICY ASSISTANT RAG AUDIT")
    print("=" * 80)

    # Test cases: Grounded question + Out-of-scope question
    test_queries = [
        "What delivery time makes a completed order eligible for a full refund?",
        "Who must approve a refund above INR 1000?",
        "How fast must a fraud ticket reach the security team?",
        "What is the favourite colour of the company mascot?",
    ]

    for q in test_queries:
        ans_text, srcs = answer(q)
        print(f"\nQuestion: '{q}'")
        print(f"Sources:  {srcs}")
        print(f"Answer:   '{ans_text}'")

    print("\n" + "=" * 80)
    print("  ALL RAG INVOCATIONS COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
