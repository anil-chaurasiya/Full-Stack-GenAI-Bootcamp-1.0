"""Advanced query prompting for RAG: HyDE and Multi-Query Retrieval.

Based on ``Class-37-08-Aug-2026-prompting/retriever_advance_gemini.ipynb``.

The notebook uses Gemini to generate a hypothetical document (HyDE) and query
variants (MultiQueryRetriever). This teaching version runs without an API key:
it uses deliberately written stand-ins so you can inspect the retrieval flow.
Replace the two clearly marked functions with an LLM call only after the local
flow makes sense.

Run: .venv_genai/bin/python RAG_END2END/Prompting_tech/04_HyDE_and_MultiQuery_Retrieval.py
Needs: pip install numpy scikit-learn
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# %% [1] Tiny corpus: normally these are chunks returned by your vector database
@dataclass(frozen=True)
class Document:
    page_content: str
    metadata: dict[str, str | int]


DOCUMENTS = [
    Document("Llama 2-Chat first undergoes supervised fine-tuning using demonstrations written by annotators.", {"chunk_id": "c1", "section": "fine_tuning", "page": 8}),
    Document("Human preference data is collected by showing annotators candidate responses and asking which response is better.", {"chunk_id": "c2", "section": "fine_tuning", "page": 10}),
    Document("A reward model learns from ranked response pairs and estimates which answer humans prefer.", {"chunk_id": "c3", "section": "fine_tuning", "page": 12}),
    Document("Reinforcement learning with human feedback uses the reward model to improve helpfulness and alignment of Llama 2-Chat.", {"chunk_id": "c4", "section": "fine_tuning", "page": 11}),
    Document("Safety work includes red teaming, safety-specific annotations, evaluations, and iterative fine-tuning.", {"chunk_id": "c5", "section": "safety", "page": 20}),
]


# %% [2] A transparent in-memory vector search baseline
class TfidfSearch:
    """Small teaching-only substitute for Chroma + an embedding model."""

    def __init__(self, documents: list[Document]):
        self.documents = documents
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.document_vectors = self.vectorizer.fit_transform(
            [document.page_content for document in documents]
        ).toarray()

    def search(self, query: str, k: int = 3) -> list[tuple[float, Document]]:
        query_vector = self.vectorizer.transform([query]).toarray()[0]
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            return []
        document_norms = np.linalg.norm(self.document_vectors, axis=1)
        scores = self.document_vectors @ query_vector / (document_norms * query_norm)
        indices = np.argsort(scores)[::-1][:k]
        return [(float(scores[index]), self.documents[index]) for index in indices if scores[index] > 0]


def display(title: str, results: list[tuple[float, Document]]) -> None:
    print(f"\n{'=' * 84}\n{title}\n{'=' * 84}")
    if not results:
        print("No results. This is useful feedback: the query uses no corpus vocabulary.")
    for rank, (score, document) in enumerate(results, start=1):
        print(f"{rank}. score={score:.3f} | {document.metadata}")
        print("   ", document.page_content)


# %% [3] HyDE: use an answer-shaped passage for retrieval, not as the final answer
def generate_hypothetical_document(query: str) -> str:
    """Offline stand-in for an LLM HyDE prompt.

    In production, ask an LLM: "Write a concise passage that would answer this
    question; do not claim it is factual." Embed that generated passage, then
    retrieve REAL source documents. Never present the hypothetical passage as
    evidence or as the final answer.
    """

    del query  # The fixed text keeps this API-free classroom demonstration reproducible.
    return (
        "Llama 2-Chat alignment combines supervised fine-tuning, human preference "
        "comparisons, a reward model trained on ranked answers, and reinforcement "
        "learning with human feedback to improve helpfulness and safety."
    )


def demonstrate_hyde(search: TfidfSearch) -> None:
    # The wording is intentionally vague. Normal lexical retrieval has less to match.
    user_query = "How did people shape the assistant's behavior after initial training?"
    normal_results = search.search(user_query)
    display("1. Normal retrieval using the user query", normal_results)

    hypothetical_document = generate_hypothetical_document(user_query)
    print("\nHyDE generated passage (for SEARCH ONLY):\n", hypothetical_document)
    hyde_results = search.search(hypothetical_document)
    display("2. HyDE retrieval using the hypothetical passage", hyde_results)


# %% [4] Multi-query: search several meanings/wordings, then combine unique evidence
def generate_query_variants(original_query: str) -> list[str]:
    """Offline stand-in for an LLM query-expansion prompt.

    An LLM-powered version should output only alternative search queries, preserve
    entities and intent, and include the original query. Bad variants change the
    question and can introduce irrelevant context.
    """

    return [
        original_query,
        "How was Llama 2-Chat aligned with human preferences?",
        "How were reward models trained from ranked responses?",
        "How did reinforcement learning with human feedback improve Llama 2-Chat?",
    ]


def deduplicate(results: list[tuple[float, Document]]) -> list[tuple[float, Document]]:
    """Keep the best score for each chunk id, then sort once at the end."""

    best_by_id: dict[str, tuple[float, Document]] = {}
    for score, document in results:
        chunk_id = str(document.metadata["chunk_id"])
        if chunk_id not in best_by_id or score > best_by_id[chunk_id][0]:
            best_by_id[chunk_id] = (score, document)
    return sorted(best_by_id.values(), key=lambda item: item[0], reverse=True)


def demonstrate_multi_query(search: TfidfSearch) -> None:
    original_query = "How did human feedback improve Llama 2-Chat?"
    queries = generate_query_variants(original_query)
    combined_results: list[tuple[float, Document]] = []

    print("\nGenerated search queries:")
    for number, query in enumerate(queries, start=1):
        print(f"{number}. {query}")
        combined_results.extend(search.search(query, k=2))

    display("3. Multi-query retrieval after de-duplication", deduplicate(combined_results))


def main() -> None:
    search = TfidfSearch(DOCUMENTS)
    print("ADVANCED QUERY PROMPTING: HyDE and Multi-Query Retrieval")
    demonstrate_hyde(search)
    demonstrate_multi_query(search)
    print("\nDebug checklist: inspect generated queries, inspect retrieved source chunks, "
          "and evaluate whether recall improved before adding costlier LLM calls.")


if __name__ == "__main__":
    main()
