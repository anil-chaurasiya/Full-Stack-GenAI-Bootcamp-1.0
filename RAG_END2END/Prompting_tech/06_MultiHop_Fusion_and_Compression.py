"""Advanced retrieval workflows: multi-hop, weighted RRF, and compression.

This is the final beginner-readable companion to the Class 37 advanced
retriever notebook. It keeps all retrieval steps inspectable and avoids API
keys. In a production system, an LLM can plan the next hop and compress text;
the offline functions below are clearly labelled deterministic teaching stand-ins.

Run: .venv_genai/bin/python RAG_END2END/Prompting_tech/06_MultiHop_Fusion_and_Compression.py
Needs: pip install numpy scikit-learn
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# %% [1] Small evidence corpus
@dataclass(frozen=True)
class Document:
    page_content: str
    metadata: dict[str, str | int]


DOCUMENTS = [
    Document("Llama 2-Chat is first trained with supervised fine-tuning using helpful demonstrations written by annotators.", {"chunk_id": "c1", "page": 8}),
    Document("Annotators compare candidate responses to collect preference data. Their choices create ranked pairs for reward-model training.", {"chunk_id": "c2", "page": 10}),
    Document("The reward model estimates which candidate answer a human would prefer. It supplies a learning signal for reinforcement learning.", {"chunk_id": "c3", "page": 12}),
    Document("RLHF optimizes Llama 2-Chat against the learned reward model to improve helpfulness and alignment with human preferences.", {"chunk_id": "c4", "page": 11}),
    Document("Safety work includes red teaming, safety evaluations, safety annotations, and iterative fine-tuning to reduce unsafe behaviour. The research paper also describes release sizes and evaluation setup in other sections.", {"chunk_id": "c5", "page": 20}),
]


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


# %% [2] Two retrievers: vector similarity and BM25-like keyword scoring
class VectorSearch:
    def __init__(self, documents: list[Document]):
        self.documents = documents
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.vectors = self.vectorizer.fit_transform([d.page_content for d in documents]).toarray()

    def search(self, query: str, k: int = 4) -> list[tuple[float, Document]]:
        vector = self.vectorizer.transform([query]).toarray()[0]
        if np.linalg.norm(vector) == 0:
            return []
        scores = self.vectors @ vector / (np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(vector))
        ids = np.argsort(scores)[::-1][:k]
        return [(float(scores[i]), self.documents[i]) for i in ids if scores[i] > 0]


class TinyBM25:
    """Minimal BM25 to make sparse scoring inspectable; use a library at scale."""

    def __init__(self, documents: list[Document], k1: float = 1.5, b: float = 0.75):
        self.documents, self.k1, self.b = documents, k1, b
        self.corpus = [tokens(d.page_content) for d in documents]
        self.lengths = [len(doc) for doc in self.corpus]
        self.average_length = sum(self.lengths) / len(self.lengths)
        frequency: Counter[str] = Counter()
        for document in self.corpus:
            frequency.update(set(document))
        total = len(documents)
        self.idf = {term: math.log(1 + (total - count + 0.5) / (count + 0.5)) for term, count in frequency.items()}

    def search(self, query: str, k: int = 4) -> list[tuple[float, Document]]:
        scores = np.zeros(len(self.documents))
        for document_number, document_tokens in enumerate(self.corpus):
            counts = Counter(document_tokens)
            for term in tokens(query):
                tf = counts[term]
                denominator = tf + self.k1 * (1 - self.b + self.b * self.lengths[document_number] / self.average_length)
                if tf:
                    scores[document_number] += self.idf.get(term, 0) * tf * (self.k1 + 1) / denominator
        ids = np.argsort(scores)[::-1][:k]
        return [(float(scores[i]), self.documents[i]) for i in ids if scores[i] > 0]


def display(title: str, results: list[tuple[float, Document]], score_name: str) -> None:
    print(f"\n{'=' * 92}\n{title}\n{'=' * 92}")
    for rank, (score, document) in enumerate(results, start=1):
        print(f"{rank}. {score_name}={score:.4f} | {document.metadata}")
        print("   ", document.page_content)


# %% [3] Multi-hop retrieval: evidence from hop one determines hop two
def plan_second_hop(original_query: str, hop_one: list[tuple[float, Document]]) -> str:
    """Offline stand-in for the notebook's LLM prompt that plans a next query.

    A real LLM should receive the original question and hop-one evidence, then
    return exactly one standalone query for information still missing. This
    deterministic function lets us focus on the state flow without an API key.
    """

    hop_one_text = " ".join(document.page_content.lower() for _, document in hop_one)
    if "reward model" in original_query.lower() or "reward model" in hop_one_text:
        return "How are reward models trained from human preference data?"
    return "How does reinforcement learning with human feedback use a reward model?"


def demonstrate_multi_hop(vector_search: VectorSearch) -> None:
    original_query = "How was Llama 2-Chat aligned with human preferences, and what role did reward models play?"
    hop_one = vector_search.search(original_query, k=2)
    display("1. Multi-hop: Hop 1 evidence", hop_one, "cosine")

    hop_two_query = plan_second_hop(original_query, hop_one)
    print("\nHop 2 query planned from Hop 1 evidence:\n", hop_two_query)
    hop_two = vector_search.search(hop_two_query, k=2)
    display("2. Multi-hop: Hop 2 evidence", hop_two, "cosine")

    evidence = {document.metadata["chunk_id"]: document for _, document in [*hop_one, *hop_two]}
    print("\nEvidence supplied to the final answer prompt (deduplicated):")
    for document in evidence.values():
        print("-", document.page_content)
    print("A final LLM prompt must instruct: answer only from this evidence; say insufficient if needed.")


# %% [4] Reciprocal Rank Fusion: combine rankings without mixing incompatible scores
def weighted_rrf(
    result_lists: list[list[tuple[float, Document]]],
    weights: list[float],
    k: int = 4,
    constant: int = 60,
) -> list[tuple[float, Document]]:
    """Weighted RRF: add ``weight / (constant + rank)`` for each ranked list.

    This is safer than directly adding a BM25 score and a cosine score: those
    scores use different ranges. The class notebook's EnsembleRetriever applies
    this ranking-based approach internally.
    """

    if len(result_lists) != len(weights):
        raise ValueError("Each result list needs one weight.")
    scores: dict[str, float] = {}
    documents: dict[str, Document] = {}
    for results, weight in zip(result_lists, weights):
        for rank, (_, document) in enumerate(results, start=1):
            chunk_id = str(document.metadata["chunk_id"])
            documents[chunk_id] = document
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (constant + rank)
    ids = sorted(scores, key=scores.get, reverse=True)[:k]
    return [(scores[chunk_id], documents[chunk_id]) for chunk_id in ids]


def demonstrate_fusion(vector_search: VectorSearch, bm25: TinyBM25) -> None:
    query = "How does Llama 2 improve safety through red teaming?"
    vector_results = vector_search.search(query)
    bm25_results = bm25.search(query)
    display("3. Vector results", vector_results, "cosine")
    display("4. BM25 keyword results", bm25_results, "BM25")
    fused = weighted_rrf([bm25_results, vector_results], weights=[0.4, 0.6])
    display("5. Weighted RRF (BM25=0.4, vector=0.6)", fused, "weighted RRF")


# %% [5] Contextual compression: reduce retrieved text before the final answer prompt
def rule_based_compress(query: str, document: Document) -> str:
    """Offline stand-in for LLMChainExtractor.

    It keeps sentences sharing a meaningful query word. An LLM compressor is
    more flexible, but must be evaluated carefully: it can remove essential
    qualifications or introduce unsupported wording.
    """

    query_words = set(tokens(query)) - {"how", "does", "what", "the", "and", "with"}
    sentences = re.split(r"(?<=[.!?])\s+", document.page_content)
    kept = [sentence for sentence in sentences if query_words.intersection(tokens(sentence))]
    return " ".join(kept) or "[No directly matching sentence found.]"


def demonstrate_compression(vector_search: VectorSearch) -> None:
    query = "What safety techniques reduce unsafe behaviour?"
    retrieved = vector_search.search(query, k=3)
    display("6. Before contextual compression", retrieved, "cosine")

    compressed = [
        (score, Document(rule_based_compress(query, document), document.metadata))
        for score, document in retrieved
    ]
    display("7. After rule-based contextual compression", compressed, "original cosine")
    before = sum(len(document.page_content) for _, document in retrieved)
    after = sum(len(document.page_content) for _, document in compressed)
    print(f"\nCharacters before={before}, after={after}, reduction={(1 - after / before) * 100:.1f}%")


def main() -> None:
    print("ADVANCED RAG WORKFLOWS: multi-hop, fusion, and compression")
    vector_search = VectorSearch(DOCUMENTS)
    bm25 = TinyBM25(DOCUMENTS)
    demonstrate_multi_hop(vector_search)
    demonstrate_fusion(vector_search, bm25)
    demonstrate_compression(vector_search)
    print("\nDebug order: first inspect evidence, then inspect ranking, and only then let an LLM draft an answer.")


if __name__ == "__main__":
    main()
