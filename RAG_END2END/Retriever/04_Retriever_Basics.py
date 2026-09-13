"""
Retrievers: a small, debuggable RAG retrieval lesson
====================================================

This is a beginner-friendly Python version of the ideas in
``Class-36-29-July-5-Aug-2026-Retriever/retriever.ipynb``.

The previous lesson built the indexing foundation:

    documents -> chunks -> embeddings/vectors

This lesson starts with already-created chunks and teaches the next stage:

    user query -> query vector -> retriever -> relevant source chunks

It demonstrates, in increasing complexity:

1. Vector similarity retrieval (the usual starting point)
2. Score thresholds (avoid returning weak matches)
3. Metadata pre-filtering (search only a permitted section)
4. MMR / Maximal Marginal Relevance (relevance + diversity)
5. BM25 sparse keyword retrieval
6. Hybrid retrieval using Reciprocal Rank Fusion (RRF)

The classroom notebook uses OpenAI embeddings and Chroma. This file starts
with TF-IDF vectors and an in-memory index so it runs without an API key,
database, or model download. TF-IDF is a *lexical* vector baseline, not a
neural semantic embedding model. Add ``--semantic`` to use the free local
Sentence Transformers model instead (the first run can download the model).

Run from the repository root:

    .venv_genai/bin/python RAG_END2END/Retriever/04_Retriever_Basics.py

Useful variations:

    .venv_genai/bin/python RAG_END2END/Retriever/04_Retriever_Basics.py \
        --query "How did human feedback align Llama 2-Chat?"

    .venv_genai/bin/python RAG_END2END/Retriever/04_Retriever_Basics.py --semantic

Packages:
    pip install numpy scikit-learn
    pip install sentence-transformers  # only for --semantic

For VS Code debugging, set a breakpoint in a ``# %%`` section, select the
``.venv_genai`` interpreter, and press F5.  The printed output deliberately
shows chunk ids, metadata, and scores because retrievers should be inspected
before sending their results to an LLM.
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# %% [1] A tiny indexed corpus: these Documents stand in for PDF chunks
@dataclass(frozen=True)
class LessonDocument:
    """The two parts of every useful RAG chunk: text and traceable metadata."""

    page_content: str
    metadata: dict[str, Any]


# In a real RAG system these would come from a loader + text splitter.
# The metadata mirrors the Llama 2 paper metadata used in the class notebook.
DOCUMENTS = [
    LessonDocument(
        "Llama 2 is a collection of pretrained and fine-tuned large language "
        "models released by Meta. The released models have 7 billion, 13 billion, "
        "and 70 billion parameters.",
        {"chunk_id": "llama2-01", "paper_page": 1, "section": "introduction", "year": 2023, "organization": "Meta"},
    ),
    LessonDocument(
        "The largest Llama 2 models use grouped-query attention, also called GQA. "
        "GQA improves inference scalability by reducing the key-value cache and "
        "allowing more efficient decoding than multi-head attention.",
        {"chunk_id": "llama2-02", "paper_page": 4, "section": "pretraining", "year": 2023, "organization": "Meta"},
    ),
    LessonDocument(
        "Llama 2-Chat starts with supervised fine-tuning. Annotators write helpful "
        "responses to prompts, and the model learns from these demonstration pairs.",
        {"chunk_id": "llama2-03", "paper_page": 8, "section": "fine_tuning", "year": 2023, "organization": "Meta"},
    ),
    LessonDocument(
        "After supervised fine-tuning, Meta collected preference data by asking "
        "annotators to compare candidate answers. The preferences train a reward "
        "model that estimates which answer people prefer.",
        {"chunk_id": "llama2-04", "paper_page": 10, "section": "fine_tuning", "year": 2023, "organization": "Meta"},
    ),
    LessonDocument(
        "Reinforcement learning with human feedback (RLHF) uses the reward model "
        "to further optimize Llama 2-Chat. The goal is to make responses more "
        "helpful while following human preferences.",
        {"chunk_id": "llama2-05", "paper_page": 11, "section": "fine_tuning", "year": 2023, "organization": "Meta"},
    ),
    LessonDocument(
        "Safety work for Llama 2-Chat included safety-specific data annotation, "
        "red teaming, safety evaluations, and iterative fine-tuning. Red teaming "
        "looks for prompts that can cause unsafe model behaviour.",
        {"chunk_id": "llama2-06", "paper_page": 20, "section": "safety", "year": 2023, "organization": "Meta"},
    ),
    LessonDocument(
        "A reward model is trained from ranked pairs of responses. Given one prompt "
        "and two answers, it should assign a higher score to the answer selected by "
        "human annotators.",
        {"chunk_id": "llama2-07", "paper_page": 12, "section": "fine_tuning", "year": 2023, "organization": "Meta"},
    ),
    LessonDocument(
        "A retrieval-augmented generation system searches a vector store for source "
        "chunks before asking an LLM to answer. Good chunk metadata makes the final "
        "answer traceable to a source and page.",
        {"chunk_id": "rag-01", "paper_page": 1, "section": "rag_background", "year": 2026, "organization": "Course"},
    ),
]


# %% [2] Inspection helpers: never treat retrieval as a black box
def print_documents(
    title: str,
    ranked_documents: list[tuple[float, LessonDocument]],
    score_name: str,
    max_characters: int = 360,
) -> None:
    """Print both score and provenance so results can be checked by a human."""

    print(f"\n{'=' * 94}\n{title}\n{'=' * 94}")
    if not ranked_documents:
        print("No documents were returned. This can be correct for a strict threshold/filter.")
        return

    for rank, (score, document) in enumerate(ranked_documents, start=1):
        metadata = document.metadata
        preview = " ".join(document.page_content.split())[:max_characters]
        print(
            f"\nRank {rank} | {score_name}={score:.4f} | "
            f"id={metadata['chunk_id']} | page={metadata['paper_page']} | "
            f"section={metadata['section']}"
        )
        print(preview)


def tokenize(text: str) -> list[str]:
    """A transparent tokenizer used by the hand-written BM25 implementation."""

    return re.findall(r"[a-z0-9]+", text.lower())


def cosine_scores(query_vector: np.ndarray, document_vectors: np.ndarray) -> np.ndarray:
    """Calculate one cosine similarity score for every document vector.

    Cosine similarity compares vector direction, not just magnitude. Larger is
    more similar. With normalized Sentence Transformer vectors, dot product and
    cosine similarity produce the same score.
    """

    query_norm = np.linalg.norm(query_vector)
    document_norms = np.linalg.norm(document_vectors, axis=1)
    if query_norm == 0:
        raise ValueError("The query produced a zero vector. Try a more specific query.")
    return (document_vectors @ query_vector) / (document_norms * query_norm)


# %% [3] Vector index: replaces Chroma only for this small teaching example
@dataclass
class VectorIndex:
    """Stores vectors and the exact function needed to embed future queries."""

    documents: list[LessonDocument]
    vectors: np.ndarray
    embed_query: Callable[[str], np.ndarray]
    embedding_name: str


def create_tfidf_index(documents: list[LessonDocument]) -> VectorIndex:
    """Create no-download lexical vectors.

    TF-IDF is useful for learning the retrieval mechanics and exact terms such
    as ``GQA``. It cannot reliably understand a paraphrase with different words.
    """

    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([document.page_content for document in documents]).toarray()

    def embed_query(query: str) -> np.ndarray:
        return vectorizer.transform([query]).toarray()[0]

    return VectorIndex(
        documents=documents,
        vectors=vectors,
        embed_query=embed_query,
        embedding_name="TF-IDF lexical vector baseline",
    )


def create_semantic_index(documents: list[LessonDocument]) -> VectorIndex:
    """Create real local semantic vectors using the same model for both phases."""

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "Semantic mode needs sentence-transformers. Run: pip install sentence-transformers"
        ) from error

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    vectors = np.asarray(
        model.encode(
            [document.page_content for document in documents],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    )

    def embed_query(query: str) -> np.ndarray:
        return np.asarray(model.encode(query, normalize_embeddings=True))

    return VectorIndex(documents, vectors, embed_query, model_name)


# %% [4] Similarity retrieval and a relevance-score threshold
def similarity_search(
    index: VectorIndex,
    query: str,
    k: int,
    metadata_filter: Callable[[LessonDocument], bool] | None = None,
) -> list[tuple[float, LessonDocument]]:
    """Return the k closest vectors, optionally filtering BEFORE ranking.

    This is the key behaviour of a metadata pre-filter: the retriever considers
    only eligible documents. It is better than retrieving an arbitrary top k and
    throwing ineligible results away afterwards.
    """

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    eligible_indices = [
        number
        for number, document in enumerate(index.documents)
        if metadata_filter is None or metadata_filter(document)
    ]
    if not eligible_indices:
        return []

    query_vector = index.embed_query(query)
    scores = cosine_scores(query_vector, index.vectors)
    sorted_indices = sorted(eligible_indices, key=lambda number: scores[number], reverse=True)
    return [(float(scores[number]), index.documents[number]) for number in sorted_indices[:k]]


def apply_score_threshold(
    results: list[tuple[float, LessonDocument]], threshold: float
) -> list[tuple[float, LessonDocument]]:
    """Remove weak matches. Tune thresholds on real evaluation questions."""

    return [(score, document) for score, document in results if score >= threshold]


# %% [5] MMR: diversify a set of otherwise very similar results
def mmr_search(
    index: VectorIndex,
    query: str,
    k: int,
    fetch_k: int,
    lambda_mult: float,
) -> list[tuple[float, LessonDocument]]:
    """Return relevant but non-duplicate chunks using Maximal Marginal Relevance.

    MMR score = lambda * query_relevance - (1 - lambda) * similarity_to_selected

    ``lambda_mult=1`` behaves like pure relevance. Values closer to zero favour
    diversity. ``fetch_k`` must exceed ``k`` because MMR needs extra candidates
    from which to select diverse results.
    """

    if not 0 <= lambda_mult <= 1:
        raise ValueError("lambda_mult must be between 0 and 1.")
    if not 1 <= k <= fetch_k:
        raise ValueError("Expected 1 <= k <= fetch_k.")

    query_vector = index.embed_query(query)
    relevance = cosine_scores(query_vector, index.vectors)
    candidate_indices = list(np.argsort(relevance)[::-1][:fetch_k])
    selected_indices: list[int] = []
    selected_scores: list[float] = []

    while candidate_indices and len(selected_indices) < k:
        if not selected_indices:
            chosen = candidate_indices[0]
            mmr_score = float(relevance[chosen])
        else:
            def score_candidate(candidate: int) -> float:
                # Document-to-document cosine similarity is the diversity penalty.
                candidate_vector = index.vectors[candidate]
                selected_vectors = index.vectors[selected_indices]
                duplicate_similarity = max(cosine_scores(candidate_vector, selected_vectors))
                return float(lambda_mult * relevance[candidate] - (1 - lambda_mult) * duplicate_similarity)

            chosen = max(candidate_indices, key=score_candidate)
            mmr_score = score_candidate(chosen)

        candidate_indices.remove(chosen)
        selected_indices.append(chosen)
        selected_scores.append(mmr_score)

    return [(score, index.documents[number]) for score, number in zip(selected_scores, selected_indices)]


# %% [6] BM25: sparse keyword retrieval, written out instead of hidden in a library
class BM25Index:
    """A minimal BM25 implementation for learning; production code can use a library."""

    def __init__(self, documents: list[LessonDocument], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.tokenized_documents = [tokenize(document.page_content) for document in documents]
        self.document_lengths = [len(tokens) for tokens in self.tokenized_documents]
        self.average_document_length = sum(self.document_lengths) / len(self.document_lengths)

        document_frequency: Counter[str] = Counter()
        for tokens in self.tokenized_documents:
            document_frequency.update(set(tokens))
        total_documents = len(documents)
        self.idf = {
            term: math.log(1 + (total_documents - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def score(self, query: str) -> np.ndarray:
        """Score documents by query term frequency, rarity, and document length."""

        scores = np.zeros(len(self.documents))
        for document_number, tokens in enumerate(self.tokenized_documents):
            frequencies = Counter(tokens)
            for term in tokenize(query):
                term_frequency = frequencies[term]
                if term_frequency == 0:
                    continue
                denominator = term_frequency + self.k1 * (
                    1 - self.b + self.b * self.document_lengths[document_number] / self.average_document_length
                )
                scores[document_number] += self.idf.get(term, 0.0) * term_frequency * (self.k1 + 1) / denominator
        return scores

    def search(self, query: str, k: int) -> list[tuple[float, LessonDocument]]:
        scores = self.score(query)
        best_indices = np.argsort(scores)[::-1][:k]
        return [(float(scores[number]), self.documents[number]) for number in best_indices if scores[number] > 0]


# %% [7] Hybrid retrieval: combine rankings, not incomparable raw scores
def reciprocal_rank_fusion(
    result_lists: list[list[tuple[float, LessonDocument]]],
    k: int,
    rrf_constant: int = 60,
) -> list[tuple[float, LessonDocument]]:
    """Fuse ranked lists with RRF.

    RRF adds ``1 / (rrf_constant + rank)`` for each list in which a document
    appears. We use ranks rather than raw scores because BM25 and cosine scores
    use different scales and cannot be safely added together.
    """

    fused_scores: dict[str, float] = {}
    documents_by_id: dict[str, LessonDocument] = {}
    for results in result_lists:
        for rank, (_, document) in enumerate(results, start=1):
            chunk_id = str(document.metadata["chunk_id"])
            documents_by_id[chunk_id] = document
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1 / (rrf_constant + rank)

    ordered_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)[:k]
    return [(fused_scores[chunk_id], documents_by_id[chunk_id]) for chunk_id in ordered_ids]


# %% [8] Main lesson: run every core retrieval strategy on one question
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Learn the retrieval step of RAG.")
    parser.add_argument(
        "--query",
        default="How did Meta use human feedback to align Llama 2-Chat?",
        help="Question sent to every retriever.",
    )
    parser.add_argument(
        "--semantic",
        action="store_true",
        help="Use Sentence Transformers instead of the no-download TF-IDF baseline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    print("RAG RETRIEVER FOUNDATION: query -> search -> source chunks")
    print(f"Indexed chunks: {len(DOCUMENTS)}")

    if args.semantic:
        print("Creating local semantic embeddings; first use may download a model...")
        vector_index = create_semantic_index(DOCUMENTS)
    else:
        vector_index = create_tfidf_index(DOCUMENTS)
    bm25_index = BM25Index(DOCUMENTS)

    print("Embedding/vector method:", vector_index.embedding_name)
    print("Vector matrix shape:", vector_index.vectors.shape)
    print("\nQuery:", args.query)

    # 1. The standard retriever: nearest neighbours by vector similarity.
    try:
        similarity_results = similarity_search(vector_index, args.query, k=4)
    except ValueError as error:
        # This commonly happens with TF-IDF when none of the query words occur
        # in the corpus. It is a useful demonstration of why semantic models,
        # synonyms, or query expansion can improve recall.
        print(f"\nRetrieval cannot score this query: {error}")
        print("Try words present in the chunks, use --semantic, or add query rewriting.")
        return
    print_documents("1. Vector similarity retrieval", similarity_results, "cosine similarity")

    # 2. A threshold may return fewer than k results. Do not invent evidence.
    threshold = 0.13 if not args.semantic else 0.25
    threshold_results = apply_score_threshold(similarity_results, threshold)
    print_documents(
        f"2. Similarity threshold retrieval (threshold={threshold})",
        threshold_results,
        "cosine similarity",
    )

    # 3. Filter *before* ranking when you know the correct source constraints.
    fine_tuning_only = lambda document: document.metadata["section"] == "fine_tuning"
    filtered_results = similarity_search(vector_index, args.query, k=4, metadata_filter=fine_tuning_only)
    print_documents("3. Metadata pre-filter: section=fine_tuning", filtered_results, "cosine similarity")

    # 4. MMR avoids returning three chunks that say nearly the same thing.
    mmr_results = mmr_search(vector_index, args.query, k=4, fetch_k=7, lambda_mult=0.55)
    print_documents("4. MMR retrieval (relevance + diversity)", mmr_results, "MMR selection score")

    # 5. BM25 is excellent when the question contains exact identifiers/terms.
    bm25_results = bm25_index.search(args.query, k=4)
    print_documents("5. BM25 sparse keyword retrieval", bm25_results, "BM25 score")

    # 6. Hybrid combines vector and BM25 rankings using RRF.
    hybrid_results = reciprocal_rank_fusion([similarity_results, bm25_results], k=4)
    print_documents("6. Hybrid retrieval: vector + BM25 + RRF", hybrid_results, "RRF score")

    print("\nWhat comes after this lesson?")
    print("- Query rewriting / expansion: use an LLM to improve an ambiguous query.")
    print("- HyDE: use an LLM to make a hypothetical answer for retrieval only.")
    print("- Reranking: retrieve many candidates, then use a cross-encoder to rank them precisely.")
    print("Use these only after evaluating the simpler retrievers above; each adds cost and latency.")


if __name__ == "__main__":
    main()
