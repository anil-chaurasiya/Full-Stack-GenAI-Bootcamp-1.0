"""
Chunking and embeddings: a small, debuggable RAG indexing lesson
================================================================

This file turns the most important ideas from the classroom notebooks in
``Class-35-26-July-2026-Chuking-Retriever`` into one normal Python program.
It deliberately stops *before* a vector database and an LLM.  That lets you
see and debug the foundation of RAG first:

    raw document -> chunks -> vectors -> similarity search -> useful context

Run from the repository root:

    .venv_genai/bin/python RAG_END2END/Chunking_embedding/03_Chunking_and_Embedding_Basics.py

Useful variations:

    # Extract and index the class Llama 2 PDF (requires: pip install pypdf)
    .venv_genai/bin/python RAG_END2END/Chunking_embedding/03_Chunking_and_Embedding_Basics.py --pdf

    # Index your own readable PDF
    .venv_genai/bin/python RAG_END2END/Chunking_embedding/03_Chunking_and_Embedding_Basics.py --pdf path/to/file.pdf

    # Use a real local semantic embedding model. The first run may download it.
    .venv_genai/bin/python RAG_END2END/Chunking_embedding/03_Chunking_and_Embedding_Basics.py --semantic

For VS Code debugging:
    1. Select the ``.venv_genai`` interpreter.
    2. Put a breakpoint in a ``# %%`` section or inside a function.
    3. Press F5, or run a section in the Interactive Window.

Packages
--------
Required for the complete lesson:
    pip install langchain-text-splitters scikit-learn

Optional:
    pip install pypdf                 # only for --pdf
    pip install sentence-transformers # only for --semantic

Why TF-IDF is the default embedding demo
-----------------------------------------
TF-IDF creates vectors from word importance. It is a friendly, quick way to
observe vector shapes and cosine similarity without downloading a model. It is
NOT a semantic neural embedding model: it mostly matches shared words.
``--semantic`` uses all-MiniLM-L6-v2 instead, so paraphrases can match too.

Important RAG rule: index documents and embed the user's query with the SAME
embedding model. Otherwise their vectors live in incompatible spaces.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# %% [1] Configuration and a small document
# ``Path(__file__)`` makes the PDF location work from any working directory.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CLASS_PDF = (
    REPOSITORY_ROOT
    / "Class-35-26-July-2026-Chuking-Retriever"
    / "data"
    / "llama2-research-paper.pdf"
)

# Start with small numbers so the chunks are easy to inspect in a debugger.
# In a real application, tune these with representative documents and queries.
CHUNK_SIZE = 420       # characters in this lesson (not tokens)
CHUNK_OVERLAP = 80     # repeated characters shared by neighbouring chunks
TOP_K = 3

SAMPLE_DOCUMENT = """
# Artificial Intelligence Handbook

## Artificial Intelligence

Artificial intelligence enables computers to perform tasks that normally need
human intelligence. These tasks include finding patterns, making predictions,
and helping people make decisions.

## Machine Learning

Machine learning is a branch of artificial intelligence. Instead of receiving
one rule for every situation, a machine-learning system learns patterns from
examples. Supervised learning uses labelled examples, while unsupervised
learning looks for structure in unlabelled data.

## Retrieval-Augmented Generation

Retrieval-augmented generation, usually called RAG, gives a language model
relevant information before it writes an answer. During indexing, documents are
split into chunks and each chunk is converted into an embedding vector. During
question answering, the question is embedded with the same model and the most
similar chunk vectors are retrieved.

## Chunk Overlap

Chunk overlap repeats a small amount of text at the boundary between chunks.
It helps when a definition starts near the end of one chunk and finishes at the
start of the next. Too much overlap wastes storage and creates duplicate search
results, so it should be small and intentional.

## Source Metadata

Metadata is information about a chunk rather than the chunk text itself. Useful
metadata includes the source filename, page number, heading, chunk identifier,
and starting character position. Metadata makes answers traceable to a source.
""".strip()


@dataclass
class LessonDocument:
    """A tiny stand-in for a LangChain Document used by the manual examples."""

    page_content: str
    metadata: dict[str, Any]


# %% [2] Small inspection helpers
def get_text(item: str | LessonDocument | Any) -> str:
    """Return chunk text whether the item is a string or a Document-like object."""

    return getattr(item, "page_content", str(item))


def get_metadata(item: str | LessonDocument | Any) -> dict[str, Any]:
    """Return metadata, or an empty dictionary when an item has none."""

    return getattr(item, "metadata", {})


def print_chunks(title: str, chunks: list[Any], preview_length: int = 170) -> None:
    """Print chunks in a compact form that is convenient to examine while debugging."""

    print(f"\n{'=' * 88}\n{title}\nTotal chunks: {len(chunks)}\n{'=' * 88}")
    for number, chunk in enumerate(chunks, start=1):
        text = get_text(chunk)
        one_line_preview = " ".join(text.split())
        if len(one_line_preview) > preview_length:
            one_line_preview = one_line_preview[:preview_length] + "..."
        print(f"Chunk {number:>2} | characters={len(text):>4} | metadata={get_metadata(chunk)}")
        print(f"  {one_line_preview}")


def show_overlap(left_chunk: str, right_chunk: str, width: int = 90) -> None:
    """Show the two edges where overlap is expected to occur."""

    print("\nBoundary inspection:")
    print("  end of first chunk :", repr(left_chunk[-width:]))
    print("  start of next chunk:", repr(right_chunk[:width]))


# %% [3] Document loading: text first, PDF only when requested
def load_pdf_as_documents(pdf_path: Path) -> list[LessonDocument]:
    """Extract one LessonDocument per PDF page and retain its page metadata.

    A PDF parser only extracts text; it does not understand RAG. Always print
    and inspect parser output before embedding it. Scanned PDFs need OCR first.
    """

    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise RuntimeError(
            "PDF mode needs pypdf. Install it with: pip install pypdf"
        ) from error

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    documents: list[LessonDocument] = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        print(f"PDF page {page_number}: extracted {len(page_text)} characters")
        if page_text.strip():
            documents.append(
                LessonDocument(
                    page_content=page_text,
                    metadata={"source": pdf_path.name, "page": page_number},
                )
            )

    if not documents:
        raise ValueError("No readable text was extracted. Is this a scanned PDF?")
    return documents


def choose_source_documents(pdf_argument: str | None) -> list[LessonDocument]:
    """Choose the built-in text unless the user supplied --pdf."""

    if pdf_argument is None:
        return [
            LessonDocument(
                page_content=SAMPLE_DOCUMENT,
                metadata={"source": "built_in_handbook.md", "page": 1},
            )
        ]

    pdf_path = CLASS_PDF if pdf_argument == "__CLASS_PDF__" else Path(pdf_argument)
    print(f"\nLoading PDF: {pdf_path}")
    return load_pdf_as_documents(pdf_path)


# %% [4] Manual fixed-size chunking: the simplest algorithm
def fixed_size_chunks(text: str, chunk_size: int, overlap: int = 0) -> list[str]:
    """Split at exact character positions.

    This is intentionally simple. Its drawback is visible in the result: it can
    cut a word or sentence in two. We use it to understand ``chunk_size`` and
    ``overlap`` before using a smarter splitter.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size.")

    step = chunk_size - overlap
    return [text[start : start + chunk_size] for start in range(0, len(text), step)]


def demonstrate_manual_chunking() -> None:
    """Use alphabet text so overlap is obvious, then explain the result."""

    alphabet_chunks = fixed_size_chunks("ABCDEFGHIJKLMNOPQRSTUVWXYZ", 10, overlap=3)
    print("\nManual fixed-size demo (size=10, overlap=3):")
    print(alphabet_chunks)
    print("The next chunk begins 3 characters before the previous chunk ended.")


# %% [5] Recursive chunking: a strong default for ordinary prose
def recursive_chunk_documents(
    documents: list[LessonDocument],
    chunk_size: int,
    overlap: int,
) -> list[Any]:
    """Chunk documents at natural boundaries while preserving source metadata.

    RecursiveCharacterTextSplitter tries the separators in this order. It first
    tries a paragraph boundary, then a line, then a sentence-like boundary,
    then a word, and only finally individual characters.
    """

    try:
        from langchain_core.documents import Document
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as error:
        raise RuntimeError(
            "This lesson needs langchain-text-splitters. Install it with: "
            "pip install langchain-text-splitters"
        ) from error

    langchain_documents = [
        Document(page_content=document.page_content, metadata=document.metadata)
        for document in documents
    ]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,       # These settings therefore mean *characters*.
        add_start_index=True,      # Helpful metadata when tracing a retrieval.
    )
    return splitter.split_documents(langchain_documents)


def compare_chunk_settings(text: str) -> None:
    """Show the storage/retrieval trade-off from size and overlap choices."""

    print("\nChunk-size experiment (recursive character splitter):")
    print(" size | overlap | chunks | stored characters")
    print("------|---------|--------|------------------")
    for chunk_size, overlap in [(220, 30), (420, 80), (700, 100)]:
        chunks = recursive_chunk_documents(
            [LessonDocument(text, {"source": "experiment"})], chunk_size, overlap
        )
        stored_characters = sum(len(get_text(chunk)) for chunk in chunks)
        print(f" {chunk_size:>4} | {overlap:>7} | {len(chunks):>6} | {stored_characters:>16}")
    print("Smaller chunks give more precise retrieval; larger chunks carry more context.")


# %% [6] Embeddings: turn every chunk into a numerical vector
@dataclass
class EmbeddingResult:
    """Everything needed to search the chunk vectors later."""

    vectors: np.ndarray
    embed_query: Callable[[str], np.ndarray]
    model_name: str


def create_tfidf_embeddings(chunk_texts: list[str]) -> EmbeddingResult:
    """Create fast local *lexical* vectors, requiring no model download."""

    vectorizer = TfidfVectorizer(stop_words="english")
    document_matrix = vectorizer.fit_transform(chunk_texts)

    def embed_query(query: str) -> np.ndarray:
        return vectorizer.transform([query]).toarray()[0]

    return EmbeddingResult(
        vectors=document_matrix.toarray(),
        embed_query=embed_query,
        model_name="TF-IDF teaching baseline (lexical vectors, not neural embeddings)",
    )


def create_semantic_embeddings(chunk_texts: list[str]) -> EmbeddingResult:
    """Create local neural embeddings with Sentence Transformers.

    ``normalize_embeddings=True`` makes dot product equal cosine similarity.
    The same ``model`` object is deliberately used for both documents and query.
    """

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "Semantic mode needs sentence-transformers. Install it with: "
            "pip install sentence-transformers"
        ) from error

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    document_vectors = np.asarray(
        model.encode(chunk_texts, normalize_embeddings=True, show_progress_bar=False)
    )

    def embed_query(query: str) -> np.ndarray:
        return np.asarray(model.encode(query, normalize_embeddings=True))

    return EmbeddingResult(
        vectors=document_vectors,
        embed_query=embed_query,
        model_name=model_name,
    )


def cosine_similarity(query_vector: np.ndarray, document_vectors: np.ndarray) -> np.ndarray:
    """Return one cosine-similarity score per document vector.

    Cosine similarity measures the angle between vectors. A higher value means
    vectors point in a more similar direction. The denominator prevents vector
    length alone from deciding the score.
    """

    query_norm = np.linalg.norm(query_vector)
    document_norms = np.linalg.norm(document_vectors, axis=1)
    if query_norm == 0 or np.any(document_norms == 0):
        raise ValueError("A zero vector cannot be compared with cosine similarity.")
    return (document_vectors @ query_vector) / (document_norms * query_norm)


# %% [7] Retrieval: rank chunks by similarity to a question
def retrieve(
    query: str,
    chunks: list[Any],
    embedding_result: EmbeddingResult,
    top_k: int,
) -> list[tuple[float, Any]]:
    """Embed one query, score every chunk, and return the highest-scoring ones."""

    if not 1 <= top_k <= len(chunks):
        raise ValueError(f"top_k must be between 1 and {len(chunks)}.")

    query_vector = embedding_result.embed_query(query)
    scores = cosine_similarity(query_vector, embedding_result.vectors)
    best_indices = np.argsort(scores)[::-1][:top_k]
    return [(float(scores[index]), chunks[index]) for index in best_indices]


def print_retrieval_results(query: str, results: list[tuple[float, Any]]) -> None:
    """Make the retrieval step visible before an LLM is introduced."""

    print(f"\n{'=' * 88}\nQUESTION: {query}\nTOP RETRIEVED CHUNKS\n{'=' * 88}")
    for rank, (score, chunk) in enumerate(results, start=1):
        print(f"\nRank {rank} | cosine similarity = {score:.4f}")
        print("Metadata:", get_metadata(chunk))
        print(get_text(chunk))


# %% [8] Main lesson runner
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Learn the chunking + embedding part of RAG.")
    parser.add_argument(
        "--pdf",
        nargs="?",
        const="__CLASS_PDF__",
        default=None,
        metavar="PATH",
        help="Extract a PDF. With no PATH, use the classroom Llama 2 PDF.",
    )
    parser.add_argument(
        "--semantic",
        action="store_true",
        help="Use Sentence Transformers instead of the no-download TF-IDF baseline.",
    )
    parser.add_argument(
        "--query",
        default="Why do RAG chunks need overlap and metadata?",
        help="Question used for the similarity-search demo.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    print("RAG FOUNDATION: documents -> chunks -> embeddings -> retrieval")
    print(f"Chunk settings: chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP} characters")

    demonstrate_manual_chunking()
    source_documents = choose_source_documents(args.pdf)
    source_text = "\n\n".join(document.page_content for document in source_documents)
    print(f"\nLoaded {len(source_documents)} source document(s), {len(source_text)} total characters.")

    # Compare the naive approach with a boundary-aware splitter.
    manual_chunks = fixed_size_chunks(source_text, CHUNK_SIZE, CHUNK_OVERLAP)
    print_chunks("1. Manual fixed-size chunks (can split words)", manual_chunks)

    chunks = recursive_chunk_documents(source_documents, CHUNK_SIZE, CHUNK_OVERLAP)
    print_chunks("2. Recursive chunks (recommended starting point for prose)", chunks)
    if len(chunks) >= 2:
        show_overlap(get_text(chunks[0]), get_text(chunks[1]))

    compare_chunk_settings(source_text)

    chunk_texts = [get_text(chunk) for chunk in chunks]
    if args.semantic:
        print("\nCreating semantic embeddings; first use may download the model...")
        embedding_result = create_semantic_embeddings(chunk_texts)
    else:
        embedding_result = create_tfidf_embeddings(chunk_texts)

    print("\nEmbedding model:", embedding_result.model_name)
    print("Embedding matrix shape: ", embedding_result.vectors.shape)
    print("Meaning: rows = chunks, columns = vector dimensions/features.")

    results = retrieve(args.query, chunks, embedding_result, top_k=min(TOP_K, len(chunks)))
    print_retrieval_results(args.query, results)

    print("\nNext foundation step: save these vectors and metadata in a vector database.")
    print("Before adding an LLM, verify that the retrieved chunks truly answer the question.")


if __name__ == "__main__":
    main()
