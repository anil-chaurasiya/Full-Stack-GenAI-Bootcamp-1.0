"""Class 31: build an end-to-end RAG pipeline with LangChain.

The pipeline has six deliberately visible stages:

1. **Load**: read a PDF into LangChain ``Document`` objects.
2. **Split**: divide long pages into overlapping chunks.
3. **Embed**: convert each chunk into a numerical vector.
4. **Store**: put vectors and their documents into a FAISS index.
5. **Retrieve**: find the chunks most relevant to a question.
6. **Generate**: ask a Groq model to answer using only retrieved context.

Hugging Face's ``all-MiniLM-L6-v2`` is used for local embeddings, so the
embedding stage does not require an API key. Groq is optional until the final
generation stage. Use ``--skip-generation`` to study loading, chunking, and
retrieval without making an LLM request.

Example:

    python class_31_rag_pipeline.py \
        --pdf ../../Class-31-11-July-2026-RAG-Pipeline-End-to-End/deepseek-v4-2026.pdf
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PDF = (
    SCRIPT_DIR.parent.parent
    / "Class-31-11-July-2026-RAG-Pipeline-End-to-End"
    / "deepseek-v4-2026.pdf"
)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_documents(pdf_path: Path) -> list[Document]:
    """Load one PDF page per LangChain ``Document``.

    ``PyPDFLoader`` also adds useful metadata such as the source path and page
    number. That metadata travels through splitting and retrieval and can be
    used later for citations.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    documents = PyPDFLoader(str(pdf_path)).load()
    print(f"1. LOAD  -> {len(documents)} page document(s)")
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents while retaining enough overlap for context continuity."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=400,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"2. SPLIT -> {len(documents)} page(s) became {len(chunks)} chunk(s)")
    if chunks:
        print(f"   First chunk length: {len(chunks[0].page_content)} characters")
        print(f"   First chunk metadata: {chunks[0].metadata}")
    return chunks


def create_vector_store(chunks: list[Document]) -> FAISS:
    """Embed chunks locally and store them in a searchable FAISS index."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(chunks, embeddings)
    dimensions = len(embeddings.embed_query("dimension check"))
    print(f"3. EMBED  -> {dimensions}-dimensional local embeddings")
    print(f"4. STORE  -> {len(chunks)} vectors in FAISS")
    return vector_store


def retrieve_documents(vector_store: FAISS, question: str, k: int) -> list[Document]:
    """Retrieve the top ``k`` chunks for a question."""
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    documents = retriever.invoke(question)
    print(f"5. RETRIEVE -> {len(documents)} chunk(s) for: {question}")
    for index, document in enumerate(documents, start=1):
        preview = " ".join(document.page_content.split())[:120]
        print(f"   [{index}] {preview}...")
    return documents


def format_documents(documents: list[Document]) -> str:
    """Join retrieved chunks into the context passed to the chat model."""
    return "\n\n--- Retrieved chunk ---\n\n".join(
        document.page_content for document in documents
    )


def answer_with_groq(vector_store: FAISS, question: str, k: int) -> str:
    """Run an LCEL retrieval-generation chain with Groq.

    The retriever is part of the chain, so every question gets fresh context.
    The prompt explicitly tells the model to avoid unsupported answers, which
    is the central grounding rule in a basic RAG application.
    """
    from langchain_groq import ChatGroq

    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You answer questions using only the supplied context. "
                "If the context does not contain the answer, say you do not "
                "have enough information. Do not invent facts.\n\n"
                "Context:\n{context}",
            ),
            ("human", "Question: {question}"),
        ]
    )
    model = ChatGroq(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        temperature=0,
        reasoning_format="hidden",
    )
    rag_chain = (
        {"context": retriever | format_documents, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    return rag_chain.invoke(question)


def main() -> None:
    """Run the pipeline from the command line."""
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdf",
        type=Path,
        default=DEFAULT_PDF,
        help=f"PDF to index (default: {DEFAULT_PDF})",
    )
    parser.add_argument(
        "--query",
        default="What is this document about?",
        help="Question to retrieve and answer",
    )
    parser.add_argument("--k", type=int, default=4, help="Number of chunks to retrieve")
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Stop after retrieval and do not call Groq",
    )
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be at least 1")

    documents = load_documents(args.pdf)
    chunks = split_documents(documents)
    vector_store = create_vector_store(chunks)
    retrieve_documents(vector_store, args.query, args.k)

    if args.skip_generation:
        print("6. GENERATE -> skipped by --skip-generation")
        return

    if not os.getenv("GROQ_API_KEY"):
        print("6. GENERATE -> skipped; GROQ_API_KEY is not configured")
        print("   Add GROQ_API_KEY to .env or use --skip-generation.")
        return

    answer = answer_with_groq(vector_store, args.query, args.k)
    print(f"6. GENERATE -> {answer}")


if __name__ == "__main__":
    main()