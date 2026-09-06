"""
Module 06: Retrievers (Debuggable Python Script)
=================================================
This script mirrors the workflow from 06_Retrievers.ipynb.
Covers Similarity, MMR, Multi-Query, Compression, and Ensemble retrievers.

You can run this directly in VS Code, debug with breakpoints (F5),
or run individual cells using VS Code's Interactive Window (# %%).
"""

import sys
import os
import warnings
from dotenv import load_dotenv

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# %% [1] Setup: Create vector store for retriever demos
load_dotenv()

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

from langchain_core.documents import Document

docs = [
    Document(page_content="RAG stands for Retrieval-Augmented Generation. It enhances LLM responses with external knowledge.", metadata={"topic": "RAG", "section": "intro"}),
    Document(page_content="The RAG pipeline has two phases: indexing (offline) and querying (online).", metadata={"topic": "RAG", "section": "architecture"}),
    Document(page_content="Vector databases like FAISS and ChromaDB store embeddings for similarity search.", metadata={"topic": "VectorDB", "section": "overview"}),
    Document(page_content="FAISS is developed by Facebook AI Research and is known for its speed.", metadata={"topic": "VectorDB", "section": "FAISS"}),
    Document(page_content="Cosine similarity measures the angle between two vectors, ranging from -1 to 1.", metadata={"topic": "Embeddings", "section": "metrics"}),
    Document(page_content="RecursiveCharacterTextSplitter is the recommended default chunking strategy.", metadata={"topic": "Chunking", "section": "methods"}),
    Document(page_content="Fine-tuning changes model weights, while RAG retrieves external information at inference.", metadata={"topic": "RAG", "section": "comparison"}),
    Document(page_content="BM25 is a sparse retrieval method based on term frequency that excels at keyword matching.", metadata={"topic": "Retrieval", "section": "sparse"}),
    Document(page_content="MMR (Maximal Marginal Relevance) balances relevance with diversity in retrieved results.", metadata={"topic": "Retrieval", "section": "diversity"}),
    Document(page_content="Hybrid search combines dense semantic search with sparse keyword search for best results.", metadata={"topic": "Retrieval", "section": "hybrid"}),
]


# %% [2] Initialize Embeddings & Vector Store
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    print(f"✅ Created vector store with {len(docs)} documents")

    # %% [3] Basic Similarity Retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    query = "What is the difference between RAG and fine-tuning?"
    results = retriever.invoke(query)

    print(f"\n🔍 Similarity Search for '{query}':")
    for i, doc in enumerate(results, 1):
        print(f"[{i}] {doc.page_content}")
        print(f"    Topic: {doc.metadata['topic']} | Section: {doc.metadata['section']}\n")

    # %% [4] ⭐ MMR (Maximal Marginal Relevance) Retriever
    mmr_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 10,
            "lambda_mult": 0.5,
        },
    )

    query = "Tell me about RAG"
    results = mmr_retriever.invoke(query)

    print(f"🔍 MMR (Diversity) Results for '{query}':")
    for i, doc in enumerate(results, 1):
        print(f"[{i}] {doc.page_content}")
        print(f"    Topic: {doc.metadata['topic']}\n")

    # %% [5] Ensemble Retriever (Hybrid Search: Dense + BM25)
    try:
        from langchain.retrievers import EnsembleRetriever
        from langchain_community.retrievers import BM25Retriever

        dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        bm25_retriever = BM25Retriever.from_documents(docs, k=3)

        ensemble_retriever = EnsembleRetriever(
            retrievers=[dense_retriever, bm25_retriever],
            weights=[0.5, 0.5],
        )

        query = "FAISS vector search"
        results = ensemble_retriever.invoke(query)

        print(f"🔍 Ensemble (Hybrid) Results for '{query}':")
        for i, doc in enumerate(results, 1):
            print(f"[{i}] {doc.page_content}")
            print(f"    Topic: {doc.metadata.get('topic', 'N/A')}\n")
    except ImportError as e:
        print(f"ℹ️ Ensemble retriever info: {e}")

except ImportError:
    print("⚠️ Required packages not installed. Run: pip install langchain-huggingface sentence-transformers faiss-cpu")


# %% [6] Summary & Exam Guide
print("\n" + "=" * 60)
print("✅ Module 6 Complete!")
print("=" * 60)
print("""
Retriever Selection Summary:
1. Similarity: Default nearest neighbor search
2. MMR: Balances relevance with diversity (fetch_k > k, lambda_mult)
3. Multi-Query: LLM generates multiple query phrasings for high recall
4. Compression: LLM filters irrelevant tokens from retrieved chunks
5. Ensemble / Hybrid: Combines dense (semantic) + BM25 (keyword) using RRF
""")
