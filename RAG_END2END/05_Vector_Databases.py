"""
Module 05: Vector Databases (Debuggable Python Script)
======================================================
This script mirrors the workflow from 05_Vector_Databases.ipynb.
Covers FAISS, ChromaDB, and standalone vector store integrations.

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

# %% [1] Setup: Load environment variables and prepare data
load_dotenv()

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

from langchain_core.documents import Document

# Sample documents for all examples
sample_docs = [
    Document(page_content="RAG combines retrieval systems with generative AI for accurate responses.", metadata={"source": "rag_guide.pdf", "topic": "RAG"}),
    Document(page_content="Vector databases store embeddings and enable fast similarity search.", metadata={"source": "vectordb_guide.pdf", "topic": "VectorDB"}),
    Document(page_content="FAISS is a library for efficient similarity search developed by Facebook AI.", metadata={"source": "faiss_docs.pdf", "topic": "VectorDB"}),
    Document(page_content="ChromaDB is an open-source embedding database perfect for local development.", metadata={"source": "chroma_docs.pdf", "topic": "VectorDB"}),
    Document(page_content="Chunking splits documents into smaller pieces for better retrieval precision.", metadata={"source": "rag_guide.pdf", "topic": "Chunking"}),
    Document(page_content="Embeddings convert text into numerical vectors capturing semantic meaning.", metadata={"source": "embedding_guide.pdf", "topic": "Embeddings"}),
    Document(page_content="Pinecone is a fully managed vector database service for production use.", metadata={"source": "pinecone_docs.pdf", "topic": "VectorDB"}),
    Document(page_content="Fine-tuning modifies model weights while RAG retrieves external knowledge.", metadata={"source": "rag_guide.pdf", "topic": "RAG"}),
]

print(f"✅ Loaded {len(sample_docs)} sample documents")


# %% [2] Embeddings Initialization
try:
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print(f"📐 Embedding dimension: {len(embeddings.embed_query('test'))}")
except ImportError:
    print("⚠️ langchain_huggingface not installed. Run: pip install langchain-huggingface sentence-transformers")
    embeddings = None


# %% [3] FAISS — Fast Local Search
if embeddings:
    try:
        from langchain_community.vectorstores import FAISS

        faiss_db = FAISS.from_documents(sample_docs, embeddings)
        print("\n✅ FAISS vector store created")

        # Similarity search
        query = "What is a vector database?"
        results = faiss_db.similarity_search(query, k=3)

        print(f"\n🔍 Query: '{query}'")
        print(f"📄 Top {len(results)} results:")
        for i, doc in enumerate(results, 1):
            print(f"  [{i}] {doc.page_content}")
            print(f"      Metadata: {doc.metadata}")

        # Search with scores
        results_with_scores = faiss_db.similarity_search_with_score(query, k=3)
        print(f"\n🔍 Results with scores (lower = more similar in FAISS / L2 distance):")
        for doc, score in results_with_scores:
            print(f"  Score: {score:.4f} | {doc.page_content[:60]}...")

        # Save and Load
        faiss_save_path = os.path.join(DATA_DIR, "faiss_index")
        faiss_db.save_local(faiss_save_path)
        print(f"\n💾 Saved FAISS index to {faiss_save_path}")

        loaded_db = FAISS.load_local(
            faiss_save_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        print(f"📂 Loaded FAISS index from {faiss_save_path}")
    except ImportError:
        print("ℹ️ faiss-cpu not installed (optional: pip install faiss-cpu)")
    except Exception as e:
        print(f"ℹ️ FAISS info: {e}")


# %% [4] ChromaDB — Embedded Vector Database
if embeddings:
    try:
        from langchain_chroma import Chroma

        chroma_path = os.path.join(DATA_DIR, "chroma_db")
        chroma_db = Chroma.from_documents(
            documents=sample_docs,
            embedding=embeddings,
            persist_directory=chroma_path,
            collection_name="exam_prep",
        )
        print("\n✅ ChromaDB vector store created (auto-persisted)")

        results = chroma_db.similarity_search("What is RAG?", k=3)
        print(f"\n🔍 ChromaDB: Top {len(results)} results for 'What is RAG?':")
        for i, doc in enumerate(results, 1):
            print(f"  [{i}] {doc.page_content}")
    except ImportError:
        print("ℹ️ langchain_chroma / chromadb not installed (optional: pip install langchain-chroma chromadb)")
    except Exception as e:
        print(f"ℹ️ ChromaDB info: {e}")


# %% [5] Summary & DB Selection Guide
print("\n" + "=" * 60)
print("✅ Module 5 Complete!")
print("=" * 60)
print("""
Vector Database Selection Guide:
1. FAISS: Fastest local search library (L2 distance default, manual save_local)
2. ChromaDB: Easiest local embedded DB (cosine default, auto-persisting)
3. Pinecone: Fully managed cloud service for production scale
4. Qdrant: Versatile local/cloud with rich payload filtering
""")
