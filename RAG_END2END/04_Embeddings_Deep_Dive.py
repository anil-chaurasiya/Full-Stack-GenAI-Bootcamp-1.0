"""
Module 04: Embeddings Deep Dive (Debuggable Python Script)
==========================================================
This script mirrors the workflow from 04_Embeddings_Deep_Dive.ipynb.
You can run this directly in VS Code, debug with breakpoints (F5),
or run individual cells using VS Code's Interactive Window (# %%).
"""

import sys
import warnings
from dotenv import load_dotenv

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# %% [1] Setup: Load environment variables
load_dotenv()
print("✅ Environment loaded")


# %% [2] HuggingFace Embeddings (Free, Local)
try:
    from langchain_huggingface import HuggingFaceEmbeddings

    # Initialize embedding model (runs locally, no API key needed)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Embed a single text
    text = "RAG combines retrieval with generation"
    vector = embeddings.embed_query(text)

    print(f"\n📝 Text: '{text}'")
    print(f"📐 Vector dimension: {len(vector)}")
    print(f"🔢 First 10 values: {vector[:10]}")
    print(f"📊 Min: {min(vector):.4f}, Max: {max(vector):.4f}")

    # %% [3] Embed multiple texts at once
    texts = [
        "RAG combines retrieval with generation",
        "Retrieval-Augmented Generation enhances LLM responses",
        "The weather today is sunny and warm",
        "Vector databases store embeddings for fast search",
        "I love eating pizza for dinner",
    ]

    # embed_documents is for batch embedding
    vectors = embeddings.embed_documents(texts)

    print(f"\n📊 Embedded {len(vectors)} texts")
    print(f"📐 Each vector has {len(vectors[0])} dimensions")

    # %% [4] Distance Metrics & Cosine Similarity
    import numpy as np

    def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    print("\n📊 COSINE SIMILARITY MATRIX")
    print("=" * 60)

    labels = ["RAG+retrieval", "RAG+LLM", "Weather", "VectorDB", "Pizza"]

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sim = cosine_similarity(vectors[i], vectors[j])
            emoji = "🟢" if sim > 0.5 else "🟡" if sim > 0.3 else "🔴"
            print(f"{emoji} {labels[i]:>15} vs {labels[j]:<15}: {sim:.4f}")

    print("\n🟢 High similarity (>0.5) | 🟡 Medium (0.3-0.5) | 🔴 Low (<0.3)")

except ImportError:
    print("⚠️ langchain_huggingface not installed. Run: pip install langchain-huggingface sentence-transformers")


# %% [5] Summary
print("\n" + "=" * 60)
print("✅ Module 4 Complete!")
print("=" * 60)
print("""
Key Takeaways:
1. Embeddings convert text to dense vectors that capture semantic meaning
2. Same model must be used for indexing AND querying (e.g., all-MiniLM-L6-v2)
3. Default: all-MiniLM-L6-v2 produces 384 dimensions
4. Cosine similarity measures angle between vectors (-1 to 1)
5. Dimension mismatch is the #1 embedding debugging error
""")
