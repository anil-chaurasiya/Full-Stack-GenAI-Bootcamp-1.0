"""
Module 03: Chunking Strategies (Debuggable Python Script)
=========================================================
This script mirrors the workflow from 03_Chunking_Strategies.ipynb.
Uses modern langchain_text_splitters package.

You can run this directly in VS Code, debug with breakpoints (F5),
or run individual cells using VS Code's Interactive Window (# %%).
"""

import sys
import warnings

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# %% [1] Setup: Create sample document for chunking demos
sample_text = """
Chapter 1: Introduction to Artificial Intelligence

Artificial Intelligence (AI) is the simulation of human intelligence processes by computer systems. These processes include learning (the acquisition of information and rules for using the information), reasoning (using rules to reach approximate or definite conclusions), and self-correction.

AI can be categorized into three types: Narrow AI (also known as Weak AI), General AI (also known as Strong AI), and Super AI. Narrow AI is designed to perform specific tasks, such as voice recognition or image classification. General AI would have the ability to understand, learn, and apply intelligence broadly, similar to human cognitive abilities. Super AI would surpass human intelligence in all aspects.

Chapter 2: Machine Learning Fundamentals

Machine Learning (ML) is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. The learning process begins with observations or data, such as examples, direct experience, or instruction.

There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning. In supervised learning, the algorithm learns from labeled training data. In unsupervised learning, the algorithm identifies patterns in unlabeled data. Reinforcement learning involves an agent learning to make decisions by interacting with an environment.

Chapter 3: Deep Learning and Neural Networks

Deep Learning is a subset of machine learning that uses artificial neural networks with multiple layers (hence 'deep') to model and understand complex patterns in data. These deep neural networks are inspired by the structure and function of the human brain.

Key architectures in deep learning include Convolutional Neural Networks (CNNs) for image processing, Recurrent Neural Networks (RNNs) for sequential data, and Transformers for natural language processing. The Transformer architecture, introduced in the paper 'Attention is All You Need', has revolutionized NLP and forms the basis of modern large language models.
"""

print(f"📄 Sample text length: {len(sample_text)} characters")
print(f"📊 Approximate words: {len(sample_text.split())}")


# %% [2] CharacterTextSplitter (Basic)
# Modern LangChain standard: import from langchain_text_splitters
from langchain_text_splitters import CharacterTextSplitter

# Splits on a single separator
splitter = CharacterTextSplitter(
    separator="\n\n",  # Split on double newlines (paragraphs)
    chunk_size=300,
    chunk_overlap=50,
    length_function=len
)

chunks = splitter.split_text(sample_text)

print(f"\n📊 CharacterTextSplitter — Number of chunks: {len(chunks)}\n")
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ({len(chunk)} chars) ---")
    print(chunk[:150] + "..." if len(chunk) > 150 else chunk)
    print()


# %% [3] ⭐ RecursiveCharacterTextSplitter (DEFAULT CHOICE)
# Modern LangChain standard: import from langchain_text_splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Tries separators in order: \n\n → \n → . → " " → ""
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]  # Default hierarchy
)

chunks = splitter.split_text(sample_text)

print(f"📊 RecursiveCharacterTextSplitter — Number of chunks: {len(chunks)}\n")
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ({len(chunk)} chars) ---")
    print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
    print()


# %% [4] Effect of different chunk_size values
print("=" * 60)
print("EFFECT OF chunk_size ON CHUNKING")
print("=" * 60)

for size in [100, 300, 500, 1000]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=int(size * 0.2)  # 20% overlap
    )
    chunks = splitter.split_text(sample_text)
    avg_len = sum(len(c) for c in chunks) / len(chunks)

    print(f"\nchunk_size={size:>5} | overlap={int(size*0.2):>4} | chunks={len(chunks):>3} | avg_chunk_len={avg_len:.0f}")


# %% [5] Visualizing chunk_overlap
print("\n" + "=" * 60)
print("VISUALIZING CHUNK OVERLAP")
print("=" * 60)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)
chunks = splitter.split_text(sample_text)

# Show overlap between chunk 1 and chunk 2
if len(chunks) >= 2:
    chunk1_end = chunks[0][-60:]
    chunk2_start = chunks[1][:60]

    print(f"\n📝 Chunk 1 ends with:")
    print(f"   ...{chunk1_end}")
    print(f"\n📝 Chunk 2 starts with:")
    print(f"   {chunk2_start}...")
    print(f"\n🔄 The overlap ensures context is not lost at chunk boundaries!")


# %% [6] Using with Documents (preserving metadata)
from langchain_core.documents import Document

# When you have Document objects (from loaders), use split_documents()
documents = [
    Document(page_content=sample_text, metadata={"source": "textbook.pdf", "page": 1})
]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

# split_documents preserves and propagates metadata!
chunks = splitter.split_documents(documents)

print(f"\n📊 Input: {len(documents)} document → Output: {len(chunks)} chunks\n")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {len(chunk.page_content)} chars | metadata: {chunk.metadata}")


# %% [7] Summary
print("\n" + "=" * 60)
print("✅ Module 3 Complete!")
print("=" * 60)
print("""
Key Takeaways:
1. Always import from langchain_text_splitters (not langchain.text_splitter)
2. RecursiveCharacterTextSplitter is the standard default for text
3. Recommended: chunk_size=1000, chunk_overlap=200 (10-20% overlap)
4. Use split_documents() to preserve metadata on Document objects
""")
