"""
Module 02: Data Parsing & Document Loaders (Debuggable Python Script)
=====================================================================
This script mirrors the workflow from 02_Data_Parsing_and_Loaders.ipynb.
You can run this directly in VS Code, debug with breakpoints (F5),
or run individual cells using VS Code's Interactive Window (# %%).
"""

import sys
import os
import csv
import json
import warnings
from dotenv import load_dotenv

# Suppress sunset/migration deprecation warnings from legacy community packages
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# %% [1] Setup: Load environment variables & paths
load_dotenv()

# Base directory for robust file path resolution
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

print("✅ Environment loaded")
print(f"📁 Working directory: {CURRENT_DIR}")


# %% [2] TextLoader — Plain Text Files
# Note: langchain_core defines BaseLoader & Document interfaces.
# Concrete document loaders are imported from langchain_community.document_loaders.
from langchain_community.document_loaders import TextLoader

sample_txt_path = os.path.join(DATA_DIR, "sample.txt")

# Create sample.txt if it doesn't exist
if not os.path.exists(sample_txt_path):
    with open(sample_txt_path, "w", encoding="utf-8") as f:
        f.write(
            "This is a sample document for testing RAG pipeline components.\n\n"
            "Chapter 1: Introduction to AI\n"
            "Artificial Intelligence simulates human intelligence processes in software.\n\n"
            "Chapter 2: Retrieval-Augmented Generation\n"
            "RAG grounds LLM generation in external retrieved documents."
        )

loader = TextLoader(sample_txt_path, encoding="utf-8")
documents = loader.load()

print(f"\n📄 TextLoader: Loaded {len(documents)} document(s)")
print(f"📝 Content preview: {documents[0].page_content[:200]}...")
print(f"📋 Metadata: {documents[0].metadata}")


# %% [3] PyPDFLoader — PDF Files
from langchain_community.document_loaders import PyPDFLoader

# Example PDF path from class materials
pdf_path = os.path.abspath(
    os.path.join(
        CURRENT_DIR,
        "..",
        "Class-29-RAG-Introduction",
        "Class-29-handwritten-notes-04-July-2026-RAG-Intro.pdf",
    )
)

try:
    if os.path.exists(pdf_path):
        loader = PyPDFLoader(pdf_path)
        pdf_docs = loader.load()
        print(f"\n📄 PyPDFLoader: Total pages loaded: {len(pdf_docs)}")
        print(f"📝 Page 1 preview: {pdf_docs[0].page_content[:200]}...")
        print(f"📋 Metadata: {pdf_docs[0].metadata}")
    else:
        print(f"\nℹ️ PDF not found at {pdf_path} (skipping live PDF load)")
except Exception as e:
    print(f"\n⚠️ Could not load PDF: {e}")


# %% [4] CSVLoader — Create sample CSV and load it
csv_data = [
    ["Concept", "Description", "Category"],
    ["RAG", "Retrieval-Augmented Generation - combines retrieval with LLM generation", "Architecture"],
    ["FAISS", "Facebook AI Similarity Search - fast local vector search library", "Vector DB"],
    ["ChromaDB", "Open-source embedding database for local development", "Vector DB"],
    ["Chunking", "Splitting documents into smaller pieces for better retrieval", "Data Processing"],
    ["Embedding", "Converting text to numerical vectors that capture semantic meaning", "Representation"],
]

csv_path = os.path.join(DATA_DIR, "concepts.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(csv_data)

print(f"\n✅ Created sample CSV at {csv_path}")

from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(csv_path)
csv_docs = loader.load()

print(f"📄 CSVLoader: Loaded {len(csv_docs)} rows as documents")
print(f"📝 First document:\n{csv_docs[0].page_content}")
print(f"📋 Metadata: {csv_docs[0].metadata}")


# %% [5] JSONLoader — Structured JSON
json_data = {
    "topics": [
        {
            "name": "RAG Pipeline",
            "description": "End-to-end pipeline for retrieval-augmented generation",
            "components": ["Loader", "Splitter", "Embeddings", "Vector Store", "Retriever", "LLM"]
        },
        {
            "name": "Vector Database",
            "description": "Database optimized for storing and searching vector embeddings",
            "components": ["FAISS", "ChromaDB", "Pinecone", "Qdrant"]
        }
    ]
}

json_path = os.path.join(DATA_DIR, "topics.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(json_data, f, indent=2)

print(f"\n✅ Created sample JSON at {json_path}")

try:
    from langchain_community.document_loaders import JSONLoader

    # jq_schema specifies which field to extract
    loader = JSONLoader(
        file_path=json_path,
        jq_schema=".topics[].description",
        text_content=False,
    )
    json_docs = loader.load()

    print(f"📄 JSONLoader: Loaded {len(json_docs)} topic description(s)")
    for doc in json_docs:
        print(f"  📝 {doc.page_content}")
except ImportError:
    print("ℹ️ jq package not installed for JSONLoader schema parsing (optional: pip install jq)")
except Exception as e:
    print(f"ℹ️ JSONLoader info: {e}")


# %% [6] DirectoryLoader — Load Multiple Files
from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader(
    DATA_DIR,
    glob="*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
)
dir_docs = loader.load()

print(f"\n📄 DirectoryLoader: Loaded {len(dir_docs)} document(s) matching '*.txt'")
for doc in dir_docs:
    print(f"  📁 Source: {doc.metadata.get('source', 'unknown')}")


# %% [7] Advanced Parser: PyMuPDF (fitz)
try:
    import fitz  # PyMuPDF

    if os.path.exists(pdf_path):
        doc = fitz.open(pdf_path)
        print(f"\n📄 PyMuPDF: PDF has {len(doc)} pages")
        page = doc[0]
        print(f"📝 Page 1 text (first 200 chars):\n{page.get_text()[:200]}")
        doc.close()
except ImportError:
    print("\n⚠️ PyMuPDF not installed. Run: pip install pymupdf")
except Exception as e:
    print(f"\n⚠️ PyMuPDF info: {e}")


# %% [8] Summary & Architecture Guide
print("\n" + "=" * 60)
print("✅ Module 2 Complete!")
print("=" * 60)
print("""
Architecture Notes:
1. langchain_core: BaseLoader & Document data structures
2. langchain_community.document_loaders: TextLoader, PyPDFLoader, CSVLoader, etc.
3. Path best practice: Always resolve file paths relative to script directory
4. Every loader returns Document(page_content="...", metadata={...})
""")
