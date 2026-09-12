---
name: vector-databases
description: Set up and use vector databases including ChromaDB, FAISS, Pinecone, and Qdrant for similarity search, indexing, and retrieval in GenAI applications.
---

# Vector Databases Skill

Use this skill when the user wants to set up, configure, or troubleshoot vector databases.

## ChromaDB (Local — Recommended for Development)

### Setup
```bash
pip install langchain_chroma chromadb
```

### Usage
```python
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create from documents
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="my_collection"
)

# Load existing
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="my_collection"
)

# Search
results = vectorstore.similarity_search("query", k=5)
results_with_scores = vectorstore.similarity_search_with_score("query", k=5)
```

## FAISS (Local — Fast)

### Setup
```bash
pip install faiss-cpu  # or faiss-gpu for GPU support
```

### Usage
```python
from langchain_community.vectorstores import FAISS

# Create
vectorstore = FAISS.from_documents(chunks, embeddings)

# Save & Load
vectorstore.save_local("./faiss_index")
vectorstore = FAISS.load_local("./faiss_index", embeddings, allow_dangerous_deserialization=True)

# Search
results = vectorstore.similarity_search("query", k=5)
```

## Pinecone (Cloud)

### Setup
```bash
pip install pinecone langchain_pinecone
```

### Usage
```python
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import os

# Initialize
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Create index (one-time)
pc.create_index(
    name="my-index",
    dimension=384,  # Must match embedding dimension
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)

# Create vector store
vectorstore = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name="my-index"
)

# Search
results = vectorstore.similarity_search("query", k=5)
```

## Qdrant (Cloud or Local)

### Setup
```bash
pip install langchain_qdrant qdrant-client
```

### Usage
```python
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# Local (in-memory)
vectorstore = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    location=":memory:",
    collection_name="my_collection"
)

# Local (persistent)
vectorstore = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    path="./qdrant_data",
    collection_name="my_collection"
)

# Cloud
vectorstore = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    url="https://your-cluster.qdrant.io",
    api_key=os.getenv("QDRANT_API_KEY"),
    collection_name="my_collection"
)
```

## Comparison Matrix

| Feature          | ChromaDB  | FAISS     | Pinecone  | Qdrant    |
|------------------|-----------|-----------|-----------|-----------|
| Deployment       | Local     | Local     | Cloud     | Both      |
| Ease of setup    | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐  | ⭐⭐⭐    | ⭐⭐⭐⭐  |
| Speed            | ⭐⭐⭐    | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐  | ⭐⭐⭐⭐  |
| Scalability      | ⭐⭐      | ⭐⭐      | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Filtering        | ⭐⭐⭐    | ⭐⭐      | ⭐⭐⭐⭐  | ⭐⭐⭐⭐⭐ |
| Free tier        | ✅        | ✅        | ✅ (limited) | ✅ (limited) |

## Tips
- Always check **embedding dimension** matches your vector store configuration
- Use `allow_dangerous_deserialization=True` for FAISS `load_local` (only with trusted data)
- ChromaDB auto-persists; FAISS requires explicit `save_local()`
- For production, prefer Pinecone or Qdrant cloud for managed infrastructure
