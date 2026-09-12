# 📋 RAG Pipeline — Quick Reference Cheat Sheet

## GenAI L2 Exam — Rapid Revision (30 min)

---

## 🏗️ RAG Pipeline (6 Stages)

```
LOAD → CHUNK → EMBED → STORE → RETRIEVE → GENERATE
```

| Stage | LangChain Class | Default Choice |
|-------|----------------|----------------|
| Load | `PyPDFLoader`, `TextLoader` | `PyPDFLoader` for PDFs |
| Chunk | `RecursiveCharacterTextSplitter` | chunk_size=1000, overlap=200 |
| Embed | `HuggingFaceEmbeddings` | `all-MiniLM-L6-v2` (384d, free) |
| Store | `FAISS`, `Chroma` | ChromaDB (dev), Pinecone (prod) |
| Retrieve | `vectorstore.as_retriever()` | similarity, k=5 |
| Generate | `ChatGroq`, `RetrievalQA` | stuff chain type |

---

## ⚡ Essential Code Snippets

### Minimal RAG Pipeline
```python
from dotenv import load_dotenv; load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# Load → Chunk → Embed → Store
docs = PyPDFLoader("file.pdf").load()
chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)

# Retrieve → Generate
qa = RetrievalQA.from_chain_type(
    llm=ChatGroq(model="llama-3.1-8b-instant"),
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True
)
result = qa.invoke({"query": "What is RAG?"})
```

### LCEL Chain
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer based on context only."),
    ("human", "Context: {context}\nQuestion: {question}")
])

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)
```

---

## 🆚 RAG vs Fine-Tuning

| | RAG | Fine-Tuning |
|---|-----|-------------|
| **Data** | External, real-time | Baked into weights |
| **Cost** | Low | High (GPU) |
| **Freshness** | ✅ Always current | ❌ Frozen |
| **Hallucination** | Reduced | Can still occur |
| **Best for** | Q&A, knowledge base | Style, format, behavior |

**Rule**: Need external/private/fresh data → **RAG**. Need behavior change → **Fine-tuning**.

---

## 📏 Chunking Quick Reference

| Parameter | Recommended | Too Small | Too Large |
|-----------|------------|-----------|-----------|
| chunk_size | 500-1000 | Fragments meaning | Dilutes relevance |
| chunk_overlap | 10-20% of size | Loses boundary context | Redundant storage |

**Default**: `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)`  
**Separators**: `["\n\n", "\n", ". ", " ", ""]`

---

## 🗄️ Vector Database Comparison

| | FAISS | ChromaDB | Pinecone | Qdrant |
|---|-------|----------|----------|--------|
| **Deploy** | Local | Local | Cloud | Both |
| **Speed** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Scale** | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Filter** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Persist** | Manual | Auto | Auto | Auto |
| **Metric** | L2 (lower=better) | Cosine (higher=better) | Cosine | Cosine |

**FAISS gotcha**: `allow_dangerous_deserialization=True` for `load_local()`

---

## 🔍 Retriever Types

| Type | Use When | Key Param |
|------|----------|-----------|
| **Similarity** | Default choice | `search_type="similarity"` |
| **MMR** | Need diversity | `search_type="mmr"`, `fetch_k > k` |
| **Multi-Query** | Ambiguous query | Uses LLM to rephrase |
| **Ensemble** | Production (hybrid) | `weights=[0.5, 0.5]` |
| **BM25** | Exact keywords | Sparse retrieval |
| **Compression** | Noisy chunks | LLM extracts relevant parts |

---

## 🔗 Chain Types

| Type | How | Use When |
|------|-----|----------|
| **stuff** | All docs → single prompt | Default ⭐ (small context) |
| **map_reduce** | Each doc → LLM → combine | Many large documents |
| **refine** | Sequential refinement | Need detailed, iterative answer |
| **map_rerank** | Each doc → scored → best wins | Need single best answer |

---

## 📊 RAGAS Evaluation Metrics

| Metric | Measures | Low Score Means |
|--------|----------|----------------|
| **Faithfulness** | Answer grounded in context? | Hallucination → fix prompt |
| **Answer Relevance** | Answer matches question? | Off-topic → fix prompt/LLM |
| **Context Precision** | Retrieved docs relevant? | Bad retrieval → fix chunking |
| **Context Recall** | All relevant docs found? | Missing docs → increase k |

---

## 📐 Embedding Models

| Model | Dims | Cost | Speed |
|-------|------|------|-------|
| `all-MiniLM-L6-v2` | 384 | Free | Fast ⭐ |
| `all-mpnet-base-v2` | 768 | Free | Medium |
| `text-embedding-3-small` | 1536 | Paid | Fast |

**⚠️ CRITICAL**: Same model for indexing AND querying. Dimension mismatch = errors!

---

## 🎯 Top 8 Exam Scenarios

1. **"Company internal docs updated weekly"** → RAG (not fine-tuning)
2. **"Retrieval returns redundant docs"** → MMR retrieval
3. **"Retrieval misses exact keywords"** → Hybrid search (dense + BM25)
4. **"Context window overflow"** → Reduce k, smaller chunks, or map_reduce
5. **"Answers contain hallucinated info"** → Grounding prompt + temperature=0
6. **"Changed embedding model, errors occur"** → Dimension mismatch, re-index
7. **"Need production vector DB for millions of users"** → Pinecone or Qdrant Cloud
8. **"Quick prototype for local dev"** → ChromaDB + HuggingFace embeddings

---

## 🔑 Key Imports Cheat Sheet (Modern LangChain Architecture)

```python
# Document Data Structure & Base Loaders
from langchain_core.documents import Document

# Document Loaders (Concrete file loaders)
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    DirectoryLoader,
    JSONLoader,
)

# Text Splitters (Modern dedicated package)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)

# Embeddings (Partner package)
from langchain_huggingface import HuggingFaceEmbeddings

# Vector Stores
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_qdrant import QdrantVectorStore

# LLMs (Partner package)
from langchain_groq import ChatGroq

# Prompts, Runnables & Parsers (langchain_core)
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# Retrievers
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
```

---

> **📌 Last-minute tip**: Focus on **WHY** (reasoning about scenarios), not just **WHAT** (definitions). L2 exams test applied understanding!
