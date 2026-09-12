# 📚 GenAI L2 Exam — RAG Pipeline Study Roadmap

> **Goal**: Master RAG pipelines end-to-end for the GenAI L2 exam  
> **Duration**: 5 days of focused study  
> **Method**: Theory → Code → Exam Tips → Self-Quiz per topic

---

## 📅 Day-by-Day Schedule

### Day 1 — Foundations & Data Ingestion
| Status | Topic | File | Est. Time |
|--------|-------|------|-----------|
| ☐ | RAG Fundamentals & Architecture | `01_RAG_Fundamentals.ipynb` | 1.5 hrs |
| ☐ | Data Parsing & Document Loaders | `02_Data_Parsing_and_Loaders.ipynb` | 2 hrs |
| ☐ | Day 1 Self-Assessment | Review quiz sections in both notebooks | 30 min |

**Key Exam Topics**: RAG vs Fine-tuning, pipeline stages, when to use RAG, document loader selection.

---

### Day 2 — Text Processing & Representation
| Status | Topic | File | Est. Time |
|--------|-------|------|-----------|
| ☐ | Chunking Strategies | `03_Chunking_Strategies.ipynb` | 2 hrs |
| ☐ | Embeddings Deep Dive | `04_Embeddings_Deep_Dive.ipynb` | 2 hrs |
| ☐ | Day 2 Self-Assessment | Review quiz sections in both notebooks | 30 min |

**Key Exam Topics**: chunk_size/overlap tradeoffs, RecursiveCharacterTextSplitter, embedding dimensions, cosine similarity.

---

### Day 3 — Storage & Retrieval
| Status | Topic | File | Est. Time |
|--------|-------|------|-----------|
| ☐ | Vector Databases (FAISS, Chroma, Pinecone, Qdrant) | `05_Vector_Databases.ipynb` | 2 hrs |
| ☐ | Retrievers (Similarity, MMR, Hybrid, Ensemble) | `06_Retrievers.ipynb` | 2 hrs |
| ☐ | Day 3 Self-Assessment | Review quiz sections in both notebooks | 30 min |

**Key Exam Topics**: DB selection criteria, dense vs sparse retrieval, MMR, hybrid search, metadata filtering.

---

### Day 4 — Integration & Prompting
| Status | Topic | File | Est. Time |
|--------|-------|------|-----------|
| ☐ | RAG Chain End-to-End | `07_RAG_Chain_End2End.ipynb` | 2 hrs |
| ☐ | Prompting for RAG | `08_Prompting_for_RAG.ipynb` | 1.5 hrs |
| ☐ | Day 4 Self-Assessment | Review quiz sections in both notebooks | 30 min |

**Key Exam Topics**: RetrievalQA chain types (stuff/map_reduce/refine), LCEL, prompt templates, output parsers.

---

### Day 5 — Evaluation, Revision & Practice
| Status | Topic | File | Est. Time |
|--------|-------|------|-----------|
| ☐ | Evaluation & Optimization | `09_Evaluation_and_Optimization.md` | 1 hr |
| ☐ | Quick Reference Cheat Sheet | `10_Quick_Reference_CheatSheet.md` | 1 hr |
| ☐ | Practice MCQs (50+ questions) | `11_Practice_MCQs.ipynb` | 2 hrs |
| ☐ | Re-review weak areas | Revisit flagged topics | 1 hr |

**Key Exam Topics**: RAGAS metrics, failure modes, optimization, scenario-based problem solving.

---

### Exam Day — Quick Revision
| Status | Topic | File | Est. Time |
|--------|-------|------|-----------|
| ☐ | Final rapid revision | `10_Quick_Reference_CheatSheet.md` | 30 min |

---

## 🎯 High-Priority Exam Topics (Star these!)

Based on common L2 exam patterns:

1. **⭐ RAG vs Fine-tuning** — When to use which, tradeoffs
2. **⭐ Chunking Tradeoffs** — chunk_size too large/small effects
3. **⭐ Vector DB Selection** — Scenario-based: "given X requirement, which DB?"
4. **⭐ Retriever Types** — Similarity vs MMR vs Hybrid, when to use each
5. **⭐ Chain Types** — stuff vs map_reduce vs refine — when to use
6. **⭐ Embedding Dimension Mismatch** — Common debugging scenario
7. **⭐ RAGAS Metrics** — Faithfulness, Answer Relevance, Context Precision/Recall
8. **⭐ Failure Modes** — Hallucination, poor retrieval, context window overflow

---

## 📁 File Index

| # | File | Type | Topic |
|---|------|------|-------|
| 00 | `00_STUDY_ROADMAP.md` | Guide | This file — study schedule |
| 01 | `01_RAG_Fundamentals.ipynb` | Notebook | Core RAG concepts |
| 02 | `02_Data_Parsing_and_Loaders.ipynb` | Notebook | Document loading |
| 03 | `03_Chunking_Strategies.ipynb` | Notebook | Text splitting |
| 04 | `04_Embeddings_Deep_Dive.ipynb` | Notebook | Embedding models |
| 05 | `05_Vector_Databases.ipynb` | Notebook | Vector store setup |
| 06 | `06_Retrievers.ipynb` | Notebook | Retrieval methods |
| 07 | `07_RAG_Chain_End2End.ipynb` | Notebook | Full pipeline |
| 08 | `08_Prompting_for_RAG.ipynb` | Notebook | Prompt engineering |
| 09 | `09_Evaluation_and_Optimization.md` | Guide | Metrics & tuning |
| 10 | `10_Quick_Reference_CheatSheet.md` | Reference | Rapid revision |
| 11 | `11_Practice_MCQs.ipynb` | Practice | 50+ exam questions |

---

> **Tip**: After each notebook, mark the status as ✅ in this roadmap. If you struggled with a topic, mark it with 🔴 and revisit on Day 5.
