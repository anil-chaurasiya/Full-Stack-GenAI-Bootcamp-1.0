# 📖 Module 09: RAG Evaluation & Optimization

## GenAI L2 Exam Preparation

**Topics Covered:**
- RAGAS evaluation framework
- Key metrics: Faithfulness, Answer Relevance, Context Precision/Recall
- Common failure modes and fixes
- Optimization strategies

---

## 1. Why Evaluate RAG?

A RAG pipeline has **multiple components** that can fail independently. Without evaluation, you can't tell **which component** is causing poor results.

```
Bad answers could be caused by:
├── Poor document parsing (data quality)
├── Wrong chunk size (too large/small)
├── Bad embeddings (wrong model)
├── Wrong vector DB settings
├── Poor retrieval (wrong search type, low k)
├── Bad prompt (no grounding rules)
└── Weak LLM (too small, wrong temperature)
```

---

## 2. ⭐ RAGAS Framework (Exam Critical!)

**RAGAS** (Retrieval Augmented Generation Assessment) is the standard framework for evaluating RAG pipelines.

### Core Metrics

| Metric | What it measures | Formula (Intuition) | Range |
|--------|-----------------|---------------------|-------|
| **Faithfulness** | Is the answer supported by the context? | Claims in answer that are in context / Total claims | 0-1 (1 = best) |
| **Answer Relevance** | Does the answer address the question? | How well the answer matches the question intent | 0-1 (1 = best) |
| **Context Precision** | Are the retrieved docs actually relevant? | Relevant docs retrieved / Total docs retrieved | 0-1 (1 = best) |
| **Context Recall** | Did we retrieve ALL relevant docs? | Relevant docs retrieved / Total relevant docs in DB | 0-1 (1 = best) |

### How Each Metric Helps

```
Low Faithfulness    → LLM is hallucinating (not using context)
                      Fix: Better grounding in prompt, lower temperature

Low Answer Relevance → LLM's answer doesn't match the question
                       Fix: Better prompt template, different LLM

Low Context Precision → Retrieving irrelevant documents
                        Fix: Better chunking, better embeddings, MMR

Low Context Recall   → Missing relevant documents in retrieval
                       Fix: Increase k, multi-query retriever, better embeddings
```

### 🎯 Exam Tip
> **Faithfulness** = "Does the answer stick to the facts in the context?"  
> **Answer Relevance** = "Does the answer actually answer the question?"  
> **Context Precision** = "Is the retrieved context actually useful?"  
> **Context Recall** = "Did we find everything that was relevant?"

---

## 3. Common Failure Modes

### Failure Mode 1: Hallucination
- **Symptom**: LLM generates information not in the context
- **Metric**: Low Faithfulness
- **Causes**: No grounding instruction, temperature too high, context too vague
- **Fixes**:
  - Add "Only use provided context" to prompt
  - Set `temperature=0`
  - Add "If you don't know, say so" fallback

### Failure Mode 2: Irrelevant Retrieval
- **Symptom**: Retrieved docs don't match the query
- **Metric**: Low Context Precision
- **Causes**: Poor embeddings, wrong chunk size, keyword mismatch
- **Fixes**:
  - Try better embedding model
  - Adjust chunk_size (smaller for precision)
  - Use hybrid search (dense + BM25)
  - Add metadata filtering

### Failure Mode 3: Missing Context
- **Symptom**: Relevant information exists but isn't retrieved
- **Metric**: Low Context Recall
- **Causes**: k too small, poor query, docs not indexed
- **Fixes**:
  - Increase `k` value
  - Use multi-query retriever
  - Check indexing completeness
  - Use ensemble retriever

### Failure Mode 4: Context Window Overflow
- **Symptom**: Error or truncated input to LLM
- **Metric**: N/A (system error)
- **Causes**: Too many chunks, chunk_size too large, k too high
- **Fixes**:
  - Reduce `k` value
  - Use smaller chunk_size
  - Switch to `map_reduce` chain type
  - Use contextual compression retriever

### Failure Mode 5: Embedding Dimension Mismatch
- **Symptom**: Error when querying vector store
- **Metric**: N/A (system error)
- **Causes**: Changed embedding model without re-indexing
- **Fixes**:
  - Use same model for indexing and querying
  - Re-index all documents with new model
  - Check with `len(embeddings.embed_query('test'))`

---

## 4. Optimization Strategies

### 4.1 Improve Retrieval Quality

| Strategy | How | Impact |
|----------|-----|--------|
| Adjust chunk_size | Smaller = more precise, larger = more context | High |
| Use chunk_overlap | 10-20% of chunk_size prevents boundary issues | Medium |
| Better embedding model | `all-mpnet-base-v2` > `all-MiniLM-L6-v2` | Medium |
| MMR retrieval | Diverse results, avoid redundancy | Medium |
| Hybrid search | Dense + BM25 ensemble | High |
| Multi-query | Generate query variations | Medium-High |
| Metadata filtering | Narrow search scope | Medium |
| Reranking | Re-score results with a cross-encoder | High |

### 4.2 Improve Generation Quality

| Strategy | How | Impact |
|----------|-----|--------|
| Better prompt | Grounding rules, clear format, fallback | High |
| Lower temperature | `temperature=0` for factual answers | Medium |
| Few-shot examples | Show desired output format | Medium |
| Better LLM | Larger model = better reasoning | High |
| Chain type | `refine` for detailed, `stuff` for simple | Medium |

### 4.3 Reduce Latency

| Strategy | How | Impact |
|----------|-----|--------|
| Faster vector DB | FAISS > ChromaDB for speed | Medium |
| Smaller embedding model | MiniLM (384d) vs larger models | Medium |
| Cache embeddings | Don't re-embed unchanged docs | High |
| Reduce k | Fewer docs = less processing | Medium |
| Streaming | Stream LLM output for perceived speed | Medium |

### 4.4 Reduce Cost

| Strategy | How | Impact |
|----------|-----|--------|
| Free embedding models | HuggingFace (local) vs OpenAI (paid) | High |
| Smaller LLM | Llama 3.1 8B vs GPT-4 | High |
| Cache responses | Cache frequent queries | Medium |
| Efficient chunking | Fewer, better chunks = fewer tokens | Medium |

---

## 5. Evaluation Checklist for Production RAG

```
□ Faithfulness > 0.8 (answers grounded in context)
□ Answer Relevance > 0.8 (answers match questions)
□ Context Precision > 0.7 (retrieved docs are useful)
□ Context Recall > 0.7 (all relevant docs are found)
□ Latency < 3 seconds (acceptable response time)
□ No hallucination on edge cases (empty context, off-topic questions)
□ Source attribution works (citations are correct)
□ Error handling in place (API failures, empty results)
```

---

## 🧠 Self-Assessment Quiz

**Q1.** Your RAG system returns answers that sound correct but contain information NOT in the retrieved documents. Which RAGAS metric would be low?

<details>
<summary>Click for Answer</summary>

**Faithfulness** — it measures whether the answer's claims are supported by the retrieved context. Low faithfulness = the LLM is hallucinating beyond the provided context.
</details>

---

**Q2.** You increased `k` from 3 to 20 and now the answers are worse. Why?

<details>
<summary>Click for Answer</summary>

More documents means **lower context precision** — you're including irrelevant documents that dilute the useful context. Also, too much context may **overflow the context window** or confuse the LLM. Solution: Use a moderate k (3-5) with better retrieval (MMR, hybrid search) instead.
</details>

---

**Q3.** Your system retrieves relevant documents but the LLM ignores them and gives its own opinion. What's the fix?

<details>
<summary>Click for Answer</summary>

Add strong **grounding instructions** to the prompt:
- "Only use the provided context to answer"
- "Do NOT use your training knowledge"
- "If the context doesn't contain the answer, say 'I don't know'"
Also set `temperature=0` for deterministic outputs.
</details>

---

**Q4.** A user searches for "FAISS" but doesn't get results about FAISS even though the documents contain information about it. Dense retrieval is being used. What might fix this?

<details>
<summary>Click for Answer</summary>

Use **hybrid search** (ensemble retriever with BM25). Dense retrieval might not rank "FAISS" high because it focuses on semantic meaning, not exact keyword matching. BM25 (sparse retrieval) excels at exact term matching and would find documents containing "FAISS" directly.
</details>

---

## ✅ Module 9 Complete!

**Key Takeaways:**
1. RAGAS metrics: Faithfulness, Answer Relevance, Context Precision, Context Recall
2. Low Faithfulness → hallucination problem → fix the prompt
3. Low Context Precision → irrelevant retrieval → fix chunking/embeddings
4. Low Context Recall → missing docs → increase k, use multi-query
5. Optimization: balance quality, latency, and cost
6. Production RAG needs monitoring and evaluation on real queries
