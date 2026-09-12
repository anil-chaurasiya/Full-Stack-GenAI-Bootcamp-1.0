# 📚 Document Parsing, Chunking & Embeddings — Complete Study Guide (GenAI L2 Exam)

> **Goal**: Master the entire data processing pipeline for RAG — from raw documents to searchable vectors.
> Covers document parsing, text extraction, chunking strategies, embedding models, distance metrics, and indexing.

---

## Table of Contents

1. [The Data Processing Pipeline — Big Picture](#1-the-data-processing-pipeline--big-picture)
2. [Document Parsing & Extraction](#2-document-parsing--extraction)
   - 2.1 Why Parsing Matters
   - 2.2 Document Types & Parsers
   - 2.3 PDF Parsing Deep Dive
   - 2.4 Table Extraction
   - 2.5 OCR for Scanned Documents
   - 2.6 Multimodal Document Processing
3. [LangChain Document Loaders](#3-langchain-document-loaders)
4. [The Document Object](#4-the-document-object)
5. [Chunking Strategies](#5-chunking-strategies)
   - 5.1 Why Chunking is Necessary
   - 5.2 RecursiveCharacterTextSplitter (Default)
   - 5.3 CharacterTextSplitter
   - 5.4 TokenTextSplitter
   - 5.5 MarkdownHeaderTextSplitter
   - 5.6 HTMLHeaderTextSplitter
   - 5.7 CodeTextSplitter
   - 5.8 SemanticChunker
   - 5.9 Agentic Chunking
6. [Chunking Parameters — The Art of Tuning](#6-chunking-parameters--the-art-of-tuning)
7. [Document Processing & Summarization Strategies](#7-document-processing--summarization-strategies)
   - 7.1 The Multi-Document Dilemma & Context Limits
   - 7.2 Strategy 1: Stuffing ("Stuff" Documents)
   - 7.3 Strategy 2: Map-Reduce
   - 7.4 Strategy 3: Refine
   - 7.5 Strategy 4: Batch Summarization & Map-Rerank
   - 7.6 Strategy Comparison Matrix
   - 7.7 Exam Traps & Selection Flowchart
8. [Embeddings — From Text to Vectors](#8-embeddings--from-text-to-vectors)
   - 8.1 What are Embeddings?
   - 8.2 How Embeddings Work
   - 8.3 Embedding Model Comparison
   - 8.4 Key Embedding Concepts
9. [Distance Metrics & Similarity Search](#9-distance-metrics--similarity-search)
10. [Vector Indexing Algorithms](#10-vector-indexing-algorithms)
11. [Matryoshka Representation Learning (MRL)](#11-matryoshka-representation-learning-mrl)
12. [End-to-End Pipeline: Parsing → Chunking → Embedding](#12-end-to-end-pipeline-parsing--chunking--embedding)
13. [Best Practices for Production](#13-best-practices-for-production)
14. [Common Pitfalls & Debugging](#14-common-pitfalls--debugging)
15. [Key Terminology Glossary](#15-key-terminology-glossary)
16. [Common Exam Patterns & Traps](#16-common-exam-patterns--traps)

---

## 1. The Data Processing Pipeline — Big Picture

```
┌────────────────────────────────────────────────────────────────────┐
│                  RAG DATA PROCESSING PIPELINE                      │
│                                                                    │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────┐  │
│  │  PARSE   │──▶│  CHUNK   │──▶│  EMBED   │──▶│  INDEX/STORE │  │
│  │          │   │          │   │          │   │              │  │
│  │ Raw docs │   │ Split    │   │ Convert  │   │ Vector DB    │  │
│  │ → text   │   │ into     │   │ text to  │   │ (ChromaDB,   │  │
│  │          │   │ pieces   │   │ vectors  │   │  FAISS,      │  │
│  │ PDF,CSV, │   │          │   │          │   │  Pinecone)   │  │
│  │ HTML,DOCX│   │ Optimal  │   │ Dense    │   │              │  │
│  │ JSON,TXT │   │ size +   │   │ numerical│   │ Indexed for  │  │
│  │          │   │ overlap  │   │ arrays   │   │ fast search  │  │
│  └──────────┘   └──────────┘   └──────────┘   └──────────────┘  │
│                                                                    │
│  Quality at each step directly impacts RAG answer quality!        │
│  "Garbage in → Garbage out" applies at EVERY stage.               │
└────────────────────────────────────────────────────────────────────┘
```

### Exam Key ⚡
- This is the **indexing phase** of RAG (happens before any user query).
- Each step's quality affects the final answer: **bad parsing → bad chunks → bad retrieval → bad answers**.
- The same embedding model MUST be used for indexing AND querying.

---

## 2. Document Parsing & Extraction

### 2.1 Why Parsing Matters

Parsing is the **first and most critical step** in the RAG pipeline. If you extract garbage text from your documents, no amount of good chunking or embeddings will save you.

| Challenge | Impact on RAG |
|-----------|--------------|
| Missing text (extraction failure) | Answer not found even though it's in the document |
| Broken tables | Tabular data becomes incomprehensible text |
| Lost headings/structure | Chunks lose hierarchical context |
| OCR errors in scanned docs | Wrong words → wrong embeddings → wrong retrieval |
| Mixed content (text + images) | Images completely ignored in text-only parsing |

### 2.2 Document Types & Parsers

| Document Type | Common Parsers | LangChain Loader |
|---------------|---------------|-----------------|
| **PDF (digital)** | PyPDF, PyMuPDF (fitz), pdfplumber | `PyPDFLoader`, `PyMuPDFLoader` |
| **PDF (scanned)** | Tesseract OCR, EasyOCR + PyMuPDF | `UnstructuredPDFLoader` |
| **Plain Text** | Built-in Python | `TextLoader` |
| **CSV** | Python csv module | `CSVLoader` |
| **JSON** | Python json module | `JSONLoader` |
| **HTML** | BeautifulSoup, Unstructured | `BSHTMLLoader`, `WebBaseLoader` |
| **Word (DOCX)** | python-docx, docx2txt | `Docx2txtLoader` |
| **Markdown** | Built-in | `UnstructuredMarkdownLoader` |
| **PowerPoint** | python-pptx | `UnstructuredPowerPointLoader` |
| **Excel** | openpyxl, pandas | `UnstructuredExcelLoader` |
| **Images** | Vision LLMs (Gemini, GPT-4V) | Custom multimodal pipeline |
| **Entire directories** | Multiple loaders | `DirectoryLoader` |

### 2.3 PDF Parsing Deep Dive

PDFs are the most common document type in RAG — and the hardest to parse correctly.

#### Parser Comparison

| Parser | Speed | Text Quality | Tables | Images | Scanned PDFs | Best For |
|--------|:---:|:---:|:---:|:---:|:---:|---------|
| **PyPDF** | 🟢 Fast | 🟡 Basic | ❌ | ❌ | ❌ | Simple text-only PDFs |
| **PyMuPDF (fitz)** | 🟢 Fast | 🟢 Good | ⚠️ Basic | ✅ Extract | ❌ | General-purpose, fast |
| **pdfplumber** | 🟡 Medium | 🟢 Good | ✅ Excellent | ❌ | ❌ | Table-heavy PDFs |
| **Unstructured** | 🔴 Slow | 🟢 Excellent | ✅ Good | ✅ Good | ✅ (with OCR) | Complex layouts |
| **LlamaParse** | 🔴 Slow (API) | 🟢 Excellent | ✅ Excellent | ✅ Good | ✅ | Maximum quality |
| **Docling** | 🟡 Medium | 🟢 Excellent | ✅ Excellent | ✅ Good | ✅ | Open-source best |

#### The Hybrid Approach (Best Practice)

```
Incoming PDF → Is it simple text-only?
                    │
              Yes ──┼── No (complex layout/tables/scans)
                    │         │
              PyMuPDF     Unstructured / Docling / LlamaParse
              (fast)       (slower but more accurate)
```

### 2.4 Table Extraction

Tables are a major challenge — standard text extractors destroy table structure.

| Approach | How It Works | Accuracy |
|----------|-------------|:---:|
| **pdfplumber** | Detects cell boundaries geometrically | 🟢 High (native tables) |
| **Camelot** | Lattice/stream table detection | 🟢 High (bordered tables) |
| **Unstructured (hi_res)** | AI-based layout detection | 🟢 High (complex) |
| **Table Transformer (TATR)** | Vision model for table detection | 🟢 High (any table) |
| **Raw text extraction** | No table awareness | 🔴 Very poor |

#### Best Practice: Table → Markdown → Summary

```
Table in PDF → Extract as Markdown table
             → ALSO generate a natural language summary using LLM
             → Embed BOTH the table and the summary
             → Retrieval finds the table via semantic query match
```

### 2.5 OCR for Scanned Documents

| OCR Tool | Type | Quality | Speed |
|----------|------|:---:|:---:|
| **Tesseract** | Open source | 🟡 Medium | 🟢 Fast |
| **EasyOCR** | Open source (GPU) | 🟢 Good | 🟡 Medium |
| **Google Vision API** | Cloud service | 🟢 Excellent | 🟡 Medium |
| **Azure Document Intelligence** | Cloud service | 🟢 Excellent | 🟡 Medium |

### 2.6 Multimodal Document Processing

Modern approach for documents with images, charts, diagrams:

```
PDF Page → Render as Image → Vision LLM (Gemini, GPT-4V)
                            → Generate text description
                            → Embed the description
                            → Retrieve via semantic search
```

This is called **Visual Retrieval** — treats each page as an image and uses vision models for understanding.

### Exam Key ⚡
- **PyPDF** is simplest but loses tables/formatting.
- **PyMuPDF** is the best balance of speed and quality for most use cases.
- **pdfplumber** is best for table extraction.
- **Unstructured** handles the widest range of document types.
- **Markdown output** is preferred over raw text (preserves structure).
- Always **audit your parser output** — check if tables and structure are preserved.

---

## 3. LangChain Document Loaders

### Commonly Tested Loaders

```python
# Plain text
from langchain_community.document_loaders import TextLoader
loader = TextLoader("file.txt", encoding="utf-8")

# PDF
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("file.pdf")  # One Document per page

# CSV — each row becomes a Document
from langchain_community.document_loaders import CSVLoader
loader = CSVLoader("data.csv")

# Web page
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://example.com")

# Entire directory
from langchain_community.document_loaders import DirectoryLoader
loader = DirectoryLoader("./docs", glob="*.pdf", loader_cls=PyPDFLoader)

# JSON with jq schema
from langchain_community.document_loaders import JSONLoader
loader = JSONLoader("data.json", jq_schema=".items[].text")

# Load documents
documents = loader.load()
```

### Exam Key ⚡
- All loaders return `list[Document]` — standardized format.
- `PyPDFLoader` creates **one Document per page** with `metadata["page"]`.
- `CSVLoader` creates **one Document per row**.
- `DirectoryLoader` loads **all matching files** in a directory.
- Always specify `encoding="utf-8"` for text files on Windows.

---

## 4. The Document Object

```python
from langchain_core.documents import Document

doc = Document(
    page_content="The actual text content of the document...",
    metadata={
        "source": "report.pdf",
        "page": 3,
        "author": "John Doe",
        "created": "2025-01-15"
    }
)
```

### Key Properties

| Property | Type | Description |
|----------|------|-------------|
| `page_content` | `str` | The text content |
| `metadata` | `dict` | Source tracking, page numbers, custom fields |

### Why Metadata Matters

- **Source tracking**: Know which document an answer came from (citations).
- **Filtering**: Retrieve only from specific sources, date ranges, etc.
- **Debugging**: Trace retrieval issues back to the source.
- **Metadata is preserved** when splitting with `split_documents()`.

---

## 5. Chunking Strategies

### 5.1 Why Chunking is Necessary

```
Problem: Documents are too large to fit in an LLM's context window
         AND too large for meaningful embedding (semantic dilution)

Solution: Split into smaller, semantically meaningful chunks

┌───────────────────────────────────────────────┐
│           Original Document (50 pages)         │
│                                               │
│  Too large to embed meaningfully.             │
│  Embedding captures the "average meaning"     │
│  which is too vague for precise retrieval.    │
└───────────────┬───────────────────────────────┘
                │
                ▼
┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
│Chunk1│ │Chunk2│ │Chunk3│ │Chunk4│ │Chunk5│ ...
│      │ │      │ │      │ │      │ │      │
│ 500  │ │ 500  │ │ 500  │ │ 500  │ │ 500  │
│ chars│ │ chars│ │ chars│ │ chars│ │ chars│
└──────┘ └──────┘ └──────┘ └──────┘ └──────┘

Each chunk has a focused, specific meaning →
Better embeddings → Better retrieval → Better answers
```

### The Chunking Tradeoff

| Too Small Chunks | Too Large Chunks |
|:---:|:---:|
| Lose context | Dilute meaning |
| May split sentences | Noise in retrieval |
| More chunks to search | Fewer chunks (may miss info) |
| Higher precision | Lower precision |
| Lower recall | Higher recall |

---

### 5.2 RecursiveCharacterTextSplitter ⭐ (DEFAULT)

**THE most important splitter to know for the exam.**

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_documents(documents)
```

#### How It Works

```
Input Text
    │
    ▼
Try split by "\n\n" (paragraphs)
    │
    ├── Chunks ≤ chunk_size? → Done ✅
    │
    └── Chunks still too large?
            │
            ▼
        Try split by "\n" (lines)
            │
            ├── Chunks ≤ chunk_size? → Done ✅
            │
            └── Still too large?
                    │
                    ▼
                Try split by ". " (sentences)
                    │
                    ├── Done ✅
                    │
                    └── Try " " (words) → "" (chars)
```

#### Default Separators (Priority Order)

| Priority | Separator | Splits By |
|:---:|----------|-----------|
| 1 | `\n\n` | Paragraphs |
| 2 | `\n` | Lines |
| 3 | `. ` | Sentences |
| 4 | `" "` | Words |
| 5 | `""` | Characters (last resort) |

### Exam Key ⚡
- **RecursiveCharacterTextSplitter is the DEFAULT recommendation.**
- It preserves semantic units as much as possible.
- `chunk_overlap` prevents information loss at boundaries.
- `split_documents()` preserves metadata; `split_text()` works on raw strings.

---

### 5.3 CharacterTextSplitter

Splits on a **single separator** only.

```python
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator="\n\n",     # Only splits on double newlines
    chunk_size=500,
    chunk_overlap=50,
)
```

**Limitation**: If no separator is found, the entire text becomes one chunk. **Less flexible** than Recursive.

---

### 5.4 TokenTextSplitter

Splits based on **token count** rather than character count.

```python
from langchain_text_splitters import TokenTextSplitter

splitter = TokenTextSplitter(
    chunk_size=500,       # 500 tokens per chunk
    chunk_overlap=50,     # 50 token overlap
)
```

**When to Use**: When you need precise control over token counts (e.g., fitting within model context windows).

---

### 5.5 MarkdownHeaderTextSplitter

Splits Markdown documents by **heading hierarchy**, preserving document structure.

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

headers = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
chunks = splitter.split_text(markdown_text)
# Each chunk's metadata includes its heading hierarchy
```

**Best For**: Documentation, wikis, README files.

---

### 5.6 HTMLHeaderTextSplitter

Similar to Markdown splitter but for HTML documents — splits by `<h1>`, `<h2>`, `<h3>`, etc.

---

### 5.7 CodeTextSplitter

Language-aware splitting for source code.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=2000,
    chunk_overlap=200,
)
```

**Supported Languages**: Python, JavaScript, Java, Go, Ruby, Rust, C++, and many more.

---

### 5.8 SemanticChunker

Uses **embedding similarity** to find natural break points.

```python
from langchain_experimental.text_splitter import SemanticChunker

splitter = SemanticChunker(
    embeddings=embedding_model,
    breakpoint_threshold_type="percentile"  # or "standard_deviation"
)
```

**How It Works**: Embeds consecutive sentences, measures similarity. When similarity drops significantly → split boundary.

**Best For**: When you want chunks that are truly semantically coherent.

---

### 5.9 Agentic Chunking

Uses an **LLM** to decide where to split — the most intelligent but most expensive approach.

```
LLM reads the document → decides "this paragraph belongs with the next one"
                        → creates semantically complete chunks
```

**Trade-off**: Highest quality but slowest and most expensive.

---

### Splitter Comparison Table

| Splitter | Intelligence | Speed | Cost | Best For |
|----------|:---:|:---:|:---:|---------|
| **RecursiveCharacter** | 🟡 Rule-based | 🟢 Fast | 🟢 Free | General text (DEFAULT) |
| **Character** | 🔴 Basic | 🟢 Fast | 🟢 Free | Single-separator needs |
| **Token** | 🟡 Token-aware | 🟢 Fast | 🟢 Free | Token-count precision |
| **Markdown/HTML** | 🟡 Structure-aware | 🟢 Fast | 🟢 Free | Structured documents |
| **Code** | 🟡 Language-aware | 🟢 Fast | 🟢 Free | Source code |
| **Semantic** | 🟢 Embedding-based | 🟡 Medium | 🟡 Embedding cost | Semantic coherence |
| **Agentic** | 🟢 LLM-powered | 🔴 Slow | 🔴 LLM cost | Maximum quality |

---

## 6. Chunking Parameters — The Art of Tuning

### chunk_size

| Value | Effect | Use Case |
|-------|--------|----------|
| **100-300** | Very small, precise | Short Q&A, factoid retrieval |
| **500-1000** | Balanced (RECOMMENDED) | General RAG, most use cases |
| **1000-2000** | Large, more context | Summarization, complex reasoning |
| **2000+** | Very large | Long-form analysis |

### chunk_overlap

| Strategy | Overlap | Effect |
|----------|---------|--------|
| No overlap | `0` | Information loss at boundaries |
| Low overlap | `10-15%` of chunk_size | Minimal redundancy, some loss |
| Standard overlap | `15-20%` of chunk_size | **RECOMMENDED** — good balance |
| High overlap | `25-50%` | Maximum continuity, more redundancy |

### Recommended Defaults

```
chunk_size = 1000 characters
chunk_overlap = 200 characters (20% of chunk_size)
splitter = RecursiveCharacterTextSplitter
```

### Exam Key ⚡
- **Recommended**: `chunk_size=500-1000`, `chunk_overlap=10-20%` of chunk_size.
- `chunk_overlap` **prevents information loss** at chunk boundaries.
- **Too small** chunks → context loss. **Too large** chunks → semantic dilution.
- Always test with your actual data and evaluate retrieval quality.

---

## 7. Document Processing & Summarization Strategies

When working with large documents (reports, legal contracts, research papers, books) or large sets of retrieved chunks, a single LLM call is often not enough or impossible due to **context window limits**, **cost**, and the **"Lost-in-the-Middle" phenomenon**.

LangChain provides four primary architectural strategies to process, synthesize, and summarize documents:
1. **Stuffing ("Stuff")**
2. **Map-Reduce**
3. **Refine**
4. **Batch Processing / Map-Rerank**

---

### 7.1 The Multi-Document Dilemma & Context Limits

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    THE MULTI-DOCUMENT DILEMMA                           │
│                                                                         │
│  User wants: "Summarize this 150-page financial report" (100k tokens)   │
│                                                                         │
│  Problem 1: Context Window Ceiling                                      │
│  - Older/smaller LLMs have 4k-32k token limits. You literally CANNOT     │
│    stuff all pages into one prompt.                                     │
│                                                                         │
│  Problem 2: "Lost in the Middle" Effect (Liu et al., Stanford)          │
│  - LLMs recall facts at the BEGINNING (primacy) and END (recency)       │
│    of prompts with high accuracy, but miss critical facts buried        │
│    in the middle of massive context windows.                            │
│                                                                         │
│  Problem 3: Latency & Cost                                              │
│  - Processing 100k tokens in a single request has high Time-To-First-   │
│    Token (TTFT) and can be expensive on commercial APIs.                │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### 7.2 Strategy 1: Stuffing ("Stuff" Documents)

#### Concept
The simplest approach: concatenate all chunks/documents together with delimiters and insert ("stuff") them into a single prompt sent to the LLM.

#### Architecture
```
┌─────────────┐
│ Document 1  │──┐
└─────────────┘  │
┌─────────────┐  │    ┌───────────────────────────┐     ┌───────────┐     ┌────────────────┐
│ Document 2  │──┼───▶│  Single Prompt with All   │────▶│    LLM    │────▶│ Final Response │
└─────────────┘  │    │  Concatenated Documents   │     │ (1 Call)  │     │   or Summary   │
┌─────────────┐  │    └───────────────────────────┘     └───────────┘     └────────────────┘
│ Document 3  │──┘
└─────────────┘
```

#### LangChain Implementation
```python
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_template(
    """Write a concise summary of the following documents:
    
    {context}
    
    CONCISE SUMMARY:"""
)

# Modern LCEL pattern:
stuff_chain = create_stuff_documents_chain(llm, prompt)
summary = stuff_chain.invoke({"context": docs})

# Classic LangChain legacy pattern:
# from langchain.chains.summarize import load_summarize_chain
# chain = load_summarize_chain(llm, chain_type="stuff")
```

#### Pros & Cons
| Pros 🟢 | Cons 🔴 |
|---|---|
| **Only 1 LLM call**: Lowest API cost and fastest wall-clock execution when it fits | **Context window limit**: Fails completely if combined documents exceed context window |
| **Full global context**: LLM sees all text simultaneously, enabling seamless cross-chunk comparisons | **Lost in the middle**: Performance degrades as document size approaches token limit |
| **No intermediate state**: Simpler orchestration, no intermediate hallucination risks | **Memory heavy**: High prompt token consumption in a single shot |

#### Exam Key ⚡
- Best for: Small documents, top-3 to top-5 retrieved chunks in standard RAG pipelines (< 4,000–8,000 tokens).
- Default chain in most LangChain RAG templates (`create_stuff_documents_chain`).

---

### 7.3 Strategy 2: Map-Reduce

#### Concept
A distributed, divide-and-conquer approach:
1. **Map Phase**: Run an LLM prompt on each document/chunk **independently and in parallel** to extract key points or generate sub-summaries.
2. **Reduce Phase**: Take all intermediate summaries, combine them into a single prompt, and run an LLM call to produce the final synthesized output.
3. **Iterative Collapsing (Collapse Phase)**: If the intermediate summaries combined still exceed context limits, LangChain recursively collapses subsets of summaries until they fit the final reduce prompt!

#### Architecture
```
┌─────────────┐     ┌───────────┐     ┌───────────┐
│ Document 1  │────▶│ Map LLM   │────▶│ Summary 1 │──┐
└─────────────┘     └───────────┘     └───────────┘  │
┌─────────────┐     ┌───────────┐     ┌───────────┐  │    ┌───────────────────────────┐     ┌────────────┐     ┌──────────────┐
│ Document 2  │────▶│ Map LLM   │────▶│ Summary 2 │──┼───▶│ Combine Prompt with       │────▶│ Reduce LLM │────▶│ Final Global │
└─────────────┘     └───────────┘     └───────────┘  │    │ Intermediate Summaries    │     │  (1 Call)  │     │   Summary    │
┌─────────────┐     ┌───────────┐     ┌───────────┐  │    └───────────────────────────┘     └────────────┘     └──────────────┘
│ Document 3  │────▶│ Map LLM   │────▶│ Summary 3 │──┘
└─────────────┘     └───────────┘     └───────────┘
         ▲                                   
         └────── RUNS IN PARALLEL! ───────┘
```

#### LangChain Implementation
```python
from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

map_template = """The following is a section of a document:
{text}
Identify and summarize the key facts in this section:
SUMMARY:"""
map_prompt = PromptTemplate.from_template(map_template)

reduce_template = """The following is a set of summaries:
{text}
Synthesize these into a cohesive, comprehensive final summary:
FINAL SUMMARY:"""
reduce_prompt = PromptTemplate.from_template(reduce_template)

# Using classic load_summarize_chain with map_reduce
map_reduce_chain = load_summarize_chain(
    llm=llm,
    chain_type="map_reduce",
    map_prompt=map_prompt,
    combine_prompt=reduce_prompt,
    return_intermediate_steps=True  # Helpful for inspection!
)

result = map_reduce_chain.invoke({"input_documents": docs})
final_summary = result["output_text"]
```

#### Pros & Cons
| Pros 🟢 | Cons 🔴 |
|---|---|
| **Arbitrary scale**: Can process documents of **unlimited size** (100s or 1,000s of pages) | **Higher total cost**: Requires `N + 1` LLM calls (N map calls + 1 reduce call) |
| **Highly parallelizable**: All Map calls can run concurrently, keeping wall-clock latency low | **Context boundary loss**: Map prompts only see isolated chunks; cross-chunk references are missed |
| **Hierarchical collapsing**: Handles cases where intermediate summaries exceed context limit | **Information degradation**: Final summary is a "summary of summaries", losing fine-grained detail |

#### Exam Key ⚡
- Total LLM calls: `N + 1` (where N is number of chunks), plus extra collapse calls if needed.
- Best for: Very large documents (entire books, annual reports) where chunks can be evaluated independently.
- **Trap**: Does Map-Reduce preserve chronological narrative? No, intermediate chunks are evaluated in isolation during the map step.

---

### 7.4 Strategy 3: Refine

#### Concept
An iterative, sequential, rolling-update pattern:
1. Process Document 1 with an `initial_prompt` to generate Summary 1.
2. Pass Summary 1 **along with Document 2** into a `refine_prompt` to update/refine the summary.
3. Pass Summary 2 along with Document 3 into the next step, repeating sequentially until all documents are processed.

#### Architecture
```
┌─────────────┐
│ Document 1  │────▶ [ Initial LLM Call ] ──▶ Summary 1
└─────────────┘                                     │
                                                   ▼
┌─────────────┐                       ┌─────────────────────────┐
│ Document 2  │──────────────────────▶│ Refine LLM Call (Doc 2   │──▶ Summary 2
└─────────────┘                       │ + Existing Summary 1)   │          │
                                      └─────────────────────────┘          ▼
┌─────────────┐                                               ┌─────────────────────────┐
│ Document 3  │──────────────────────────────────────────────▶│ Refine LLM Call (Doc 3   │──▶ Final
└─────────────┘                                               │ + Existing Summary 2)   │    Summary
                                                              └─────────────────────────┘
         ▲
         └────── STRICTLY SEQUENTIAL (NO PARALLELISM!) ───────┘
```

#### LangChain Implementation
```python
from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

initial_prompt = PromptTemplate.from_template(
    """Provide an initial concise summary of this first section:
    
    {text}
    
    INITIAL SUMMARY:"""
)

refine_prompt = PromptTemplate.from_template(
    """You have an existing summary:
    {existing_answer}
    
    We have additional context below:
    {text}
    
    Refine and update the existing summary with the new context. 
    If the context isn't useful, return the existing summary.
    REFINED SUMMARY:"""
)

refine_chain = load_summarize_chain(
    llm=llm,
    chain_type="refine",
    question_prompt=initial_prompt,
    refine_prompt=refine_prompt,
    return_intermediate_steps=True
)

result = refine_chain.invoke({"input_documents": docs})
```

#### Pros & Cons
| Pros 🟢 | Cons 🔴 |
|---|---|
| **Cumulative context**: Builds a continuous, coherent narrative over time | **Strictly sequential (Slowest)**: Cannot run in parallel. Wall-clock latency = N × single_call_latency |
| **High detail preservation**: Can incorporate specific nuances from every page | **Error & Hallucination propagation**: Early mistakes or bias can compound through subsequent steps |
| **Ideal for chronological sequences**: Great for timeline analysis and meeting transcripts | **Recency / Directional bias**: Later documents might disproportionately alter the final summary |

#### Exam Key ⚡
- Total LLM calls: `N` calls, but **strictly sequential**.
- **Exam Question Trap**: "Which summarization strategy CANNOT be parallelized?" → **Refine**.
- **Exam Question Trap**: "Which strategy passes the previous answer into the next prompt?" → **Refine** (`existing_answer`).

---

### 7.5 Strategy 4: Batch Summarization & Map-Rerank

#### Concept: Batch Summarization & Chunk Batching
Instead of sending 1 chunk per call (Map-Reduce) or all chunks in 1 call (Stuff), **batching** groups chunks into sub-batches (e.g., 3-5 chunks per prompt):
- Minimizes HTTP connection overhead and API roundtrips.
- Fits within context budget while cutting down total LLM calls from `N` to `N / batch_size`.
- Async batch execution with `llm.abatch()` or LangChain's `batch()` leverages concurrency under API rate limits.

#### Concept: Map-Rerank Strategy
Map-Rerank is specifically designed for Question Answering across multiple documents:
1. **Map Step**: Each chunk is evaluated independently with a prompt asking:
   - "Does this chunk answer the question? If so, provide the answer AND give a score from 1-100 on your confidence."
2. **Rerank Step**: The system parses out the scores, sorts the candidate answers in descending order, and selects the answer with the **highest confidence score**.

```
Doc 1 ──▶ [LLM] ──▶ Answer: "Revenue grew 14%" (Score: 95)  ──┐
Doc 2 ──▶ [LLM] ──▶ Answer: "Not mentioned"     (Score: 0)   ──┼─▶ Winner: Doc 1 (Score: 95)
Doc 3 ──▶ [LLM] ──▶ Answer: "Q3 revenue was up" (Score: 60)  ──┘
```

#### LangChain Implementation (Map-Rerank)
```python
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Classic Map-Rerank chain for Question Answering
qa_chain = load_qa_chain(llm=llm, chain_type="map_rerank", return_intermediate_steps=True)
result = qa_chain.invoke({"input_documents": docs, "question": "What was the Q4 net profit?"})

print("Best Answer:", result["output_text"])
```

---

### 7.6 Comprehensive Strategy Comparison Matrix

| Feature / Dimension | Stuff ("Stuffing") | Map-Reduce | Refine | Map-Rerank / Batch |
|---|:---:|:---:|:---:|:---:|
| **Execution Pattern** | Single shot | Parallel Map → Combined Reduce | Sequential rolling loop | Parallel Map + Scoring / Chunk grouping |
| **Number of LLM Calls** | `1` | `N + 1` (+ collapse calls) | `N` | `N` (or `N / batch_size`) |
| **Wall-Clock Latency** | 🟢 Lowest (1 call) | 🟡 Medium (Map is parallel) | 🔴 Highest (N sequential calls) | 🟡 Medium (Parallelized) |
| **Total Token / API Cost** | 🟢 Lowest | 🔴 High | 🔴 High | 🟡 Medium to High |
| **Max Document Size** | Limited to context window | **Unlimited** (hierarchical) | **Unlimited** | **Unlimited** |
| **Can Be Parallelized?** | N/A (1 call) | 🟢 **Yes** (Map phase) | 🔴 **No** (Strictly sequential) | 🟢 **Yes** |
| **Cross-Chunk Reasoning** | 🟢 Highest (all in prompt) | 🟡 Medium (via intermediate summaries) | 🟢 High (continuous update) | 🔴 Low (evaluates chunks independently) |
| **Vulnerable to "Lost in Middle"**| 🔴 High risk for large prompts | 🟢 Low (each chunk processed small) | 🟢 Low | 🟢 Low |
| **Risk of Error Propagation** | None | Low | 🔴 High (early error persists) | None |
| **Best Suited For** | 3-5 short chunks, fast Q&A | Large books, contracts, annual reports | Chronological events, meeting transcripts | Targeted factual Q&A where one chunk holds answer |

---

### 7.7 Exam Traps & Selection Flowchart

#### Exam Flashcard Questions 💡
1. **"Which chain type is best when all documents comfortably fit in the model's context window?"**
   → **Stuff**: Simplest, fastest, lowest cost, best global context.
2. **"Which chain type should be chosen when summarizing a 1,000-page book where parallel processing is needed to minimize latency?"**
   → **Map-Reduce**: Map phase runs concurrently across chunks.
3. **"Which chain type is required when the order of documents is chronological and each step must build upon previous reasoning?"**
   → **Refine**: Feeds `existing_answer` sequentially into the next chunk.
4. **"What is the collapse step in Map-Reduce?"**
   → When the mapped summaries combined exceed the reduce prompt context window, they are iteratively grouped and summarized until they fit.
5. **"Why can't the Refine chain be parallelized?"**
   → Step `k` strictly requires the output of step `k - 1` as `existing_answer`.

#### Strategy Selection Flowchart
```
Are all documents/chunks ≤ context window?
     ├── YES ──▶ Use STUFF (create_stuff_documents_chain) ⭐
     │
     └── NO  ──▶ Does chronological/cumulative narrative matter?
                   ├── YES ──▶ Use REFINE (RefineDocumentsChain)
                   │
                   └── NO  ──▶ Need a single best factual answer from candidate chunks?
                                 ├── YES ──▶ Use MAP-RERANK (scores candidate answers)
                                 │
                                 └── NO  ──▶ Use MAP-REDUCE (MapReduceDocumentsChain) ⭐
```

---

## 8. Embeddings — From Text to Vectors

### 7.1 What are Embeddings?

**Embeddings** are dense numerical vectors (arrays of floats) that represent the **semantic meaning** of text. Texts with similar meanings have vectors that are close together in the embedding space.

```
"The cat sat on the mat"  →  [0.12, -0.45, 0.78, ..., 0.33]  (384 dimensions)
"A kitten was on the rug" →  [0.11, -0.44, 0.76, ..., 0.31]  (similar vector!)
"Stock market crashed"    →  [0.89, 0.23, -0.67, ..., -0.15] (very different)
```

### 7.2 How Embeddings Work

```
┌─────────────────────────────────────────────────────┐
│                 EMBEDDING MODEL                      │
│                                                     │
│  Input Text ──▶ Tokenizer ──▶ Neural Network ──▶ Vector
│                                                     │
│  "RAG combines        [101, 15334,     [0.12, -0.45,│
│   retrieval with       8883, 2007,      0.78, 0.56,│
│   generation"          4245, ...]       ..., 0.33] │
│                                                     │
│                                    384/768/1536 dims│
└─────────────────────────────────────────────────────┘
```

### 7.3 Embedding Model Comparison

| Model | Provider | Dimensions | Cost | Quality | Best For |
|-------|----------|:---:|:---:|:---:|---------|
| `all-MiniLM-L6-v2` | HuggingFace | 384 | 🟢 Free (local) | 🟡 Good | Development, prototyping |
| `all-mpnet-base-v2` | HuggingFace | 768 | 🟢 Free (local) | 🟢 Better | Production (self-hosted) |
| `text-embedding-3-small` | OpenAI | 1536 | 🟡 Paid API | 🟢 Excellent | General production |
| `text-embedding-3-large` | OpenAI | 3072 | 🟡 Paid API | 🟢 Best | Maximum quality |
| `text-embedding-004` | Google | 768 | 🟡 Paid API | 🟢 Excellent | Google Cloud ecosystem |
| `gemini-embedding-001` | Google | 3072 | 🟡 Paid API | 🟢 Best | Multimodal, latest |
| `voyage-3` | Voyage AI | 1024 | 🟡 Paid API | 🟢 Excellent | Code & technical docs |

### 7.4 Key Embedding Concepts

#### embed_query vs embed_documents

| Method | Purpose | When to Use |
|--------|---------|-------------|
| `embed_query(text)` | Embed a single text | User's search query |
| `embed_documents(texts)` | Embed a batch of texts | Indexing documents |

Some models use different internal processing for queries vs documents (e.g., adding a task prefix).

#### The Golden Rule

> **ALWAYS use the SAME embedding model for indexing and querying.**
> 
> Different models produce vectors in incompatible spaces.
> Comparing vectors from different models gives random results.

#### Dimensionality

| Dimensions | Storage Per Vector | Notes |
|:---:|:---:|-------|
| 384 | ~1.5 KB | Fast, small, good for dev |
| 768 | ~3 KB | Balanced |
| 1536 | ~6 KB | Industry standard (OpenAI small) |
| 3072 | ~12 KB | Maximum quality, higher cost |

Higher dimensions ≠ always better. The **curse of dimensionality** means more dimensions can sometimes hurt retrieval performance.

---

## 9. Distance Metrics & Similarity Search

### Metric Comparison

| Metric | Formula | Range | When to Use |
|--------|---------|:---:|-------------|
| **Cosine Similarity** | cos(θ) = (A·B)/(‖A‖·‖B‖) | [-1, 1] | **Default for text** — measures angle, robust to text length |
| **Euclidean Distance (L2)** | ‖A - B‖₂ | [0, ∞) | When magnitude matters; with normalized vectors = same as cosine |
| **Dot Product** | A · B | (-∞, ∞) | When models are trained with dot-product loss |
| **Manhattan Distance (L1)** | Σ\|Aᵢ - Bᵢ\| | [0, ∞) | Sparse data, high dimensions |

### Cosine Similarity Visual

```
         High Similarity (cos ≈ 1.0)      Low Similarity (cos ≈ 0.0)
              ↗ Vector B                        ↑ Vector B
             /                                  |
            / small angle                       | 90° angle
           /                                    |
Vector A ──────▶                    Vector A ───────▶
```

### Exam Key ⚡
- **Cosine similarity is the industry standard** for text embeddings.
- Cosine similarity measures **angle** (direction), not magnitude — robust to text length variation.
- When vectors are **normalized** (unit length), cosine similarity and Euclidean distance give **identical rankings**.
- Higher cosine similarity = more semantically similar.
- Range: -1 (opposite) to 1 (identical). Most text similarities fall in [0, 1].

---

## 10. Vector Indexing Algorithms

### Common Algorithms

| Algorithm | Full Name | Speed | Recall | Memory | Best For |
|-----------|-----------|:---:|:---:|:---:|---------|
| **Flat (Brute Force)** | Exact search | 🔴 Slow | 🟢 100% | 🔴 High | Small datasets (<10K) |
| **HNSW** | Hierarchical Navigable Small World | 🟢 Fast | 🟢 ~99% | 🟡 Medium | **Production standard** |
| **IVF** | Inverted File Index | 🟢 Fast | 🟡 ~95% | 🟢 Low | Large datasets |
| **PQ** | Product Quantization | 🟢 Very fast | 🟡 ~90% | 🟢 Very low | Massive datasets (100M+) |
| **IVF-PQ** | Combined IVF + PQ | 🟢 Very fast | 🟡 ~93% | 🟢 Very low | Massive datasets |

### HNSW (Most Important for Exam)

```
HNSW — Hierarchical Navigable Small World

Layer 2 (sparse):    [A] ─────────────── [M]
                      │                    │
Layer 1 (medium):    [A] ── [D] ── [H] ── [M]
                      │      │      │      │
Layer 0 (dense):     [A][B][C][D][E][F][G][H][I][J][K][L][M]

Search: Start at top layer (fast, coarse navigation)
        → Descend to lower layers (slow, precise navigation)
        → Find nearest neighbors at bottom layer
```

### Exam Key ⚡
- **HNSW is the default in most vector databases** (ChromaDB, Qdrant, pgvector).
- HNSW provides **fast approximate nearest neighbor (ANN)** search with ~99% recall.
- **Flat search** is 100% accurate but doesn't scale.
- **ANN** = Approximate Nearest Neighbor — sacrifices tiny accuracy for massive speed gains.

---

## 11. Matryoshka Representation Learning (MRL)

A modern technique that allows **flexible dimensionality** from a single model.

```
Full vector (3072 dims): [0.12, -0.45, 0.78, ..., 0.33, -0.21, 0.56]
                         ─────────────────────────────────────────────
                         ████████████████████████████████████████████

Truncated to 1536 dims:  [0.12, -0.45, 0.78, ..., 0.33]
                         ████████████████████████

Truncated to 512 dims:   [0.12, -0.45, ..., 0.78]
                         ████████████

Each truncation retains most of the semantic information!
```

### Key Points
- Supported by OpenAI `text-embedding-3`, Google `gemini-embedding-001`, Jina.
- Use the `dimensions` parameter to set output size.
- Trade storage/speed for a small accuracy loss.
- Example: 3072 → 512 dims = ~6× storage savings with ~2% accuracy loss.

---

## 12. End-to-End Pipeline: Parsing → Chunking → Embedding

```python
# ━━━ COMPLETE RAG INDEXING PIPELINE ━━━

# Step 1: PARSE — Load documents
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("report.pdf")
documents = loader.load()

# Step 2: CHUNK — Split into pieces
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = splitter.split_documents(documents)

# Step 3: EMBED + STORE — Convert to vectors and index
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Step 4: RETRIEVE — Create retriever for query time
retriever = vectorstore.as_retriever(
    search_type="mmr",       # Maximum Marginal Relevance
    search_kwargs={"k": 4}   # Top-4 results
)
```

---

## 13. Best Practices for Production

### Document Parsing ✅
1. **Audit parser output** — check if tables, headings, and formatting are preserved
2. **Use hybrid parsing** — fast parser for simple docs, AI parser for complex ones
3. **Convert tables to Markdown** — LLMs understand Markdown well
4. **Generate table summaries** — embed both the table and its natural-language summary
5. **Handle encoding** — always specify `encoding="utf-8"` on Windows

### Chunking ✅
1. **RecursiveCharacterTextSplitter** as the default
2. `chunk_size=500-1000`, `chunk_overlap=100-200` (10-20%)
3. Use **structure-aware splitters** (Markdown, HTML, Code) for structured documents
4. Use `split_documents()` to **preserve metadata**
5. **Test and evaluate** — measure retrieval quality with different chunk sizes

### Embeddings ✅
1. **Same model for indexing AND querying** — non-negotiable
2. **Specify model explicitly** — never rely on defaults
3. Use **MRL/dimension reduction** to balance storage vs accuracy
4. **Cosine similarity** as the default distance metric
5. For local/free: `all-MiniLM-L6-v2` (384d). For production API: `text-embedding-3-small` (1536d)

---

## 14. Common Pitfalls & Debugging

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Different embedding models for index/query | Random, irrelevant results | Use SAME model for both |
| Chunks too small | Good retrieval but answer lacks context | Increase `chunk_size` |
| Chunks too large | Retrieved chunks contain irrelevant noise | Decrease `chunk_size` |
| No chunk overlap | Answers at chunk boundaries are missed | Add `chunk_overlap` (10-20%) |
| Broken tables in parsing | Table data is garbled text | Use pdfplumber or Unstructured |
| No metadata preserved | Can't trace answers to source | Use `split_documents()` not `split_text()` |
| Wrong distance metric | Poor retrieval quality | Match metric to model training (usually cosine) |
| Encoding errors (Windows) | UnicodeDecodeError | Specify `encoding="utf-8"` |
| Dimension mismatch | Error during search | Ensure consistent model + rebuild index |

---

## 15. Key Terminology Glossary

| Term | Definition |
|------|-----------|
| **Document Parsing** | Extracting text and structure from raw files (PDF, DOCX, etc.) |
| **OCR** | Optical Character Recognition — converting scanned images to text |
| **Document Object** | LangChain object with `page_content` (text) and `metadata` (dict) |
| **Chunking** | Splitting large documents into smaller, meaningful pieces |
| **chunk_size** | Maximum number of characters (or tokens) per chunk |
| **chunk_overlap** | Number of characters shared between consecutive chunks |
| **Recursive Splitting** | Trying separators in priority order (paragraphs → lines → words) |
| **Semantic Chunking** | Using embeddings to find natural split boundaries |
| **Stuffing ("Stuff")** | Putting all document chunks into a single prompt (1 LLM call) |
| **Map-Reduce** | Summarizing chunks in parallel (Map), then synthesizing summaries together (Reduce) |
| **Refine** | Iteratively updating a summary by passing previous summary + next chunk sequentially |
| **Map-Rerank** | Evaluating each chunk independently, assigning a confidence score, picking top score |
| **Embedding** | Dense numerical vector representing text meaning |
| **Embedding Model** | Neural network that converts text to vectors |
| **Dimensionality** | Number of elements in an embedding vector (e.g., 384, 1536) |
| **Cosine Similarity** | Metric measuring angle between vectors (standard for text) |
| **Euclidean Distance** | Metric measuring straight-line distance between vectors |
| **HNSW** | Hierarchical Navigable Small World — fast ANN index algorithm |
| **ANN** | Approximate Nearest Neighbor — fast approximate search |
| **MRL** | Matryoshka Representation Learning — flexible dimensionality |
| **Flat Index** | Brute-force exact search (slow but 100% accurate) |
| **MMR** | Maximum Marginal Relevance — diversity-aware search |
| **Hybrid Search** | Combining dense (semantic) + sparse (keyword) retrieval |

---

## 16. Common Exam Patterns & Traps

### ❓ Frequently Tested Concepts

1. **"What is the recommended default text splitter?"**
   → `RecursiveCharacterTextSplitter` — splits by paragraphs → lines → sentences → words.

2. **"Why do we need chunk_overlap?"**
   → To prevent information loss at chunk boundaries.

3. **"What happens if you use different embedding models for indexing and querying?"**
   → Vectors are in incompatible spaces → similarity search returns random/meaningless results.

4. **"What is cosine similarity?"**
   → Measures the angle between two vectors. Range: [-1, 1]. Standard for text embeddings.

5. **"What is HNSW?"**
   → Hierarchical Navigable Small World — the standard ANN index algorithm used in most vector databases.

6. **"Why are small chunks better for precision but worse for context?"**
   → Small chunks match queries more precisely but may lack surrounding context for the LLM to generate a complete answer.

7. **"What is the purpose of metadata in Document objects?"**
   → Source tracking, filtering during retrieval, citations, and debugging.

8. **"Which parser is best for table extraction from PDFs?"**
   → `pdfplumber` for native tables; `Unstructured` for complex layouts.

9. **"Which summarization strategy CANNOT be parallelized?"**
   → **Refine** — each step strictly requires the previous step's output (`existing_answer`).

10. **"Which strategy uses the fewest LLM calls for documents that fit in the context window?"**
    → **Stuff** — exactly 1 LLM call.

11. **"What happens in the Map-Reduce chain if intermediate summaries exceed the context limit?"**
    → An iterative **collapse phase** recursively groups and summarizes them before the final reduce step.

12. **"Which strategy is best for single-fact QA across many independent chunks?"**
    → **Map-Rerank** — scores each chunk independently and selects the answer with the highest confidence.

### ⚠️ Common Traps

| Trap | Why It's Wrong | Correct Answer |
|------|---------------|----------------|
| "Larger chunks are always better" | Large chunks dilute meaning and add noise | Balance: 500-1000 chars is recommended |
| "Higher embedding dimensions = always better" | Curse of dimensionality; diminishing returns | MRL allows truncation with minimal quality loss |
| "Cosine and Euclidean always give different results" | With normalized vectors they give identical rankings | If vectors are unit-normalized, both metrics are equivalent |
| "OCR is always needed for PDFs" | Only scanned/image PDFs need OCR | Digital PDFs have extractable text layers |
| "split_text() preserves metadata" | Only split_documents() preserves metadata | Use split_documents() with Document objects |
| "RecursiveCharacterTextSplitter only splits on \\n\\n" | It tries \\n\\n first, then \\n, then '. ', then ' ', then '' | Recursive = tries multiple separators in order |
| "Embeddings are the same as one-hot encoding" | One-hot is sparse/binary; embeddings are dense/continuous | Embeddings capture semantic relationships |
| "You can mix embedding models across your pipeline" | Incompatible vector spaces | Must use ONE model for the entire pipeline |
| "Map-Reduce is sequential like Refine" | Map phase runs completely in parallel | Map is parallel; Refine is sequential |
| "Stuffing works on 500-page books" | Exceeds context limits and causes lost-in-the-middle | Use Map-Reduce or hierarchical chunking |
| "Refine is faster than Map-Reduce" | Refine has N sequential steps (slowest wall-clock time) | Map-Reduce is faster due to parallel Map calls |

---

## Quick Decision Flowcharts 🧭

### Choosing a Parser
```
Simple text-only PDF?        → PyPDF / PyMuPDF (fast)
PDF with tables?             → pdfplumber (excellent table support)
Complex layout / scanned?    → Unstructured / Docling (AI-powered)
Maximum quality needed?      → LlamaParse (API-based, best quality)
Web pages?                   → WebBaseLoader / BeautifulSoup
CSV / JSON?                  → CSVLoader / JSONLoader
```

### Choosing a Splitter
```
General text?                → RecursiveCharacterTextSplitter ⭐
Markdown documentation?      → MarkdownHeaderTextSplitter
HTML content?                → HTMLHeaderTextSplitter
Source code?                 → CodeTextSplitter (language-aware)
Need semantic coherence?     → SemanticChunker
Maximum quality?             → Agentic Chunking (LLM-powered)
```

### Choosing an Embedding Model
```
Free, local development?     → all-MiniLM-L6-v2 (384d)
Production, general purpose? → text-embedding-3-small (1536d)
Maximum quality?             → text-embedding-3-large (3072d)
Google ecosystem?            → text-embedding-004 (768d)
Code & technical docs?       → voyage-3 (1024d)
```

### Choosing a Summarization / Processing Strategy
```
All chunks fit context comfortably?  → Stuff (fastest, lowest cost, 1 call) ⭐
Massive doc, parallel speed needed?  → Map-Reduce (scales infinitely, N+1 calls) ⭐
Chronological / narrative buildup?   → Refine (sequential rolling update, N calls)
Pinpoint single best factual chunk?  → Map-Rerank (scores each candidate answer)
High throughput / rate limit budget? → Batch Processing (groups chunks into mini-batches)
```

---

> **Final Tip for L2 Exam**: The data processing pipeline is "**Garbage in, Garbage out**" — parsing quality → chunk quality → embedding quality → retrieval quality → answer quality. Always start by verifying your parser output and optimizing your chunking strategy BEFORE tweaking the model or prompt.

---

*Study Material Created for GenAI L2 Exam Preparation*  
*Path: `RAG_END2END/Chunking_embedding/`*
