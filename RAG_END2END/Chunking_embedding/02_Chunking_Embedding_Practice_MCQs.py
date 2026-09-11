1
# 🧠 Chunking, Embeddings & Document Processing — Practice MCQs (GenAI L2 Exam)
# ==============================================================================
# 75 Multiple Choice Questions in 3 Difficulty Levels:
#   🟢 EASY (25 questions)    — Definitions, basic parsers, chunkers, embeddings
#   🟡 MEDIUM (25 questions)  — Strategies (Stuff, Map-Reduce, Refine), MRL, HNSW
#   🔴 HARD (25 questions)    — Production traps, failure modes, tuning & algorithms
#
# Run this script to take an interactive self-assessment quiz:
#   python 02_Chunking_Embedding_Practice_MCQs.py

import random

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🟢 EASY QUESTIONS (Definitions, Basic Loaders & Fundamentals)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EASY_QUESTIONS = [
    {
        "q": "What is the primary role of 'Document Parsing' in the RAG pipeline?",
        "options": [
            "A) Translating text into multiple languages",
            "B) Extracting raw text and structural metadata from files (PDF, DOCX, HTML, etc.)",
            "C) Generating vector embeddings directly from binary files",
            "D) Compressing text files into zip archives",
        ],
        "answer": "B",
        "explanation": "Document parsing is the initial extraction phase that reads raw file formats and extracts text along with structural metadata.",
    },
    {
        "q": "What are the two mandatory attributes of a LangChain Document object?",
        "options": [
            "A) `text` and `id`",
            "B) `page_content` and `metadata`",
            "C) `raw_data` and `file_path`",
            "D) `body` and `headers`",
        ],
        "answer": "B",
        "explanation": "Every LangChain Document has `page_content` (string containing the text) and `metadata` (dictionary for source, page numbers, etc.).",
    },
    {
        "q": "Which text splitter is the recommended DEFAULT in LangChain for general text?",
        "options": [
            "A) CharacterTextSplitter",
            "B) TokenTextSplitter",
            "C) RecursiveCharacterTextSplitter",
            "D) NLTKTextSplitter",
        ],
        "answer": "C",
        "explanation": "RecursiveCharacterTextSplitter is the default because it tries paragraph, line, and word boundaries recursively to maintain semantic coherence.",
    },
    {
        "q": "What is the purpose of `chunk_overlap` in text chunking?",
        "options": [
            "A) To make chunks as large as possible",
            "B) To prevent critical information from being split and lost at chunk boundaries",
            "C) To increase the speed of the embedding model",
            "D) To encrypt chunks before storing them",
        ],
        "answer": "B",
        "explanation": "Chunk overlap ensures that sentences or context split across boundaries appear in adjacent chunks, preventing context loss.",
    },
    {
        "q": "What is an 'Embedding' in Generative AI?",
        "options": [
            "A) A hardware chip designed to run LLMs locally",
            "B) A dense numerical vector of floating-point numbers representing semantic meaning",
            "C) A software plugin inserted into a web browser",
            "D) A sparse one-hot encoded binary matrix",
        ],
        "answer": "B",
        "explanation": "An embedding is a dense vector of floats (e.g., 384, 1536 dimensions) where semantically similar texts are placed close together.",
    },
    {
        "q": "What is the standard distance metric used for comparing text embeddings?",
        "options": [
            "A) Hamming Distance",
            "B) Cosine Similarity",
            "C) Jaccard Index",
            "D) Levenshtein Distance",
        ],
        "answer": "B",
        "explanation": "Cosine similarity is the industry standard for text embeddings because it measures the angle between vectors, making it robust to length.",
    },
    {
        "q": "What is the range of values for Cosine Similarity?",
        "options": [
            "A) [0, ∞)",
            "B) [-1, 1]",
            "C) [0, 100]",
            "D) (-∞, ∞)",
        ],
        "answer": "B",
        "explanation": "Cosine similarity ranges from -1 (exact opposite directions) to 1 (identical directions). In text embeddings, most scores lie between 0 and 1.",
    },
    {
        "q": "What is the 'Golden Rule' of embeddings in RAG?",
        "options": [
            "A) Always use at least 4096 dimensions",
            "B) You MUST use the exact same embedding model for indexing and querying",
            "C) Embeddings must be recalculated on every user prompt",
            "D) Embeddings can only be generated from English text",
        ],
        "answer": "B",
        "explanation": "Different embedding models map concepts into incompatible vector spaces. Using different models for indexing and querying yields random results.",
    },
    {
        "q": "In LangChain, what is the difference between `embed_query()` and `embed_documents()`?",
        "options": [
            "A) `embed_query` is for single text; `embed_documents` is for a list/batch of texts",
            "B) `embed_query` is free; `embed_documents` requires a paid API key",
            "C) `embed_query` outputs sparse vectors; `embed_documents` outputs dense vectors",
            "D) `embed_query` works only on SQL; `embed_documents` works on PDFs",
        ],
        "answer": "A",
        "explanation": "`embed_query(text)` takes a single string query, while `embed_documents(texts)` takes a list of strings for batch indexing.",
    },
    {
        "q": "What is the 'Stuff' (Stuffing) document summarization strategy?",
        "options": [
            "A) Iteratively refining a summary across sequential chunks",
            "B) Summarizing each chunk individually and then combining them",
            "C) Concatenating all document chunks into a single prompt for 1 LLM call",
            "D) Storing chunks into a relational database table",
        ],
        "answer": "C",
        "explanation": "The 'Stuff' strategy puts ('stuffs') all retrieved text or document pieces into a single prompt sent to the LLM in one single API call.",
    },
    {
        "q": "How many LLM calls does the 'Stuff' strategy make?",
        "options": [
            "A) One call per chunk",
            "B) Exactly 1 LLM call",
            "C) 2 calls per chunk",
            "D) Logarithmic number of calls",
        ],
        "answer": "B",
        "explanation": "The Stuff strategy concatenates all context into one prompt, making exactly 1 LLM call.",
    },
    {
        "q": "What are the two phases of the 'Map-Reduce' document processing strategy?",
        "options": [
            "A) Parsing phase and Splitting phase",
            "B) Map phase (summarize individual chunks in parallel) and Reduce phase (combine sub-summaries)",
            "C) Indexing phase and Search phase",
            "D) Prompting phase and Evaluation phase",
        ],
        "answer": "B",
        "explanation": "Map-Reduce first maps an LLM call over each chunk independently to produce sub-summaries, then reduces (combines) them into a final synthesis.",
    },
    {
        "q": "How does the 'Refine' summarization strategy work?",
        "options": [
            "A) It evaluates chunks in parallel and discards non-matching ones",
            "B) It summarizes the first chunk, then passes that summary + the next chunk sequentially to iteratively update the summary",
            "C) It replaces words with synonyms to improve readability",
            "D) It compresses vectors to smaller dimensions",
        ],
        "answer": "B",
        "explanation": "Refine is sequential: each step takes the previous summary (`existing_answer`) and the current chunk to generate an updated summary.",
    },
    {
        "q": "Which PDF loader in LangChain is typically recommended when tables need to be extracted from digital PDFs?",
        "options": [
            "A) PyPDFLoader",
            "B) pdfplumber",
            "C) CSVLoader",
            "D) BSHTMLLoader",
        ],
        "answer": "B",
        "explanation": "`pdfplumber` specializes in inspecting visual layout and extracting tables cleanly from digital PDFs.",
    },
    {
        "q": "What does OCR stand for in document parsing?",
        "options": [
            "A) Optical Character Recognition",
            "B) Optimal Chunk Retention",
            "C) Open Context Retrieval",
            "D) Orchestrated Chain Routing",
        ],
        "answer": "A",
        "explanation": "OCR = Optical Character Recognition. It converts images of text (such as scanned documents or photos) into machine-readable characters.",
    },
    {
        "q": "What is the recommended rule of thumb for `chunk_overlap` relative to `chunk_size`?",
        "options": [
            "A) 0% (no overlap)",
            "B) 10% to 20% of `chunk_size`",
            "C) 75% to 90% of `chunk_size`",
            "D) 100% of `chunk_size`",
        ],
        "answer": "B",
        "explanation": "A chunk overlap of 10-20% (e.g., 100-200 characters for a 1000-character chunk) provides sufficient continuity without excessive redundancy.",
    },
    {
        "q": "When splitting LangChain Document objects, which method should you call to preserve metadata?",
        "options": [
            "A) `split_text()`",
            "B) `split_documents()`",
            "C) `split_raw()`",
            "D) `parse_and_split()`",
        ],
        "answer": "B",
        "explanation": "`split_documents()` accepts a list of Document objects and propagates the metadata to every resulting chunk. `split_text()` only works on raw strings.",
    },
    {
        "q": "What vector indexing algorithm is the default in ChromaDB, Qdrant, and pgvector?",
        "options": [
            "A) B-Tree Index",
            "B) HNSW (Hierarchical Navigable Small World)",
            "C) Bitmap Index",
            "D) Hash Map",
        ],
        "answer": "B",
        "explanation": "HNSW is the production standard Approximate Nearest Neighbor (ANN) indexing algorithm for vector databases.",
    },
    {
        "q": "What does ANN stand for in vector search?",
        "options": [
            "A) Artificial Neural Network",
            "B) Approximate Nearest Neighbor",
            "C) Automated Node Navigation",
            "D) Asynchronous Network Node",
        ],
        "answer": "B",
        "explanation": "ANN = Approximate Nearest Neighbor — an algorithm family that finds vectors near a query extremely fast by trading off 100% exactness for speed.",
    },
    {
        "q": "Which open-source, local embedding model produces 384-dimensional vectors?",
        "options": [
            "A) OpenAI text-embedding-3-large",
            "B) all-MiniLM-L6-v2",
            "C) Google text-embedding-004",
            "D) Voyage-3",
        ],
        "answer": "B",
        "explanation": "`all-MiniLM-L6-v2` from sentence-transformers is a popular lightweight local model that outputs 384-dimensional vectors.",
    },
    {
        "q": "What does MRL stand for in modern embedding models?",
        "options": [
            "A) Multi-Retrieval Language",
            "B) Matryoshka Representation Learning",
            "C) Maximum Recurrent Layer",
            "D) Model Routing Logic",
        ],
        "answer": "B",
        "explanation": "MRL = Matryoshka Representation Learning, allowing embeddings to be truncated to smaller dimensions while preserving semantic accuracy.",
    },
    {
        "q": "What happens if you use a 'CharacterTextSplitter' with `separator='\\n\\n'` on text that has no double newlines?",
        "options": [
            "A) It automatically falls back to splitting by sentences",
            "B) It returns the entire text as a single chunk, ignoring `chunk_size`",
            "C) It raises an unhandled IndexError exception",
            "D) It truncates the text at 100 characters",
        ],
        "answer": "B",
        "explanation": "CharacterTextSplitter does not fall back recursively. If the separator does not exist, the whole string remains one oversized chunk.",
    },
    {
        "q": "Which distance metric measures the straight-line geometric distance between two vectors?",
        "options": [
            "A) Cosine Similarity",
            "B) Euclidean Distance (L2)",
            "C) Dot Product",
            "D) Jaccard Distance",
        ],
        "answer": "B",
        "explanation": "Euclidean distance (L2 norm) computes the straight-line distance: sqrt(sum((a_i - b_i)^2)).",
    },
    {
        "q": "What is the primary danger of setting `chunk_size` too small (e.g., 50 characters)?",
        "options": [
            "A) Exceeding LLM token limits",
            "B) Loss of surrounding context, making chunks incomprehensible to the LLM",
            "C) Vector databases refusing to accept vectors",
            "D) Embedding generation running out of RAM",
        ],
        "answer": "B",
        "explanation": "If chunks are too small, sentences get cut into fragments. The retrieved chunk lacks enough context for the LLM to form an answer.",
    },
    {
        "q": "Which loader would you use to scrape and parse content from a web URL in LangChain?",
        "options": [
            "A) WebBaseLoader",
            "B) CSVLoader",
            "C) JSONLoader",
            "D) DirectoryLoader",
        ],
        "answer": "A",
        "explanation": "`WebBaseLoader` uses urllib and BeautifulSoup to fetch HTML and extract readable text from web pages.",
    },
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🟡 MEDIUM QUESTIONS (Architecture, Strategies, Tuning & MRL)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MEDIUM_QUESTIONS = [
    {
        "q": "In RecursiveCharacterTextSplitter, what is the default hierarchy of separators?",
        "options": [
            "A) `[' ', '.', '\\n', '\\n\\n']`",
            "B) `['\\n\\n', '\\n', ' ', '']`",
            "C) `['\\t', '\\n', ' ', '']`",
            "D) `['#', '##', '###', '']`",
        ],
        "answer": "B",
        "explanation": "The default separators are `['\\n\\n', '\\n', ' ', '']` — prioritizing splitting by paragraphs, then lines, then words, and finally characters.",
    },
    {
        "q": "What is the 'Lost-in-the-Middle' phenomenon in LLM document processing?",
        "options": [
            "A) When an embedding model loses the center dimensions of a vector",
            "B) LLMs recall facts placed at the beginning and end of long prompts much better than facts in the middle",
            "C) Chunks that are stored in the middle of a vector database become corrupt",
            "D) When Map-Reduce loses the intermediate summary file",
        ],
        "answer": "B",
        "explanation": "Research (Liu et al., Stanford) showed LLMs suffer from primacy and recency bias, frequently missing information in the middle of large contexts.",
    },
    {
        "q": "Why can the 'Refine' summarization strategy NOT be parallelized?",
        "options": [
            "A) Because LangChain limits Python to a single thread",
            "B) Each step strictly requires the previous step's output (`existing_answer`) as input",
            "C) Vector databases do not support concurrent reads",
            "D) The LLM API blocks parallel connections from the same IP",
        ],
        "answer": "B",
        "explanation": "Refine is inherently sequential: Step k cannot execute until Step k-1 completes and produces `existing_answer`.",
    },
    {
        "q": "What is the 'collapse' step in LangChain's MapReduceDocumentsChain?",
        "options": [
            "A) Deleting corrupt chunks automatically",
            "B) When intermediate summaries exceed the context limit, recursively grouping and summarizing them until they fit the reduce prompt",
            "C) Compressing text into gzip format",
            "D) Converting dense embeddings into sparse vectors",
        ],
        "answer": "B",
        "explanation": "If the intermediate summaries from the Map step are still too large to fit in the combine prompt, the collapse phase recursively groups and summarizes them.",
    },
    {
        "q": "How does the 'Map-Rerank' document chain determine the final answer?",
        "options": [
            "A) It averages the vector distances of all chunks",
            "B) It asks the LLM to answer AND output a confidence score for each chunk, then returns the highest-scoring answer",
            "C) It selects the chunk with the most words",
            "D) It prompts the LLM to rewrite all chunks into a single paragraph",
        ],
        "answer": "B",
        "explanation": "In Map-Rerank, the LLM generates an answer and a certainty/confidence score for each chunk. The answer with the highest score is selected.",
    },
    {
        "q": "When vectors are unit-normalized (length = 1.0), what is the mathematical relationship between Cosine Similarity and Euclidean Distance (L2)?",
        "options": [
            "A) They produce completely uncorrelated, random rankings",
            "B) Euclidean distance squared equals 2 * (1 - Cosine Similarity), so they produce identical rankings",
            "C) Cosine similarity is always twice the Euclidean distance",
            "D) Euclidean distance cannot be calculated on normalized vectors",
        ],
        "answer": "B",
        "explanation": "For unit vectors: ||A - B||^2 = 2 - 2*(A · B) = 2*(1 - cos(theta)). Thus, sorting by highest cosine similarity is identical to sorting by lowest Euclidean distance.",
    },
    {
        "q": "What is the primary advantage of Matryoshka Representation Learning (MRL) in OpenAI's `text-embedding-3` models?",
        "options": [
            "A) It converts text directly to audio vectors",
            "B) It allows truncating embeddings to fewer dimensions (e.g. 1536 → 512) with minimal loss of retrieval accuracy, saving storage and compute",
            "C) It encrypts vectors for zero-knowledge search",
            "D) It increases the number of dimensions dynamically during query time",
        ],
        "answer": "B",
        "explanation": "MRL trains models so earlier dimensions contain the most critical information. You can truncate vectors to save storage and search time with negligible accuracy loss.",
    },
    {
        "q": "How does a `SemanticChunker` determine where to split text?",
        "options": [
            "A) By counting the number of syllables per sentence",
            "B) By calculating the embedding similarity between consecutive sentences and splitting when similarity drops significantly",
            "C) By checking grammar using Python regex",
            "D) By splitting exactly every 500 characters",
        ],
        "answer": "B",
        "explanation": "`SemanticChunker` splits text into sentences, embeds them, and calculates cosine distance between adjacent sentences. When distance exceeds a threshold, a new chunk starts.",
    },
    {
        "q": "In `SemanticChunker`, what does `breakpoint_threshold_type='percentile'` do?",
        "options": [
            "A) Splits the document into exactly 100 equal chunks",
            "B) Sets the split threshold at a specified percentile (e.g., 95th percentile) of all sentence-to-sentence distance differences",
            "C) Drops 10% of the lowest-scoring words",
            "D) Replaces 90% of words with synonyms",
        ],
        "answer": "B",
        "explanation": "Percentile thresholding computes distance differences between all adjacent sentences and splits at the points whose distance exceeds a specified percentile.",
    },
    {
        "q": "Why is `MarkdownHeaderTextSplitter` superior to character splitters for Markdown documentation?",
        "options": [
            "A) It removes all markdown syntax from the text",
            "B) It splits content by header levels (#, ##, ###) and attaches the header hierarchy to each chunk's `metadata`",
            "C) It converts markdown into binary bytecode",
            "D) It automatically publishes markdown files to GitHub",
        ],
        "answer": "B",
        "explanation": "`MarkdownHeaderTextSplitter` preserves the contextual hierarchy (e.g., Header 1 > Header 2) in the metadata of every resulting chunk.",
    },
    {
        "q": "What is the total number of LLM calls in a Map-Reduce chain processing N chunks (assuming no collapse is needed)?",
        "options": [
            "A) 1 call",
            "B) N calls",
            "C) N + 1 calls (N map calls + 1 reduce call)",
            "D) 2 * N calls",
        ],
        "answer": "C",
        "explanation": "Map-Reduce makes 1 map call for each of the N chunks, plus 1 final reduce call to synthesize the results = N + 1 calls.",
    },
    {
        "q": "What is the primary risk of using the 'Refine' summarization chain over a long document?",
        "options": [
            "A) Chunks are processed out of order",
            "B) Error and hallucination propagation, plus recency bias where the final chunk overly influences the summary",
            "C) Vector DB connection timeout",
            "D) Embedding dimension overflow",
        ],
        "answer": "B",
        "explanation": "Because Refine passes the accumulated summary forward, any early hallucination or error persists and may be amplified, and the last chunks may dominate.",
    },
    {
        "q": "In HNSW, how does the hierarchical multi-layer graph accelerate vector search?",
        "options": [
            "A) Top layers have sparse nodes with long-range links for fast coarse navigation; bottom layers have dense nodes for fine search",
            "B) It sorts vectors alphabetically by first letter",
            "C) It converts vectors into SQL relational tables",
            "D) It eliminates the need for vector distance calculations",
        ],
        "answer": "A",
        "explanation": "Inspired by Skip Lists, HNSW has sparse top layers with long links to traverse space rapidly, then descends to denser bottom layers for precise nearest neighbors.",
    },
    {
        "q": "When processing large tables in PDF reports for RAG, what is considered the best practice?",
        "options": [
            "A) Strip all numbers and keep only row headers",
            "B) Extract tables into structured Markdown or HTML format and generate natural-language table summaries for embedding",
            "C) Replace table data with random characters",
            "D) Treat tables as a single giant word",
        ],
        "answer": "B",
        "explanation": "Converting tables to Markdown/HTML preserves column-row relationships, and generating a text summary ensures the table is retrievable via semantic search.",
    },
    {
        "q": "How does `TokenTextSplitter` differ from `CharacterTextSplitter`?",
        "options": [
            "A) `TokenTextSplitter` splits based on BPE token counts (matching the LLM's tokenizer) rather than raw character counts",
            "B) `TokenTextSplitter` only works on JSON Web Tokens",
            "C) `TokenTextSplitter` is much slower and deprecated",
            "D) `TokenTextSplitter` does not support overlap",
        ],
        "answer": "A",
        "explanation": "LLM context windows are measured in tokens, not characters. `TokenTextSplitter` ensures chunks strictly respect model token budgets.",
    },
    {
        "q": "In a RAG pipeline, why might a query return irrelevant chunks even if cosine similarity calculation is working correctly?",
        "options": [
            "A) Because cosine similarity cannot exceed 0.5",
            "B) Semantic dilution caused by chunks being too large and containing multiple unrelated topics",
            "C) The vector database has too many CPU cores",
            "D) The user typed in lowercase letters",
        ],
        "answer": "B",
        "explanation": "If chunks are too large, distinct topics get blended into a single average vector (semantic dilution), pulling the embedding away from the specific topic.",
    },
    {
        "q": "What is the difference between Inverted File Index (IVF) and Flat vector indexing?",
        "options": [
            "A) Flat calculates exact distance to every vector (brute force); IVF clusters vectors into Voronoi cells and only searches nearby centroids",
            "B) IVF is 100% exact; Flat is approximate",
            "C) Flat requires GPU; IVF runs only in browser memory",
            "D) There is no difference",
        ],
        "answer": "A",
        "explanation": "Flat search computes distance against all vectors (100% recall, O(N)). IVF clusters vectors with k-means and only searches candidate clusters (ANN, O(clusters)).",
    },
    {
        "q": "When using `load_summarize_chain` with `chain_type='map_reduce'`, which parameter allows inspecting sub-summaries?",
        "options": [
            "A) `debug_mode=True`",
            "B) `return_intermediate_steps=True`",
            "C) `show_chunks=True`",
            "D) `verbose_output=True`",
        ],
        "answer": "B",
        "explanation": "Setting `return_intermediate_steps=True` includes the individual mapped chunk summaries in the output dictionary.",
    },
    {
        "q": "Why is batching chunk processing (e.g. `llm.batch()` or `llm.abatch()`) important in large-scale document pipelines?",
        "options": [
            "A) It turns off API rate limits completely",
            "B) It reduces HTTP connection overhead and leverages concurrency to maximize throughput under rate limits",
            "C) It converts Python synchronous functions to C++ binaries",
            "D) It prevents embedding vectors from drifting",
        ],
        "answer": "B",
        "explanation": "Batching and async execution allow multiple chunks to be processed concurrently without the serial overhead of individual sequential HTTP requests.",
    },
    {
        "q": "What is the effect of setting `chunk_overlap` greater than `chunk_size`?",
        "options": [
            "A) It creates faster retrieval speeds",
            "B) It causes an error or infinite loop, as overlap cannot be greater than or equal to chunk size",
            "C) It improves vector normalization",
            "D) It automatically enables Map-Reduce",
        ],
        "answer": "B",
        "explanation": "Chunk overlap must always be strictly less than chunk_size; otherwise, the sliding window cannot advance, causing an error or infinite loop.",
    },
    {
        "q": "Which loader is recommended for complex scanned PDFs requiring layout analysis, OCR, and table detection?",
        "options": [
            "A) UnstructuredPDFLoader",
            "B) TextLoader",
            "C) CSVLoader",
            "D) JSONLoader",
        ],
        "answer": "A",
        "explanation": "`UnstructuredPDFLoader` (or tools like Docling/LlamaParse) performs deep layout analysis, OCR, and bounding box detection for complex documents.",
    },
    {
        "q": "In modern LangChain, what is the LCEL replacement for `StuffDocumentsChain`?",
        "options": [
            "A) `create_stuff_documents_chain(llm, prompt)`",
            "B) `llm.stuff_documents()`",
            "C) `DocumentPipe()`",
            "D) `StuffExpression()`",
        ],
        "answer": "A",
        "explanation": "In LangChain 0.2+, `create_stuff_documents_chain` from `langchain.chains.combine_documents` is the standard LCEL constructor.",
    },
    {
        "q": "What happens if a text file contains UTF-8 characters but is loaded with default Windows encoding (cp1252) in Python?",
        "options": [
            "A) The file is automatically translated",
            "B) A `UnicodeDecodeError` is raised or text becomes garbled ('mojibake')",
            "C) The vector database crashes permanently",
            "D) All numbers are stripped",
        ],
        "answer": "B",
        "explanation": "On Windows, Python defaults to `cp1252`. Reading UTF-8 files without specifying `encoding='utf-8'` causes `UnicodeDecodeError` or mojibake characters.",
    },
    {
        "q": "What is the primary trade-off of using 'Agentic Chunking' over rule-based chunking?",
        "options": [
            "A) Agentic chunking produces low-quality chunks but is very fast",
            "B) Agentic chunking produces the highest quality semantic boundaries, but is very slow and expensive due to LLM calls",
            "C) Agentic chunking only works with markdown files",
            "D) Agentic chunking cannot use embeddings",
        ],
        "answer": "B",
        "explanation": "Agentic chunking has an LLM read the text to decide natural chunk boundaries. It yields premium chunks but incurs significant LLM latency and cost.",
    },
    {
        "q": "Why is Cosine Similarity preferred over Dot Product when vector lengths are NOT normalized?",
        "options": [
            "A) Dot product only works on 2D vectors",
            "B) Dot product is biased toward longer texts with larger vector magnitudes, whereas cosine similarity normalizes by vector length",
            "C) Cosine similarity cannot be computed on GPUs",
            "D) Dot product always returns negative numbers",
        ],
        "answer": "B",
        "explanation": "Dot product equals |A| * |B| * cos(theta). Longer texts with higher token frequencies have larger norms, which artificially inflates dot product.",
    },
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🔴 HARD QUESTIONS (Edge Cases, Production Traps, Math & Code)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HARD_QUESTIONS = [
    {
        "q": "You are building a real-time customer support RAG system with a strict 2-second SLA. Which document processing strategy should you choose for synthesizing top-3 retrieved chunks (total 1,500 tokens)?",
        "options": [
            "A) Refine (because it builds cumulative context)",
            "B) Stuff (create_stuff_documents_chain) (because it requires only 1 LLM call and minimizes latency)",
            "C) Map-Reduce with hierarchical collapse",
            "D) Agentic chunking on the fly",
        ],
        "answer": "B",
        "explanation": "With 1,500 tokens comfortably under context limits and a 2-second SLA, Stuff is optimal because it executes in a single LLM request without multi-step orchestration.",
    },
    {
        "q": "A legal team needs to summarize a 300-page chronological deposition. Why is 'Map-Reduce' a poor choice compared to 'Refine' for this specific task?",
        "options": [
            "A) Map-Reduce cannot run on legal texts",
            "B) Map-Reduce processes chunks independently during the Map phase, breaking chronological narrative and cross-page testimony progression",
            "C) Refine uses fewer total LLM calls than Map-Reduce",
            "D) Map-Reduce requires all chunks to be written in markdown",
        ],
        "answer": "B",
        "explanation": "The Map phase evaluates chunks in isolation. If an alibi on page 50 contradicts testimony on page 10, Map-Reduce will miss the timeline synthesis that Refine preserves.",
    },
    {
        "q": "In an HNSW vector index, what do the parameters `M` and `efConstruction` control?",
        "options": [
            "A) `M` = number of dimensions; `efConstruction` = number of documents",
            "B) `M` = max bidirectional links per node; `efConstruction` = size of the dynamic candidate list evaluated during index build time",
            "C) `M` = memory limit in GB; `efConstruction` = number of CPU cores",
            "D) `M` = margin of error; `efConstruction` = embedding batch size",
        ],
        "answer": "B",
        "explanation": "In HNSW, `M` defines the number of bidirectional connections per element, and `efConstruction` governs the search depth during index construction (higher = better recall, slower build).",
    },
    {
        "q": "In LangChain's `MapReduceDocumentsChain`, what triggers the `collapse_documents_chain` to execute?",
        "options": [
            "A) When an LLM call throws a rate-limit 429 error",
            "B) When the combined token count of intermediate mapped summaries exceeds `token_max` of the final combine prompt",
            "C) When vector distance drops below zero",
            "D) When chunk overlap is set to 0",
        ],
        "answer": "B",
        "explanation": "If the intermediate summaries from the map phase are too voluminous to fit into the combine prompt, the collapse chain groups them into batches and shrinks them recursively.",
    },
    {
        "q": "Suppose you truncate an OpenAI `text-embedding-3-large` embedding from 3072 dimensions to 512 dimensions using MRL. What mathematical step MUST you perform before calculating dot product similarity?",
        "options": [
            "A) Add random noise to prevent zeroes",
            "B) Re-normalize the truncated 512-dimensional vector to unit length (L2 norm = 1.0)",
            "C) Transpose the vector into a row matrix",
            "D) Invert all negative numbers",
        ],
        "answer": "B",
        "explanation": "Truncating an already-normalized 3072-d vector changes its L2 norm to < 1.0. You must re-normalize the truncated vector to unit length so dot product equals cosine similarity.",
    },
    {
        "q": "What is the purpose of Maximum Marginal Relevance (MMR) retrieval, and what does the `lambda_mult` parameter balance?",
        "options": [
            "A) It balances CPU usage vs GPU usage",
            "B) It balances relevance to the query (lambda_mult -> 1.0) with diversity among selected chunks (lambda_mult -> 0.0)",
            "C) It balances chunk_size vs chunk_overlap",
            "D) It balances token count vs character count",
        ],
        "answer": "B",
        "explanation": "MMR avoids returning redundant chunks. `lambda_mult` weights query relevance against diversity (dissimilarity to already-selected documents).",
    },
    {
        "q": "A production RAG system indexes chunks using `text-embedding-3-small` (1536 dims). Due to a configuration error, the query endpoint uses `all-MiniLM-L6-v2` (384 dims). What occurs when ChromaDB performs the vector search?",
        "options": [
            "A) ChromaDB automatically pads the query vector with zeroes to 1536 dims",
            "B) A dimensionality mismatch exception is raised immediately because 384 != 1536",
            "C) The query succeeds but returns only the top 1 result",
            "D) The index is automatically wiped and rebuilt",
        ],
        "answer": "B",
        "explanation": "Vector databases enforce strict vector dimensionality per collection. A vector of 384 dims cannot be compared against a 1536-d index and raises a dimension mismatch error.",
    },
    {
        "q": "Why is Product Quantization (PQ) widely utilized when scaling vector indexes to 100+ million vectors?",
        "options": [
            "A) It decomposes high-dimensional vectors into low-dimensional sub-vectors and quantizes them into centroid codes, slashing RAM usage by up to 95%",
            "B) It converts all vectors to ASCII text files",
            "C) It completely eliminates the need for an embedding model",
            "D) It guarantees 100% exact nearest neighbor recall",
        ],
        "answer": "A",
        "explanation": "PQ splits high-dimensional vectors into subspaces and assigns them codebook indices, compressing 32-bit floats into byte codes to allow massive indexes to fit in RAM.",
    },
    {
        "q": "You are parsing a 50-page financial report where key balance sheets span multiple pages. What is the biggest danger of using naive `RecursiveCharacterTextSplitter(chunk_size=1000)`?",
        "options": [
            "A) Python will run out of memory",
            "B) Tables will be sliced mid-row or mid-column, destroying header-cell associations and causing hallucinated financial metrics",
            "C) LangChain will refuse to split strings containing digits",
            "D) The PDF loader will delete the file",
        ],
        "answer": "B",
        "explanation": "Naive character splitters don't understand tabular geometry. Slicing through tables separates headers from numbers, destroying the relational meaning.",
    },
    {
        "q": "In LangChain's `RefineDocumentsChain`, what are the two required input variables in the `refine_prompt`?",
        "options": [
            "A) `{input}` and `{output}`",
            "B) `{existing_answer}` and `{text}` (or the document content variable)",
            "C) `{query}` and `{chunks}`",
            "D) `{history}` and `{summary}`",
        ],
        "answer": "B",
        "explanation": "The refine prompt needs `{existing_answer}` (the summary generated so far) and `{text}` (the context from the current chunk being evaluated).",
    },
    {
        "q": "Consider a 10-chunk document processed with the 'Refine' chain. The single LLM call latency is 1.5 seconds. What is the theoretical minimum wall-clock time for the chain?",
        "options": [
            "A) 1.5 seconds (all 10 run in parallel)",
            "B) 15.0 seconds (10 sequential calls * 1.5s)",
            "C) 3.0 seconds",
            "D) 0.15 seconds",
        ],
        "answer": "B",
        "explanation": "Because Refine is strictly sequential, the wall-clock time is the sum of all individual calls: 10 * 1.5s = 15.0 seconds.",
    },
    {
        "q": "Consider the same 10-chunk document processed with 'Map-Reduce'. Single LLM call latency is 1.5s. If Map calls run concurrently and combine takes 1.5s, what is the theoretical minimum wall-clock time?",
        "options": [
            "A) 15.0 seconds",
            "B) ~3.0 seconds (1.5s parallel Map + 1.5s Reduce)",
            "C) 0.5 seconds",
            "D) 30.0 seconds",
        ],
        "answer": "B",
        "explanation": "With full concurrency, all 10 Map calls execute simultaneously (~1.5s total), followed by 1 Reduce call (~1.5s) = ~3.0 seconds total.",
    },
    {
        "q": "Why does PyMuPDF (`fitz`) generally outperform `pypdf` in enterprise production pipelines?",
        "options": [
            "A) PyMuPDF is written in pure Python",
            "B) PyMuPDF is a high-performance C-library wrapper (MuPDF) that is 10-20x faster and offers superior text positioning and bounding box extraction",
            "C) PyMuPDF has built-in fine-tuning for LLMs",
            "D) PyMuPDF does not require memory",
        ],
        "answer": "B",
        "explanation": "PyMuPDF wraps the MuPDF C library, providing order-of-magnitude faster parsing and precise spatial coordinates for words, lines, and blocks.",
    },
    {
        "q": "In a 2-stage retrieval pipeline using Matryoshka embeddings, how does asymmetric dimension search work?",
        "options": [
            "A) You search with small dimensions (e.g. 256d) to quickly filter top-100 candidates, then re-rank only those 100 using full 3072d vectors",
            "B) You use text embeddings for documents and audio embeddings for queries",
            "C) Half of the documents are embedded with OpenAI and half with HuggingFace",
            "D) Vectors are sorted by file size",
        ],
        "answer": "A",
        "explanation": "2-stage MRL retrieval: fast, cheap search over the entire corpus using 256d vectors, then re-rank top-k candidates using the full 3072d vectors for maximum precision.",
    },
    {
        "q": "What is the primary vulnerability of the 'Map-Rerank' strategy when answering complex, multi-part synthesis questions?",
        "options": [
            "A) It makes too few LLM calls",
            "B) It assumes the complete answer exists in a single chunk; it cannot synthesize answers split across multiple chunks",
            "C) It only accepts integer scores",
            "D) It only supports ChromaDB",
        ],
        "answer": "B",
        "explanation": "Map-Rerank picks the single highest-scoring chunk answer. If an answer requires combining Fact A from Chunk 1 and Fact B from Chunk 8, Map-Rerank fails.",
    },
    {
        "q": "Why does `HTMLHeaderTextSplitter` attach structural header metadata to chunks instead of relying solely on raw HTML tags?",
        "options": [
            "A) Raw HTML tags consume excessive token context, while header metadata cleanly preserves semantic hierarchy without wasting prompt tokens",
            "B) LLMs are incapable of reading HTML tags",
            "C) HTML tags cannot be stored in ChromaDB",
            "D) HTML tags prevent cosine similarity from working",
        ],
        "answer": "A",
        "explanation": "Passing raw `<div>`, `<span>`, and `<table>` boilerplate wastes precious context window tokens. Converting tags to clean metadata preserves context efficiently.",
    },
    {
        "q": "Which text splitting strategy is optimal when preparing Python, JavaScript, or C++ source code files for a code-search RAG pipeline?",
        "options": [
            "A) CharacterTextSplitter with separator=' '",
            "B) `RecursiveCharacterTextSplitter.from_language()` using Language-specific AST/grammar separators (e.g. `def `, `class `)",
            "C) TokenTextSplitter with chunk_overlap=0",
            "D) SemanticChunker with percentile=50",
        ],
        "answer": "B",
        "explanation": "`from_language()` configures separators tailored to programming languages (class definitions, functions, scopes) so functions and classes aren't split arbitrarily.",
    },
    {
        "q": "Suppose you run a RAG pipeline and observe that queries containing acronyms ('EBITDA', 'ARR') fail to retrieve the right chunks. What is the most effective solution?",
        "options": [
            "A) Switch to pure Map-Reduce",
            "B) Implement Hybrid Search (BM25 sparse keyword search + Dense vector search with Reciprocal Rank Fusion)",
            "C) Truncate embedding dimensions to 64",
            "D) Increase chunk_overlap to 90%",
        ],
        "answer": "B",
        "explanation": "Dense embeddings sometimes struggle with exact acronyms or specific IDs. Hybrid search pairs dense embeddings with BM25 keyword matching for exact keyword precision.",
    },
    {
        "q": "In LangChain, what happens if an unhandled document causes an OCR error during batch indexing in a production pipeline?",
        "options": [
            "A) The entire pipeline crashes unless robust exception handling or a fault-tolerant loader pattern (e.g., try/except per document) is implemented",
            "B) Python automatically retries the OCR on GPU",
            "C) The document is automatically replaced by an empty string",
            "D) The vector database generates a placeholder vector",
        ],
        "answer": "A",
        "explanation": "A failure on a corrupted or encrypted PDF will crash batch execution unless documents are loaded in a try/except loop or using error-handling loaders.",
    },
    {
        "q": "What is the primary benefit of generating a summary for each chunk during the indexing stage and embedding the summary alongside the raw chunk?",
        "options": [
            "A) It doubles the memory of the vector database",
            "B) Summaries align closer to user search queries in embedding space, improving retrieval while preserving raw text for the LLM prompt",
            "C) It allows deleting the raw chunk entirely",
            "D) It converts the vector database to a graph database",
        ],
        "answer": "B",
        "explanation": "This pattern (Hypothetical Document Embeddings or Chunk Summaries) matches abstract user queries to clean summaries, while passing full detailed text to the generation LLM.",
    },
    {
        "q": "How does ColPali (Vision-based Document Retrieval) disrupt traditional PDF parsing pipelines?",
        "options": [
            "A) It uses regex to parse text files faster",
            "B) It embeds full page images directly using Vision Language Models (ColBERT + PaliGemma), bypassing text extraction and OCR entirely",
            "C) It converts PDFs into audio waveforms",
            "D) It requires documents to be printed on physical paper",
        ],
        "answer": "B",
        "explanation": "ColPali represents document pages as multi-vector vision patches, eliminating brittle OCR, table extractors, and font heuristics completely.",
    },
    {
        "q": "When tuning `HNSW` search at query time, what does the parameter `efSearch` control?",
        "options": [
            "A) The number of nearest neighbors returned to the user",
            "B) The size of the dynamic priority queue evaluated during query time (higher `efSearch` = higher recall but higher latency)",
            "C) The timeout in milliseconds",
            "D) The maximum token count of the prompt",
        ],
        "answer": "B",
        "explanation": "`efSearch` determines the exploration depth during nearest-neighbor search. Increasing `efSearch` explores more candidate paths, increasing recall at the expense of query time.",
    },
    {
        "q": "What is the main drawback of setting `RecursiveCharacterTextSplitter(chunk_overlap=500, chunk_size=1000)` (50% overlap)?",
        "options": [
            "A) Chunk boundaries will fail to split",
            "B) High redundancy, nearly doubling total chunks and embedding storage costs, and risking duplicate information in top-k retrieval",
            "C) The splitter will throw a ValueError",
            "D) Cosine similarity cannot exceed 0.5",
        ],
        "answer": "B",
        "explanation": "A 50% overlap generates nearly double the number of chunks, doubling storage and embedding costs, and causes redundant near-identical chunks to crowd out diversity in top-k.",
    },
    {
        "q": "You need to summarize 1,000 independent customer review snippets. Context window is 8,000 tokens. Which approach is most token-efficient and cost-effective?",
        "options": [
            "A) 1,000 sequential Refine calls",
            "B) Batching reviews into groups of ~25 snippets per prompt, summarizing each batch, and then reducing batch summaries (hierarchical batch Map-Reduce)",
            "C) Stuffing all 1,000 reviews into 1 prompt",
            "D) One individual Map-Reduce per sentence",
        ],
        "answer": "B",
        "explanation": "Batching 25 reviews per call reduces 1,000 calls down to 40 batch calls, staying well within context limits while drastically reducing API latency and overhead.",
    },
    {
        "q": "Why does using `split_text()` instead of `split_documents()` break source citation capabilities in a production RAG application?",
        "options": [
            "A) `split_text()` deletes the original PDF file from disk",
            "B) `split_text()` discards the Document's `metadata` dictionary (containing `source`, `page`, `author`), returning only raw strings",
            "C) `split_text()` encrypts the text with SHA-256",
            "D) `split_text()` only outputs 10 characters at a time",
        ],
        "answer": "B",
        "explanation": "`split_text()` takes a string and returns strings. Any page numbers, file names, or IDs stored in Document `metadata` are lost, making citations impossible.",
    },
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🎯 QUIZ RUNNER FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_quiz(questions, difficulty_name, num_questions=None, emoji=""):
    """Runs an interactive quiz for a set of questions."""
    selected = list(questions)
    random.shuffle(selected)
    if num_questions and num_questions < len(selected):
        selected = selected[:num_questions]

    score = 0
    total = len(selected)

    print(f"\n{'=' * 65}")
    print(f"{emoji} {difficulty_name.upper()} LEVEL QUIZ — {total} Questions")
    print(f"{'=' * 65}\n")

    for i, item in enumerate(selected, 1):
        print(f"Q{i}/{total}: {item['q']}")
        for opt in item["options"]:
            print(f"   {opt}")

        user_choice = ""
        while user_choice not in ["A", "B", "C", "D", "Q"]:
            user_choice = input("\n👉 Your Answer (A/B/C/D) or 'Q' to quit: ").strip().upper()

        if user_choice == "Q":
            print(f"\nQuiz exited early. Score: {score}/{i - 1}")
            return score, i - 1

        if user_choice == item["answer"]:
            print("   ✅ CORRECT!")
            score += 1
        else:
            print(f"   ❌ INCORRECT. Correct answer is {item['answer']}")

        print(f"   💡 Explanation: {item['explanation']}\n")
        print("-" * 65)

    pct = (score / total) * 100 if total > 0 else 0
    print(f"\n📊 {difficulty_name} Level Complete! Score: {score}/{total} ({pct:.1f}%)")
    return score, total


def main():
    print("""
╔════════════════════════════════════════════════════════════════════╗
║    🧠 CHUNKING, EMBEDDINGS & DOCUMENT PROCESSING QUIZ             ║
║                  GenAI L2 Exam Self-Assessment                     ║
╚════════════════════════════════════════════════════════════════════╝

75 Total Practice Questions:
  🟢 1. Easy Mode    (25 Qs - Parsers, splitters, embeddings, Stuff)
  🟡 2. Medium Mode  (25 Qs - Map-Reduce, Refine, MRL, HNSW, tuning)
  🔴 3. Hard Mode    (25 Qs - Latency trade-offs, math, traps & code)
  🎲 4. Quick Mix    (15 Qs - 5 from each tier)
  🏆 5. Full Exam    (75 Qs - Complete mock exam simulation)
""")

    choice = input("Select an option (1-5) [Default: 4]: ").strip()
    if not choice:
        choice = "4"

    total_score = 0
    total_q = 0

    if choice == "1":
        s, t = run_quiz(EASY_QUESTIONS, "Easy", len(EASY_QUESTIONS), "🟢")
        total_score, total_q = s, t
    elif choice == "2":
        s, t = run_quiz(MEDIUM_QUESTIONS, "Medium", len(MEDIUM_QUESTIONS), "🟡")
        total_score, total_q = s, t
    elif choice == "3":
        s, t = run_quiz(HARD_QUESTIONS, "Hard", len(HARD_QUESTIONS), "🔴")
        total_score, total_q = s, t
    elif choice == "4":
        print("\n🎲 Starting Quick Mix (5 Easy, 5 Medium, 5 Hard)...")
        for pool, name, em in [
            (EASY_QUESTIONS, "Easy", "🟢"),
            (MEDIUM_QUESTIONS, "Medium", "🟡"),
            (HARD_QUESTIONS, "Hard", "🔴"),
        ]:
            s, t = run_quiz(pool, name, 5, em)
            total_score += s
            total_q += t
    elif choice == "5":
        print("\n🏆 FULL MOCK EXAM — 75 Questions!")
        for pool, name, em in [
            (EASY_QUESTIONS, "Easy", "🟢"),
            (MEDIUM_QUESTIONS, "Medium", "🟡"),
            (HARD_QUESTIONS, "Hard", "🔴"),
        ]:
            s, t = run_quiz(pool, name, len(pool), em)
            total_score += s
            total_q += t
    else:
        print("Invalid choice. Exiting.")
        return

    if total_q > 0:
        pct = (total_score / total_q) * 100
        print("\n" + "=" * 65)
        print(f"🏁 FINAL SCORE: {total_score}/{total_q} ({pct:.1f}%)")
        print("=" * 65)
        if pct >= 90:
            print("🌟 Outstanding! You have mastered Chunking, Embeddings & Document Processing for the GenAI L2 Exam!")
        elif pct >= 75:
            print("🎉 Great performance! Review the explanations for questions you missed.")
        elif pct >= 60:
            print("👍 Solid baseline! Re-read 01_Chunking_Embedding_Study_Guide.md to reinforce tricky areas.")
        else:
            print("📚 More practice needed. Review the study guide and retake the quiz!")


if __name__ == "__main__":
    main()
