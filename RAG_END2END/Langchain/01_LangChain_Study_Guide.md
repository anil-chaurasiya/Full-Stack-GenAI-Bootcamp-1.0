# 📚 LangChain Framework — Complete Study Guide (GenAI L2 Exam)

> **Goal**: Master every LangChain concept relevant to the GenAI L2 exam.
> Covers the full ecosystem: LangChain Core, LCEL, Agents, RAG pipeline, LangGraph, LangSmith, and Vertex AI integration.

---

## Table of Contents

1. [What is LangChain?](#1-what-is-langchain)
2. [The LangChain Ecosystem](#2-the-langchain-ecosystem)
3. [LangChain Core Architecture](#3-langchain-core-architecture)
4. [Models & LLM Wrappers](#4-models--llm-wrappers)
5. [Prompt Templates](#5-prompt-templates)
6. [LCEL — LangChain Expression Language](#6-lcel--langchain-expression-language)
7. [Document Loaders](#7-document-loaders)
8. [Text Splitters](#8-text-splitters)
9. [Embeddings](#9-embeddings)
10. [Vector Stores](#10-vector-stores)
11. [Retrievers](#11-retrievers)
12. [Output Parsers](#12-output-parsers)
13. [Chains](#13-chains)
14. [Memory](#14-memory)
15. [Tools & Tool Calling](#15-tools--tool-calling)
16. [Agents](#16-agents)
17. [LangGraph — Stateful Agent Orchestration](#17-langgraph--stateful-agent-orchestration)
18. [LangSmith — Observability & Evaluation](#18-langsmith--observability--evaluation)
19. [RAG Pipeline End-to-End](#19-rag-pipeline-end-to-end)
20. [LangChain + Google Cloud / Vertex AI](#20-langchain--google-cloud--vertex-ai)
21. [LangChain vs Alternatives](#21-langchain-vs-alternatives)
22. [Best Practices for Production](#22-best-practices-for-production)
23. [Key Terminology Glossary](#23-key-terminology-glossary)
24. [Common Exam Patterns & Traps](#24-common-exam-patterns--traps)

---

## 1. What is LangChain?

**LangChain** is an open-source framework for building applications powered by Large Language Models (LLMs). It provides modular, composable abstractions for connecting LLMs to data, tools, memory, and external services.

> **Exam Definition**: LangChain is a framework that simplifies LLM application development by providing standardized interfaces for models, prompts, chains, agents, tools, memory, retrievers, and output parsing — enabling rapid prototyping and production deployment of GenAI applications.

### Key Facts for Exam ⚡

| Fact | Detail |
|------|--------|
| **Created by** | Harrison Chase / LangChain Inc. |
| **Language** | Python & TypeScript (JavaScript) |
| **License** | MIT (open source) |
| **Core Concept** | Composability — build complex apps from simple pieces |
| **Signature Syntax** | LCEL pipe operator: `prompt \| llm \| parser` |
| **Ecosystem** | LangChain (core) + LangGraph (agents) + LangSmith (observability) |

---

## 2. The LangChain Ecosystem

```
┌──────────────────────────────────────────────────────────────┐
│                   LANGCHAIN ECOSYSTEM                        │
│                                                              │
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────┐│
│  │   LangChain     │  │   LangGraph      │  │  LangSmith  ││
│  │   (Core)        │  │   (Orchestration) │  │  (Observe)  ││
│  │                 │  │                  │  │             ││
│  │ • Models        │  │ • Stateful Graphs│  │ • Tracing   ││
│  │ • Prompts       │  │ • Nodes & Edges  │  │ • Debugging ││
│  │ • Chains (LCEL) │  │ • Checkpointing  │  │ • Evaluation││
│  │ • Retrievers    │  │ • Human-in-Loop  │  │ • Monitoring││
│  │ • Tools         │  │ • Multi-Agent    │  │ • Datasets  ││
│  │ • Output Parsers│  │ • Cycles/Loops   │  │ • Feedback  ││
│  │ • Document Ldrs │  │                  │  │             ││
│  │ • Text Splitters│  │                  │  │             ││
│  │ • Vector Stores │  │                  │  │             ││
│  │ • Memory        │  │                  │  │             ││
│  └─────────────────┘  └──────────────────┘  └─────────────┘│
│                                                              │
│  ┌──────────────────────────────────────────────────────────┐│
│  │              langchain-community / langchain-{provider}  ││
│  │  Third-party integrations: OpenAI, Google, Anthropic,    ││
│  │  Groq, HuggingFace, Pinecone, ChromaDB, FAISS, etc.     ││
│  └──────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────┘
```

### Role of Each Component

| Component | What It Does | When to Use |
|-----------|-------------|-------------|
| **LangChain** | Core primitives: models, prompts, chains, retrievers, tools | Every LLM app — the foundation |
| **LangGraph** | Stateful multi-step agent workflows with graph architecture | Complex agents, loops, human-in-the-loop |
| **LangSmith** | Tracing, debugging, evaluation, monitoring | Development → production lifecycle |

### Exam Key ⚡
- LangChain = **building blocks** (composable primitives)
- LangGraph = **orchestration engine** (stateful agents, graphs)
- LangSmith = **observability platform** (trace, debug, evaluate)
- These three work **together**, not as alternatives.

---

## 3. LangChain Core Architecture

```
┌─────────────────────────────────────────────┐
│          LangChain Application               │
│                                             │
│  ┌──────────┐   ┌────────┐   ┌───────────┐ │
│  │  Prompt   │──▶│  Model │──▶│  Output   │ │
│  │ Template  │   │  (LLM) │   │  Parser   │ │
│  └──────────┘   └────────┘   └───────────┘ │
│       │              │             │         │
│       └──── LCEL (pipe: | ) ───────┘         │
│                                             │
│  ┌───────────────────────────────────────┐  │
│  │         Supporting Components          │  │
│  │  • Document Loaders  • Text Splitters │  │
│  │  • Embeddings        • Vector Stores  │  │
│  │  • Retrievers        • Memory         │  │
│  │  • Tools             • Agents         │  │
│  └───────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

### Package Structure (Current)

| Package | Purpose |
|---------|---------|
| `langchain-core` | Base interfaces, LCEL, prompt templates, output parsers |
| `langchain` | Chains, agents, retrieval strategies |
| `langchain-community` | Third-party integrations |
| `langchain-openai` | OpenAI-specific (ChatOpenAI, OpenAIEmbeddings) |
| `langchain-google-genai` | Google Gemini integration |
| `langchain-groq` | Groq LPU integration |
| `langgraph` | Graph-based agent orchestration |
| `langsmith` | Tracing and evaluation SDK |

---

## 4. Models & LLM Wrappers

LangChain provides a **unified interface** for all LLM providers. You swap models by changing one line of code.

### Chat Models (Primary)

```python
# OpenAI
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Google Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

# Groq (free tier)
from langchain_groq import ChatGroq
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# Anthropic
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)
```

### Key Interface Methods

| Method | Description |
|--------|-------------|
| `llm.invoke(input)` | Single synchronous call |
| `llm.stream(input)` | Streaming response (token by token) |
| `llm.batch([inputs])` | Process multiple inputs in parallel |
| `llm.ainvoke(input)` | Async version of invoke |
| `llm.bind_tools(tools)` | Attach tools for function calling |

### Exam Key ⚡
- LangChain's unified interface means **models are interchangeable**.
- All chat models accept `messages` (list of System/Human/AI messages).
- `temperature=0` for factual/RAG tasks; higher for creative tasks.
- Always load API keys from environment variables (`.env`), never hardcode.

---

## 5. Prompt Templates

### ChatPromptTemplate (Primary)

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant specialized in {domain}."),
    ("human", "{question}"),
])

# Format with variables
messages = prompt.format_messages(domain="GenAI", question="What is RAG?")
```

### FewShotChatMessagePromptTemplate

```python
from langchain_core.prompts import FewShotChatMessagePromptTemplate

examples = [
    {"input": "What is FAISS?", "output": "FAISS is a vector similarity search library by Meta."},
]

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"), ("ai", "{output}")
])

few_shot = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt, examples=examples
)

final_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer concisely."), few_shot, ("human", "{input}")
])
```

### MessagesPlaceholder

```python
from langchain_core.prompts import MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),  # Dynamic message list
    ("human", "{question}"),
])
```

### Exam Key ⚡
- `ChatPromptTemplate` is the standard for chat models.
- Use `MessagesPlaceholder` for dynamic conversation history.
- Templates use `{variable}` syntax for input variables.
- Double braces `{{` for literal braces in prompts (escaping).

---

## 6. LCEL — LangChain Expression Language

**LCEL is the most important concept to understand.** It's LangChain's declarative way to compose chains.

### The Pipe Operator `|`

```python
# The simplest chain: prompt → model → parser
chain = prompt | llm | StrOutputParser()

# Invoke the chain
result = chain.invoke({"question": "What is RAG?"})
```

### How LCEL Works

```
Input Dict ──▶ Prompt Template ──▶ Model ──▶ Output Parser ──▶ Result
              (formats input)    (generates)  (structures)
```

### Key LCEL Features

| Feature | Description | Example |
|---------|-------------|---------|
| **Pipe `\|`** | Chain components left to right | `prompt \| llm \| parser` |
| **Streaming** | Automatic token-by-token streaming | `for chunk in chain.stream(input)` |
| **Batch** | Parallel processing of multiple inputs | `chain.batch([input1, input2])` |
| **Async** | Native async support | `await chain.ainvoke(input)` |
| **Parallel** | Run multiple chains in parallel | `RunnableParallel(a=chain_a, b=chain_b)` |
| **Passthrough** | Pass input data through unchanged | `RunnablePassthrough()` |
| **Lambda** | Custom transformations inline | `RunnableLambda(my_function)` |

### RunnableParallel & RunnablePassthrough

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# Pass the question through AND retrieve context in parallel
setup = RunnableParallel(
    context=retriever,                    # Fetch docs
    question=RunnablePassthrough(),       # Pass question unchanged
)

rag_chain = setup | prompt | llm | StrOutputParser()
```

### Exam Key ⚡
- LCEL uses the **pipe `|` operator** to chain components.
- Every component in LCEL is a **Runnable** with `.invoke()`, `.stream()`, `.batch()`.
- `RunnablePassthrough()` passes input through unchanged — critical for RAG chains.
- `RunnableParallel()` runs multiple steps simultaneously.
- LCEL handles streaming, batching, and async automatically.
- This is the **modern replacement** for the old `LLMChain`, `SequentialChain` pattern.

---

## 7. Document Loaders

Document loaders ingest raw data and convert it to LangChain `Document` objects.

### Common Loaders

| Loader | Source | Import |
|--------|--------|--------|
| `PyPDFLoader` | PDF files | `langchain_community.document_loaders` |
| `CSVLoader` | CSV files | `langchain_community.document_loaders` |
| `TextLoader` | Plain text | `langchain_community.document_loaders` |
| `UnstructuredFileLoader` | Multiple formats | `langchain_community.document_loaders` |
| `WebBaseLoader` | Web pages | `langchain_community.document_loaders` |
| `WikipediaLoader` | Wikipedia | `langchain_community.document_loaders` |
| `DirectoryLoader` | Entire directories | `langchain_community.document_loaders` |
| `Docx2txtLoader` | Word documents | `langchain_community.document_loaders` |

### Document Object

```python
from langchain_core.documents import Document

doc = Document(
    page_content="The actual text content...",
    metadata={"source": "file.pdf", "page": 1}
)
```

### Exam Key ⚡
- All loaders produce `Document` objects with `page_content` (text) and `metadata` (dict).
- `metadata` enables filtering during retrieval (source tracking, page numbers).
- Choose the loader based on your data source format.

---

## 8. Text Splitters

Text splitters break large documents into smaller chunks for embedding and retrieval.

### RecursiveCharacterTextSplitter (Default/Recommended)

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,       # Max characters per chunk
    chunk_overlap=200,     # Overlap between consecutive chunks
    separators=["\n\n", "\n", " ", ""]  # Priority order
)

chunks = splitter.split_documents(documents)
```

### How Recursive Splitting Works

```
Try splitting by "\n\n" (paragraphs) first
  ├── If chunks are small enough → Done ✅
  └── If chunks are too large → Try "\n" (lines)
        ├── If small enough → Done ✅
        └── If too large → Try " " (words)
              └── If too large → Split by character ""
```

### Other Splitters

| Splitter | Best For |
|----------|----------|
| `RecursiveCharacterTextSplitter` | General text (DEFAULT) |
| `CharacterTextSplitter` | Simple fixed-size splits |
| `TokenTextSplitter` | Token-aware splitting (for models) |
| `MarkdownHeaderTextSplitter` | Markdown documents (preserves headings) |
| `HTMLHeaderTextSplitter` | HTML documents |
| `CodeTextSplitter` | Source code (language-aware) |
| `SemanticChunker` | Embedding-based semantic splitting |

### Exam Key ⚡
- **`RecursiveCharacterTextSplitter` is the industry default** — always recommend this unless specific format needs.
- `chunk_overlap` prevents **information loss at boundaries**.
- Too small chunks → loss of context; too large → noise in retrieval.
- Typical values: `chunk_size=500-1500`, `chunk_overlap=50-200`.

---

## 9. Embeddings

Embeddings convert text into dense numerical vectors for semantic search.

### Common Embedding Models

```python
# OpenAI
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Google
from langchain_google_genai import GoogleGenerativeAIEmbeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# HuggingFace (free, local)
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
```

### Key Methods

| Method | Purpose |
|--------|---------|
| `embed_documents(texts)` | Embed a list of texts (for indexing) |
| `embed_query(text)` | Embed a single query (for search) |

### Exam Key ⚡
- Embeddings are the **bridge between text and vector databases**.
- **Always specify the embedding model explicitly** — don't rely on defaults.
- The same embedding model must be used for **both indexing and querying**.
- Dimension matters: `text-embedding-3-small` = 1536 dims, `all-MiniLM-L6-v2` = 384 dims.

---

## 10. Vector Stores

Vector stores index and retrieve embeddings for similarity search.

### Common Vector Stores

| Store | Type | Best For |
|-------|------|----------|
| **ChromaDB** | Local (embedded) | Development, prototyping |
| **FAISS** | Local (in-memory) | Fast local search, no persistence needed |
| **Pinecone** | Cloud (managed) | Production, scalable |
| **Qdrant** | Cloud or local | Production, filtering |
| **Weaviate** | Cloud or local | Hybrid search |

### Usage Pattern

```python
from langchain_chroma import Chroma

# Create vector store from documents
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="my_collection",
    persist_directory="./chroma_db"
)

# Convert to retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",       # or "mmr"
    search_kwargs={"k": 4}          # top-4 results
)
```

### Search Types

| Search Type | Description |
|-------------|-------------|
| `similarity` | Standard cosine similarity search |
| `mmr` | Maximum Marginal Relevance — balances relevance and diversity |
| `similarity_score_threshold` | Only return results above a score threshold |

### Exam Key ⚡
- ChromaDB for **local development**, Pinecone/Qdrant for **production**.
- `.as_retriever()` converts a vector store into a LangChain Retriever.
- **MMR** reduces redundancy in results — important for diverse context in RAG.
- Always persist vector stores for production (avoid re-indexing).

---

## 11. Retrievers

Retrievers fetch relevant documents from a data source given a query.

### Types of Retrievers

| Retriever | Description |
|-----------|-------------|
| `VectorStoreRetriever` | Standard similarity search from vector store |
| `MultiQueryRetriever` | Generates multiple query variants for broader recall |
| `ContextualCompressionRetriever` | Compresses/filters retrieved docs to remove noise |
| `EnsembleRetriever` | Combines results from multiple retrievers (hybrid search) |
| `SelfQueryRetriever` | Uses LLM to convert natural language to structured filters |
| `ParentDocumentRetriever` | Retrieves parent documents of matched chunks |
| `BM25Retriever` | Keyword-based (sparse) retrieval |

### Hybrid Search Example

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# Combine dense (vector) + sparse (keyword) retrieval
ensemble = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.7, 0.3]  # 70% semantic, 30% keyword
)
```

### Exam Key ⚡
- Retrievers are the **interface** between your data and the LLM.
- **Hybrid search** (dense + sparse) often outperforms either alone.
- `MultiQueryRetriever` improves recall by rephrasing the query.
- `k` parameter controls how many documents to retrieve (typically 3-5).

---

## 12. Output Parsers

Output parsers convert LLM text output into structured data.

### Available Parsers

| Parser | Output Type | Use Case |
|--------|-------------|----------|
| `StrOutputParser` | `str` | Raw text output |
| `JsonOutputParser` | `dict` | JSON responses |
| `PydanticOutputParser` | Pydantic model | Type-safe structured output |
| `CommaSeparatedListOutputParser` | `list[str]` | Comma-separated lists |
| `OutputFixingParser` | Any | Auto-fixes malformed output via retry |

### PydanticOutputParser Example

```python
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class MovieReview(BaseModel):
    title: str = Field(description="Movie title")
    rating: float = Field(description="Rating out of 10")
    summary: str = Field(description="Brief review summary")

parser = PydanticOutputParser(pydantic_object=MovieReview)

# Include format instructions in prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract movie review info.\n{format_instructions}"),
    ("human", "{review_text}"),
])

chain = prompt | llm | parser
result = chain.invoke({
    "review_text": "...",
    "format_instructions": parser.get_format_instructions()
})
# result is a MovieReview object with validated types
```

### Exam Key ⚡
- `StrOutputParser` is the simplest — just returns raw text.
- `PydanticOutputParser` provides **type validation** and **schema enforcement**.
- `parser.get_format_instructions()` generates instructions the LLM needs to produce correct output.
- Always use parsers in the LCEL chain: `prompt | llm | parser`.

---

## 13. Chains

Chains are sequences of operations that process input to produce output.

### Modern Chains (LCEL)

```python
# Simple QA chain
qa_chain = prompt | llm | StrOutputParser()

# RAG chain with retrieval
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Multi-step chain
step1 = prompt_extract | llm | JsonOutputParser()
step2 = prompt_summarize | llm | StrOutputParser()
multi_chain = step1 | step2
```

### Legacy Chains (Know for Exam Context)

| Legacy Chain | LCEL Equivalent |
|-------------|-----------------|
| `LLMChain` | `prompt \| llm \| parser` |
| `SequentialChain` | `chain1 \| chain2` |
| `StuffDocumentsChain` | LCEL with `RunnableParallel` |
| `MapReduceDocumentsChain` | Custom LCEL with map/reduce |
| `ConversationalRetrievalChain` | LCEL with `MessagesPlaceholder` |

### Document Combination Strategies

| Strategy | How It Works | Best For |
|----------|-------------|----------|
| **Stuff** | Concatenate all docs into one prompt | Small document sets |
| **Map-Reduce** | Process each doc separately, then combine | Large document sets |
| **Refine** | Iteratively refine answer with each doc | Sequential reasoning |
| **Map-Rerank** | Score each doc's answer, pick the best | Precise answers |

### Exam Key ⚡
- LCEL chains are the **modern standard** — legacy chains are deprecated.
- The **Stuff** strategy is simplest but limited by context window.
- **Map-Reduce** handles large document sets by processing in parallel.
- Always prefer LCEL syntax over legacy `LLMChain`, `SequentialChain`.

---

## 14. Memory

Memory allows models to retain context across interactions.

### Memory Types

| Memory Type | Description | Use Case |
|-------------|-------------|----------|
| `ConversationBufferMemory` | Stores all messages verbatim | Short conversations |
| `ConversationBufferWindowMemory` | Keeps last N messages | Medium conversations |
| `ConversationSummaryMemory` | Summarizes past conversation | Long conversations |
| `ConversationSummaryBufferMemory` | Recent messages + older summary | Balanced approach |
| `ConversationTokenBufferMemory` | Keeps messages up to token limit | Token-aware management |

### Modern Approach: Message History

```python
from langchain_core.chat_history import InMemoryChatMessageHistory

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Wrap chain with message history
from langchain_core.runnables.history import RunnableWithMessageHistory

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history",
)
```

### Exam Key ⚡
- **Short-term memory** = conversation history within a session.
- **Long-term memory** = persistent storage across sessions (via LangGraph or external DB).
- `ConversationBufferMemory` is simplest but grows unbounded.
- `ConversationSummaryMemory` is best for long conversations (compresses old messages).
- In production, use **LangGraph checkpointing** for persistent state.

---

## 15. Tools & Tool Calling

Tools are functions that agents can use to interact with external systems.

### Defining Tools

```python
from langchain_core.tools import tool

@tool
def search_web(query: str) -> str:
    """Search the web for information about a topic."""
    # ... implementation ...
    return f"Results for: {query}"

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))
```

### Binding Tools to Models

```python
# Bind tools to the model
llm_with_tools = llm.bind_tools([search_web, calculate])

# The model can now decide to call tools
response = llm_with_tools.invoke("What is 25 * 17?")
```

### Built-in Tools

| Tool | Description |
|------|-------------|
| `WikipediaQueryRun` | Query Wikipedia |
| `DuckDuckGoSearchRun` | Web search |
| `PythonREPLTool` | Execute Python code |
| `ShellTool` | Execute shell commands |
| `RequestsGetTool` | HTTP GET requests |

### Exam Key ⚡
- Use `@tool` decorator for custom tools — include clear docstrings.
- `llm.bind_tools()` enables the model to use function calling.
- Tool descriptions are **critical** — the LLM uses them to decide which tool to call.
- Tools can have **side effects** (database writes, API calls).

---

## 16. Agents

Agents use LLMs as reasoning engines to decide which tools to call.

### ReAct Agent (Standard Pattern)

```python
from langgraph.prebuilt import create_react_agent

tools = [search_web, calculate]
agent = create_react_agent(llm, tools)

result = agent.invoke({"messages": [("human", "What is the weather in NYC?")]})
```

### Agent Loop

```
User Query → LLM Reasoning → Tool Selection → Tool Execution
     ↑                                              │
     └────────── Observation ◀──────────────────────┘
                (repeat until answer is ready)
```

### Agent Types

| Agent Type | Description | Framework |
|------------|-------------|-----------|
| **ReAct** | Thought → Action → Observation loop | LangGraph |
| **Tool Calling** | Native function calling via model API | LangChain |
| **Plan and Execute** | Plan all steps first, then execute | LangGraph |
| **Multi-Agent** | Multiple specialized agents collaborate | LangGraph |

### Exam Key ⚡
- Modern agents use `create_react_agent()` from **LangGraph**, not legacy `AgentExecutor`.
- Agents are **autonomous** — they decide which tools to use based on the query.
- The ReAct pattern (Thought-Action-Observation) is the standard agent pattern.
- Always implement **error handling** and **max iterations** for safety.

---

## 17. LangGraph — Stateful Agent Orchestration

LangGraph extends LangChain for complex, stateful agent workflows.

### Core Concepts

```
┌─────────────────────────────────────────────┐
│              LANGGRAPH                       │
│                                             │
│   ┌──────┐     ┌──────┐     ┌──────┐      │
│   │Node A│────▶│Node B│────▶│Node C│      │
│   │(LLM) │     │(Tool)│     │(Parse)│      │
│   └──────┘     └──┬───┘     └──────┘      │
│                   │                         │
│          Conditional Edge                   │
│            ┌──────▼──────┐                  │
│            │  Need more  │                  │
│            │  info?      │                  │
│            └──┬──────┬───┘                  │
│           Yes │      │ No                   │
│               │      │                      │
│           ┌───▼──┐   └──▶ END               │
│           │Node D│                          │
│           │(Search)──────▶ (back to B)      │
│           └──────┘                          │
│                                             │
│   State: {messages: [...], context: "..."}  │
└─────────────────────────────────────────────┘
```

### Key Components

| Component | Description |
|-----------|-------------|
| **Nodes** | Python functions that perform work (LLM calls, tool execution) |
| **Edges** | Connections between nodes (fixed or conditional) |
| **State** | Shared data structure accessible by all nodes (TypedDict) |
| **Conditional Edges** | Logic-based routing (if/else branching) |
| **Checkpointing** | State snapshots for persistence, recovery, time-travel |
| **Human-in-the-Loop** | Interrupt execution for human approval, then resume |

### LangGraph vs LangChain Chains

| Feature | LangChain (Chains) | LangGraph (Graphs) |
|---------|:---:|:---:|
| **Flow** | Linear, sequential | Any topology (loops, branches) |
| **State** | Simple input→output | Rich, persistent, shared state |
| **Loops/Cycles** | ❌ Not supported | ✅ Core feature |
| **Human-in-the-Loop** | Difficult | ✅ Native support |
| **Checkpointing** | ❌ | ✅ Automatic state snapshots |
| **Error Recovery** | Manual | ✅ Resume from last checkpoint |
| **Use Case** | Simple Q&A, RAG | Complex agents, multi-agent |

### Exam Key ⚡
- LangGraph = **graphs** (nodes + edges + state). LangChain = **chains** (linear pipes).
- Use LangChain for simple RAG/Q&A; use LangGraph when you need **loops, state, or human-in-the-loop**.
- **Checkpointing** enables persistence, crash recovery, and debugging.
- **Conditional edges** enable decision-making in the flow.
- LangGraph is now the **standard for production agents** in the LangChain ecosystem.

---

## 18. LangSmith — Observability & Evaluation

LangSmith provides tracing, debugging, evaluation, and monitoring for LLM applications.

### Core Features

| Feature | Description |
|---------|-------------|
| **Tracing** | See every step of your chain/agent execution (inputs, outputs, latency, tokens) |
| **Debugging** | Inspect failures, identify which step broke, view exact prompts sent to LLMs |
| **Evaluation** | Run your chain against test datasets and score with automated evaluators |
| **Monitoring** | Track production metrics (latency, cost, error rates, feedback) |
| **Datasets** | Create curated test datasets for regression testing |
| **Feedback** | Collect user feedback (thumbs up/down) to improve quality |

### How Tracing Works

```
User Query → [Traced] Prompt Template → [Traced] LLM Call → [Traced] Parser
                 │                         │                     │
                 └─────────────────────────┴─────────────────────┘
                              All visible in LangSmith Dashboard
```

### Exam Key ⚡
- LangSmith is the **observability** layer — not the execution layer.
- Enable with environment variables: `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_API_KEY=...`
- Critical for debugging: see the **exact prompt** sent to the LLM at every step.
- Use **evaluators** to systematically test chain quality (correctness, relevance, faithfulness).
- Supports both **automated** and **human** evaluation.

---

## 19. RAG Pipeline End-to-End

The complete RAG pipeline using LangChain:

```
┌─────────────────── INDEXING PHASE ──────────────────────┐
│                                                          │
│  Documents → Loader → Splitter → Embeddings → VectorDB  │
│  (PDFs,      (PyPDF)  (Recursive)  (OpenAI/    (Chroma/  │
│   CSVs,               Char Text    Google)      FAISS/   │
│   Web)                Splitter                  Pinecone) │
│                                                          │
└──────────────────────────────────────────────────────────┘

┌─────────────────── RETRIEVAL PHASE ─────────────────────┐
│                                                          │
│  User Query → Embeddings → VectorDB Search → Top-K Docs │
│                (same model   (similarity/     (context)  │
│                 as indexing)   MMR search)                │
│                                                          │
└──────────────────────────────────────────────────────────┘

┌─────────────────── GENERATION PHASE ────────────────────┐
│                                                          │
│  Prompt Template                                         │
│  ┌──────────────────────────────────────┐               │
│  │ System: "Answer from context only"   │               │
│  │ Context: {retrieved_docs}            │               │
│  │ Question: {user_query}               │               │
│  └──────────────────────────────────────┘               │
│           │                                              │
│           ▼                                              │
│     LLM → Output Parser → Final Answer                   │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### Complete LCEL RAG Chain

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Prompt
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer based on context only. If not found, say 'I don't know'."),
    ("human", "Context:\n{context}\n\nQuestion: {question}\nAnswer:"),
])

# Chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("What is vector search?")
```

### Exam Key ⚡
- RAG has **3 phases**: Indexing, Retrieval, Generation.
- Use the **same embedding model** for indexing and querying.
- `RecursiveCharacterTextSplitter` is the default splitter.
- Include **grounding instructions** and **fallback** in the prompt.
- `RunnablePassthrough()` is essential to pass the question through in RAG chains.

---

## 20. LangChain + Google Cloud / Vertex AI

### Google Gemini Integration

```python
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
```

### Vertex AI Integration

```python
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings

llm = ChatVertexAI(model_name="gemini-2.0-flash", project="my-project")
embeddings = VertexAIEmbeddings(model_name="text-embedding-004")
```

### Google Cloud RAG APIs

- **Vertex AI Search** — Managed RAG service (no manual chunking/embedding needed)
- **Vertex AI Agent Builder** — Visual agent creation with grounding
- **Vertex AI Extensions** — Connect to external APIs (like MCP servers)
- **Model Garden** — Pre-trained models (Gemini, PaLM, open-source)

### Exam Key ⚡
- `langchain-google-genai` = Google AI Studio (free tier, API key auth).
- `langchain-google-vertexai` = Google Cloud Vertex AI (enterprise, service account auth).
- Google offers **managed RAG** via Vertex AI Search — no manual pipeline needed.
- LangChain integrates with Vertex AI for both **models** and **embeddings**.

---

## 21. LangChain vs Alternatives

| Dimension | LangChain | LlamaIndex | Haystack | Direct API |
|-----------|:---:|:---:|:---:|:---:|
| **Focus** | General LLM apps | RAG-specialized | Search/QA pipelines | Maximum control |
| **Complexity** | Medium | Medium | Medium | Low |
| **Agents** | ✅ (via LangGraph) | ⚠️ Basic | ⚠️ Basic | ❌ Manual |
| **RAG** | ✅ Full pipeline | ✅ Optimized | ✅ Full pipeline | ⚠️ Manual |
| **Observability** | ✅ LangSmith | ✅ Built-in | ⚠️ Limited | ❌ Custom |
| **Ecosystem** | Largest | Growing | Moderate | N/A |
| **Learning Curve** | Steeper | Moderate | Moderate | Simplest |

### When to Use What

- **LangChain**: General-purpose LLM apps, agents, complex workflows, largest ecosystem.
- **LlamaIndex**: RAG-focused applications where retrieval quality is the top priority.
- **Direct API**: Simple prototypes, maximum control, minimal dependencies.

---

## 22. Best Practices for Production

### Code Quality ✅

1. **Use LCEL** — avoid legacy chains (`LLMChain`, `SequentialChain`)
2. **Specify models explicitly** — never rely on defaults
3. **Load API keys from `.env`** — never hardcode secrets
4. **Wrap LLM calls in try/except** — graceful error handling
5. **Use type hints** — improves code readability and debugging

### RAG Pipeline ✅

1. **RecursiveCharacterTextSplitter** as default chunking strategy
2. **Same embedding model** for indexing and querying
3. **Persist vector stores** — avoid re-indexing on every run
4. **Include chunk_overlap** — prevents information loss
5. **Use MMR** when diversity in retrieved results matters

### Agents ✅

1. Use **LangGraph** for production agents (not legacy AgentExecutor)
2. Set **max iterations** to prevent infinite loops
3. Implement **human-in-the-loop** for sensitive operations
4. Use **checkpointing** for crash recovery
5. **Clear tool descriptions** — the LLM reads them

### Observability ✅

1. Enable **LangSmith tracing** in all environments
2. Create **evaluation datasets** for regression testing
3. Monitor **latency, cost, and error rates** in production
4. Collect **user feedback** to improve quality over time

---

## 23. Key Terminology Glossary

| Term | Definition |
|------|-----------|
| **LangChain** | Open-source framework for building LLM applications |
| **LCEL** | LangChain Expression Language — declarative chain composition using pipe `\|` |
| **Runnable** | Base interface for all LCEL components (invoke, stream, batch) |
| **Chain** | A sequence of operations composed with LCEL |
| **Agent** | LLM that autonomously decides which tools to use |
| **Tool** | Function an agent can call to interact with external systems |
| **Retriever** | Interface for fetching relevant documents from data sources |
| **Document** | Object with `page_content` (text) and `metadata` (dict) |
| **Document Loader** | Ingests raw data into Document objects |
| **Text Splitter** | Breaks documents into smaller chunks |
| **Embeddings** | Converts text to dense numerical vectors |
| **Vector Store** | Database for indexing and searching embeddings |
| **Output Parser** | Converts LLM text output into structured data |
| **Memory** | Mechanism for retaining context across interactions |
| **LangGraph** | Graph-based orchestration for stateful agents |
| **LangSmith** | Observability platform (tracing, debugging, evaluation) |
| **RunnablePassthrough** | Passes input through unchanged in LCEL chains |
| **RunnableParallel** | Runs multiple operations in parallel in LCEL |
| **ChatPromptTemplate** | Template for formatting chat model inputs |
| **MessagesPlaceholder** | Dynamic placeholder for conversation history |
| **ReAct** | Reasoning + Acting — standard agent pattern |
| **Checkpointing** | State snapshots in LangGraph for persistence/recovery |
| **Human-in-the-Loop** | Pausing agent execution for human approval |
| **MMR** | Maximum Marginal Relevance — diversity-aware search |
| **Stuff/Map-Reduce** | Document combination strategies for chains |

---

## 24. Common Exam Patterns & Traps

### ❓ Frequently Tested Concepts

1. **"What is LCEL?"**
   → LangChain Expression Language — compose chains with the pipe `|` operator.

2. **"What is the default text splitter recommendation?"**
   → `RecursiveCharacterTextSplitter` — splits by paragraphs, then lines, then words.

3. **"What's the difference between LangChain and LangGraph?"**
   → LangChain = linear chains. LangGraph = graph-based with loops, state, human-in-the-loop.

4. **"How do you add memory/conversation history to a chain?"**
   → Use `MessagesPlaceholder` in the prompt + `RunnableWithMessageHistory` wrapper.

5. **"What are the three parts of the LangChain ecosystem?"**
   → LangChain (core), LangGraph (orchestration), LangSmith (observability).

6. **"What does RunnablePassthrough do?"**
   → Passes input through unchanged — critical in RAG chains to pass the question alongside retrieved context.

7. **"What search type reduces redundancy in results?"**
   → **MMR (Maximum Marginal Relevance)**.

8. **"How do you convert a vector store to a retriever?"**
   → `vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})`.

### ⚠️ Common Traps

| Trap | Why It's Wrong | Correct Answer |
|------|---------------|----------------|
| "LLMChain is the current way to build chains" | LLMChain is legacy/deprecated | Use LCEL: `prompt \| llm \| parser` |
| "Agents and Chains are the same thing" | Chains are deterministic; Agents make autonomous decisions | Agents decide which tools to use; chains follow a fixed sequence |
| "You can use different embedding models for indexing and querying" | Vectors would be incompatible | Must use the SAME embedding model for both |
| "LangGraph replaces LangChain" | LangGraph builds ON TOP of LangChain | They are complementary — use together |
| "Memory stores the model's weights" | Memory stores conversation history | Memory = message history, not model parameters |
| "More chunks always improve retrieval" | Too small chunks lose context; too many add noise | Balance chunk_size and chunk_overlap for your use case |
| "LangSmith is required to run LangChain" | LangSmith is optional observability | LangChain works independently; LangSmith adds tracing |

---

## Quick Decision Flowchart 🧭

```
"What LangChain component do I need?"

Simple Q&A?                     → LCEL chain: prompt | llm | parser
Need external data?             → + Document Loader + Splitter + Vector Store
Need contextual answers?        → + Retriever + RAG chain
Need conversation history?      → + Memory / MessagesPlaceholder
Need to call external APIs?     → + Tools + bind_tools
Need autonomous decision-making?→ + Agent (create_react_agent)
Need loops/human approval?      → + LangGraph
Need debugging/monitoring?      → + LangSmith
```

---

> **Final Tip for L2 Exam**: LangChain questions focus on **understanding the components and when to use them** — not on memorizing code syntax. Know the **RAG pipeline flow** (Loader → Splitter → Embeddings → VectorStore → Retriever → LLM), understand **LCEL**, and distinguish **LangChain vs LangGraph vs LangSmith**.

---

*Study Material Created for GenAI L2 Exam Preparation*  
*Path: `RAG_END2END/Langchain/`*
