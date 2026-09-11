# 🧠 LangChain Framework — Practice MCQs (GenAI L2 Exam)
# =========================================================
# 75 Multiple Choice Questions in 3 Difficulty Levels:
#   🟢 EASY (25 questions)    — Definitions, basic components
#   🟡 MEDIUM (25 questions)  — Architecture, LCEL, RAG pipeline
#   🔴 HARD (25 questions)    — LangGraph, production scenarios, traps
#
# Run this script to take an interactive self-assessment quiz.

import random

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🟢 EASY QUESTIONS (Definitions & Basic Components)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EASY_QUESTIONS = [
    {
        "q": "What is LangChain?",
        "options": [
            "A) A blockchain for AI models",
            "B) An open-source framework for building LLM-powered applications",
            "C) A language translation service",
            "D) A neural network training library",
        ],
        "answer": "B",
        "explanation": "LangChain is an open-source framework that provides modular components for building applications powered by Large Language Models.",
    },
    {
        "q": "What does LCEL stand for?",
        "options": [
            "A) Large Chain Expression Layer",
            "B) LangChain Expression Language",
            "C) Language Chain Evaluation Logic",
            "D) Linked Component Execution Language",
        ],
        "answer": "B",
        "explanation": "LCEL = LangChain Expression Language — the declarative syntax for composing chains using the pipe operator.",
    },
    {
        "q": "What operator does LCEL use to chain components together?",
        "options": [
            "A) + (plus)",
            "B) >> (right shift)",
            "C) | (pipe)",
            "D) -> (arrow)",
        ],
        "answer": "C",
        "explanation": "LCEL uses the pipe operator |  to chain components: prompt | llm | parser. Data flows left to right.",
    },
    {
        "q": "What are the THREE parts of the LangChain ecosystem?",
        "options": [
            "A) LangChain, LangModel, LangDB",
            "B) LangChain (core), LangGraph (orchestration), LangSmith (observability)",
            "C) LangChain, LangTrain, LangDeploy",
            "D) LangCore, LangAPI, LangUI",
        ],
        "answer": "B",
        "explanation": "The ecosystem is: LangChain (core primitives), LangGraph (stateful agent orchestration), LangSmith (tracing, debugging, evaluation).",
    },
    {
        "q": "What is a 'Document' object in LangChain?",
        "options": [
            "A) A PDF file",
            "B) An object with page_content (text) and metadata (dict)",
            "C) A database table",
            "D) A prompt template",
        ],
        "answer": "B",
        "explanation": "A Document has two attributes: page_content (the text content) and metadata (a dictionary with source, page number, etc.).",
    },
    {
        "q": "What is the recommended default text splitter in LangChain?",
        "options": [
            "A) CharacterTextSplitter",
            "B) TokenTextSplitter",
            "C) RecursiveCharacterTextSplitter",
            "D) SentenceTextSplitter",
        ],
        "answer": "C",
        "explanation": "RecursiveCharacterTextSplitter is the industry standard — it recursively splits by paragraphs, then lines, then words, preserving semantic coherence.",
    },
    {
        "q": "What do Document Loaders do?",
        "options": [
            "A) Train the model on documents",
            "B) Ingest raw data from various sources and convert to Document objects",
            "C) Split documents into chunks",
            "D) Embed documents as vectors",
        ],
        "answer": "B",
        "explanation": "Document Loaders read data from sources (PDFs, CSVs, websites, etc.) and produce standardized Document objects.",
    },
    {
        "q": "What do Embeddings do in LangChain?",
        "options": [
            "A) Train the LLM model",
            "B) Convert text into dense numerical vectors for semantic search",
            "C) Parse JSON output",
            "D) Store documents in a database",
        ],
        "answer": "B",
        "explanation": "Embeddings transform text into high-dimensional vectors, enabling semantic (meaning-based) similarity search.",
    },
    {
        "q": "What is a Vector Store in LangChain?",
        "options": [
            "A) A regular SQL database",
            "B) A specialized database for indexing and searching embedding vectors",
            "C) A file system for storing models",
            "D) A caching layer for API responses",
        ],
        "answer": "B",
        "explanation": "Vector Stores (ChromaDB, FAISS, Pinecone, etc.) are databases optimized for storing and searching dense vector embeddings.",
    },
    {
        "q": "What is a Retriever in LangChain?",
        "options": [
            "A) A training algorithm",
            "B) An interface for fetching relevant documents from data sources given a query",
            "C) A web scraping tool",
            "D) A model evaluation metric",
        ],
        "answer": "B",
        "explanation": "Retrievers are interfaces that fetch relevant documents from various data sources (usually vector stores) based on a query.",
    },
    {
        "q": "Which Output Parser returns raw string output?",
        "options": [
            "A) JsonOutputParser",
            "B) PydanticOutputParser",
            "C) StrOutputParser",
            "D) XMLOutputParser",
        ],
        "answer": "C",
        "explanation": "StrOutputParser is the simplest parser — it returns the LLM's output as a plain string.",
    },
    {
        "q": "What is ChatPromptTemplate used for?",
        "options": [
            "A) Training chat models",
            "B) Formatting input messages (system, human, AI) for chat models",
            "C) Parsing chat output",
            "D) Storing chat history",
        ],
        "answer": "B",
        "explanation": "ChatPromptTemplate creates structured message sequences (system, human, AI messages) with variable placeholders for chat models.",
    },
    {
        "q": "What is the purpose of chunk_overlap in text splitting?",
        "options": [
            "A) To make chunks larger",
            "B) To prevent information loss at chunk boundaries",
            "C) To duplicate the entire document",
            "D) To speed up processing",
        ],
        "answer": "B",
        "explanation": "chunk_overlap ensures that text near boundaries appears in adjacent chunks, preventing loss of context at split points.",
    },
    {
        "q": "Which vector store is recommended for LOCAL development?",
        "options": [
            "A) Pinecone",
            "B) ChromaDB or FAISS",
            "C) Weaviate Cloud",
            "D) Elasticsearch",
        ],
        "answer": "B",
        "explanation": "ChromaDB and FAISS are recommended for local development — they run locally without cloud setup. Pinecone/Qdrant are for production.",
    },
    {
        "q": "What does the @tool decorator do in LangChain?",
        "options": [
            "A) Trains the model",
            "B) Converts a Python function into a LangChain Tool that agents can call",
            "C) Parses the output",
            "D) Creates a vector store",
        ],
        "answer": "B",
        "explanation": "The @tool decorator wraps a Python function into a LangChain Tool, complete with name and description from the function's docstring.",
    },
    {
        "q": "What is an Agent in LangChain?",
        "options": [
            "A) A fixed sequence of operations",
            "B) A system that uses an LLM as a reasoning engine to decide which tools to use",
            "C) A database connection",
            "D) A prompt template",
        ],
        "answer": "B",
        "explanation": "An Agent uses the LLM to reason about what actions to take, which tools to call, and when to return the final answer — it's autonomous.",
    },
    {
        "q": "What is LangSmith used for?",
        "options": [
            "A) Training LLMs from scratch",
            "B) Tracing, debugging, evaluating, and monitoring LLM applications",
            "C) Building user interfaces",
            "D) Managing cloud infrastructure",
        ],
        "answer": "B",
        "explanation": "LangSmith is the observability platform — it provides tracing, debugging, evaluation datasets, and production monitoring.",
    },
    {
        "q": "In LangChain, which method makes a synchronous call to an LLM?",
        "options": [
            "A) llm.run()",
            "B) llm.invoke()",
            "C) llm.execute()",
            "D) llm.call()",
        ],
        "answer": "B",
        "explanation": "The .invoke() method is the standard synchronous call. Other methods: .stream() for streaming, .batch() for parallel, .ainvoke() for async.",
    },
    {
        "q": "How do you convert a vector store to a retriever?",
        "options": [
            "A) vectorstore.to_retriever()",
            "B) vectorstore.as_retriever()",
            "C) Retriever(vectorstore)",
            "D) vectorstore.get_retriever()",
        ],
        "answer": "B",
        "explanation": "Call .as_retriever() on any vector store to convert it into a LangChain Retriever with configurable search parameters.",
    },
    {
        "q": "What does MessagesPlaceholder do in a prompt template?",
        "options": [
            "A) Replaces system messages",
            "B) Creates a dynamic placeholder for a list of messages (e.g., conversation history)",
            "C) Deletes old messages",
            "D) Formats messages as JSON",
        ],
        "answer": "B",
        "explanation": "MessagesPlaceholder inserts a dynamic list of messages at that position — typically used for conversation history (chat_history).",
    },
    {
        "q": "What search type balances relevance AND diversity in results?",
        "options": [
            "A) similarity",
            "B) keyword",
            "C) MMR (Maximum Marginal Relevance)",
            "D) exact_match",
        ],
        "answer": "C",
        "explanation": "MMR (Maximum Marginal Relevance) selects results that are both relevant to the query AND diverse from each other, reducing redundancy.",
    },
    {
        "q": "Which LangChain component processes PDF files?",
        "options": [
            "A) PDFParser",
            "B) PyPDFLoader (Document Loader)",
            "C) PDFEmbedder",
            "D) PDFVectorizer",
        ],
        "answer": "B",
        "explanation": "PyPDFLoader is a Document Loader that reads PDF files and converts each page into a Document object with page_content and metadata.",
    },
    {
        "q": "What does RunnablePassthrough() do in LCEL?",
        "options": [
            "A) Skips the current step",
            "B) Passes input data through unchanged to the next step",
            "C) Terminates the chain",
            "D) Caches the result",
        ],
        "answer": "B",
        "explanation": "RunnablePassthrough passes input through without modification — essential in RAG chains to pass the question alongside retrieved context.",
    },
    {
        "q": "What is the simplest LCEL chain pattern?",
        "options": [
            "A) llm.run(prompt)",
            "B) prompt | llm | StrOutputParser()",
            "C) Chain(prompt, llm, parser)",
            "D) llm(prompt.format())",
        ],
        "answer": "B",
        "explanation": "The basic LCEL pattern: prompt | llm | StrOutputParser() — formats input, sends to model, parses output as string.",
    },
    {
        "q": "LangChain supports which programming languages?",
        "options": [
            "A) Python only",
            "B) Python and TypeScript (JavaScript)",
            "C) Python, Java, and Go",
            "D) All programming languages",
        ],
        "answer": "B",
        "explanation": "LangChain has official SDKs in Python and TypeScript/JavaScript.",
    },
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🟡 MEDIUM QUESTIONS (Architecture, LCEL, RAG Pipeline)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MEDIUM_QUESTIONS = [
    {
        "q": "What is the correct order of a RAG indexing pipeline?",
        "options": [
            "A) Embed → Load → Split → Store",
            "B) Load → Split → Embed → Store in Vector DB",
            "C) Split → Load → Store → Embed",
            "D) Store → Load → Embed → Split",
        ],
        "answer": "B",
        "explanation": "RAG indexing: Load documents → Split into chunks → Embed as vectors → Store in vector database.",
    },
    {
        "q": "How does RecursiveCharacterTextSplitter decide where to split?",
        "options": [
            "A) Randomly at any position",
            "B) Always at fixed character intervals",
            "C) Recursively tries separators in priority order: paragraphs → lines → words → characters",
            "D) Only at sentence boundaries",
        ],
        "answer": "C",
        "explanation": "It tries '\\n\\n' (paragraphs) first, then '\\n' (lines), then ' ' (words), then '' (characters) — preserving semantic units as much as possible.",
    },
    {
        "q": "What is the difference between LangChain chains and agents?",
        "options": [
            "A) Chains are newer than agents",
            "B) Chains follow a fixed sequence; Agents autonomously decide which tools to call",
            "C) Agents are simpler than chains",
            "D) There is no difference",
        ],
        "answer": "B",
        "explanation": "Chains are deterministic (fixed sequence of steps). Agents are autonomous — the LLM decides dynamically which tools to use and in what order.",
    },
    {
        "q": "Why must you use the SAME embedding model for indexing and querying?",
        "options": [
            "A) To save money",
            "B) Different models produce vectors in different dimensional spaces — they'd be incompatible",
            "C) For security reasons",
            "D) Because LangChain requires it technically",
        ],
        "answer": "B",
        "explanation": "Different embedding models produce vectors with different dimensions and semantic spaces. Comparing vectors from different models gives meaningless similarity scores.",
    },
    {
        "q": "In LCEL, what does RunnableParallel do?",
        "options": [
            "A) Runs the chain multiple times sequentially",
            "B) Runs multiple operations in parallel and combines their outputs",
            "C) Parallelizes model training",
            "D) Splits the input into parallel streams",
        ],
        "answer": "B",
        "explanation": "RunnableParallel runs multiple runnables concurrently and collects outputs as a dictionary — e.g., retrieve context AND pass question simultaneously.",
    },
    {
        "q": "What's the RAG chain pattern using LCEL?",
        "options": [
            'A) llm.invoke(retriever.get_docs(question))',
            'B) {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | parser',
            'C) RAGChain(retriever, prompt, llm)',
            'D) prompt.format(context=retriever.invoke(q)) | llm',
        ],
        "answer": "B",
        "explanation": "The standard LCEL RAG pattern: pass question to retriever AND through passthrough in parallel, then format prompt, call LLM, parse output.",
    },
    {
        "q": "What is the 'Stuff' document combination strategy?",
        "options": [
            "A) Process each document separately and merge results",
            "B) Concatenate all retrieved documents into one prompt",
            "C) Summarize documents before combining",
            "D) Score and rank each document",
        ],
        "answer": "B",
        "explanation": "Stuff = stuff all documents into a single prompt. Simplest strategy but limited by the model's context window size.",
    },
    {
        "q": "When should you use Map-Reduce instead of Stuff?",
        "options": [
            "A) When documents are very small",
            "B) When the total document content exceeds the model's context window",
            "C) When you only have one document",
            "D) For simple Q&A tasks",
        ],
        "answer": "B",
        "explanation": "Map-Reduce processes each document separately (map), then combines results (reduce). Use it when documents don't fit in a single context window.",
    },
    {
        "q": "What does PydanticOutputParser provide that JsonOutputParser doesn't?",
        "options": [
            "A) Faster parsing speed",
            "B) Type validation and schema enforcement via Pydantic models",
            "C) Support for XML output",
            "D) Multi-language support",
        ],
        "answer": "B",
        "explanation": "PydanticOutputParser validates output against a Pydantic model with type checking, required fields, and constraints — stronger than basic JSON parsing.",
    },
    {
        "q": "How does MultiQueryRetriever improve retrieval?",
        "options": [
            "A) By using multiple vector stores",
            "B) By generating multiple query variants from the original query for broader recall",
            "C) By searching multiple times with the same query",
            "D) By using multiple embedding models",
        ],
        "answer": "B",
        "explanation": "MultiQueryRetriever uses the LLM to rephrase the query into multiple variants, retrieves documents for each, and combines unique results for better recall.",
    },
    {
        "q": "What is ConversationSummaryMemory best suited for?",
        "options": [
            "A) Very short conversations (1-2 turns)",
            "B) Long conversations where storing all messages would exceed token limits",
            "C) Image-based conversations",
            "D) Single-turn Q&A",
        ],
        "answer": "B",
        "explanation": "ConversationSummaryMemory compresses old messages into a summary, keeping token usage manageable for long-running conversations.",
    },
    {
        "q": "What is the difference between langchain-google-genai and langchain-google-vertexai?",
        "options": [
            "A) They are identical packages",
            "B) google-genai uses API keys (AI Studio/free tier); vertexai uses service accounts (enterprise/GCP)",
            "C) google-genai is for Python; vertexai is for JavaScript",
            "D) google-genai is deprecated",
        ],
        "answer": "B",
        "explanation": "langchain-google-genai = Google AI Studio (API key, free tier). langchain-google-vertexai = Vertex AI on GCP (service account, enterprise features).",
    },
    {
        "q": "What is EnsembleRetriever used for?",
        "options": [
            "A) Training an ensemble of models",
            "B) Combining results from multiple retrievers (e.g., dense + sparse) for hybrid search",
            "C) Selecting the best model from an ensemble",
            "D) Averaging embedding vectors",
        ],
        "answer": "B",
        "explanation": "EnsembleRetriever combines multiple retrievers (e.g., vector-based + BM25 keyword) with configurable weights for hybrid search.",
    },
    {
        "q": "What does llm.bind_tools(tools) do?",
        "options": [
            "A) Trains the model to use tools",
            "B) Attaches tool definitions to the model so it can decide when to call them via function calling",
            "C) Disables certain model features",
            "D) Stores tools in a database",
        ],
        "answer": "B",
        "explanation": "bind_tools attaches tool schemas to the model, enabling native function calling — the LLM can decide to invoke tools based on the user's query.",
    },
    {
        "q": "Which LangChain component would you use to load an entire directory of files?",
        "options": [
            "A) FileLoader",
            "B) DirectoryLoader",
            "C) BulkLoader",
            "D) FolderLoader",
        ],
        "answer": "B",
        "explanation": "DirectoryLoader loads all files from a directory, automatically selecting the appropriate loader based on file extensions.",
    },
    {
        "q": "What does the 'k' parameter in search_kwargs={'k': 4} control?",
        "options": [
            "A) The number of vector dimensions",
            "B) The number of top results to return from the retriever",
            "C) The chunk size",
            "D) The embedding batch size",
        ],
        "answer": "B",
        "explanation": "k controls how many top documents the retriever returns. k=4 means the 4 most relevant documents are retrieved.",
    },
    {
        "q": "What is the purpose of parser.get_format_instructions()?",
        "options": [
            "A) To format the parser's internal code",
            "B) To generate text instructions that tell the LLM how to format its output correctly",
            "C) To retrieve the parser's version",
            "D) To validate existing output",
        ],
        "answer": "B",
        "explanation": "get_format_instructions() returns human-readable instructions for the LLM explaining the expected output format (JSON schema, field descriptions, etc.).",
    },
    {
        "q": "The ReAct pattern used by LangChain agents follows which loop?",
        "options": [
            "A) Input → Output → Done",
            "B) Thought → Action → Observation → (repeat until answer ready)",
            "C) Plan → Execute → Verify → Done",
            "D) Query → Retrieve → Generate → Done",
        ],
        "answer": "B",
        "explanation": "ReAct = Reasoning + Acting. The agent reasons (Thought), acts (calls a tool), observes results (Observation), and loops until it has enough info.",
    },
    {
        "q": "What is langchain-core?",
        "options": [
            "A) The main package with all integrations",
            "B) The base package with interfaces, LCEL, prompt templates, and output parsers",
            "C) A deprecated package",
            "D) The LangGraph runtime",
        ],
        "answer": "B",
        "explanation": "langchain-core contains the foundational abstractions: Runnable interface, LCEL, ChatPromptTemplate, output parsers, and base types.",
    },
    {
        "q": "In LCEL, every component is a 'Runnable'. What methods does every Runnable have?",
        "options": [
            "A) run(), execute(), process()",
            "B) invoke(), stream(), batch(), ainvoke()",
            "C) start(), stop(), restart()",
            "D) call(), respond(), generate()",
        ],
        "answer": "B",
        "explanation": "Every Runnable has invoke() (sync), stream() (streaming), batch() (parallel), and ainvoke() (async) — a unified interface.",
    },
    {
        "q": "What does the 'temperature' parameter control when passed to an LLM wrapper?",
        "options": [
            "A) The model's training speed",
            "B) The randomness/creativity of the output (0 = deterministic, higher = creative)",
            "C) The number of output tokens",
            "D) The hardware temperature",
        ],
        "answer": "B",
        "explanation": "temperature=0 gives deterministic output (always picks most likely token). Higher values add randomness for creative tasks.",
    },
    {
        "q": "What is SemanticChunker?",
        "options": [
            "A) A text splitter that uses fixed character counts",
            "B) A text splitter that uses embedding similarity to find natural breakpoints",
            "C) A tool for semantic analysis",
            "D) A database query optimizer",
        ],
        "answer": "B",
        "explanation": "SemanticChunker splits text based on embedding similarity between sentences — when similarity drops significantly, it creates a split boundary.",
    },
    {
        "q": "How do you enable LangSmith tracing?",
        "options": [
            "A) Install a special package",
            "B) Set environment variables: LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY",
            "C) Add a decorator to every function",
            "D) Modify the model configuration",
        ],
        "answer": "B",
        "explanation": "LangSmith tracing is enabled via environment variables: LANGCHAIN_TRACING_V2=true, LANGCHAIN_API_KEY=your_key, and optionally LANGCHAIN_PROJECT.",
    },
    {
        "q": "What does OutputFixingParser do?",
        "options": [
            "A) Prevents all parsing errors",
            "B) Wraps another parser and uses an LLM to automatically fix malformed output on failure",
            "C) Converts output to a fixed format",
            "D) Stores output in a fixed location",
        ],
        "answer": "B",
        "explanation": "OutputFixingParser wraps a parser (e.g., PydanticOutputParser) and if parsing fails, it sends the output + error back to the LLM to fix.",
    },
    {
        "q": "Why is LangChain's model interface called 'unified'?",
        "options": [
            "A) Because it only supports one model",
            "B) Because the same code works with different LLM providers — just swap the import",
            "C) Because it merges all models into one",
            "D) Because it uses a unified cloud provider",
        ],
        "answer": "B",
        "explanation": "LangChain provides a common interface: ChatOpenAI, ChatGoogleGenerativeAI, ChatAnthropic all share the same invoke/stream/batch methods. Swap one line to change providers.",
    },
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🔴 HARD QUESTIONS (LangGraph, Production, Traps, Scenarios)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HARD_QUESTIONS = [
    {
        "q": "What is the KEY difference between LangChain (Chains) and LangGraph?",
        "options": [
            "A) LangGraph is a simplified version of LangChain",
            "B) LangChain supports linear flows; LangGraph supports loops, branches, state, and human-in-the-loop",
            "C) LangGraph replaces LangChain entirely",
            "D) LangChain supports agents; LangGraph does not",
        ],
        "answer": "B",
        "explanation": "LangChain = linear chains (pipe operator). LangGraph = graph-based with cycles, conditional edges, persistent state, and human-in-the-loop. They complement each other.",
    },
    {
        "q": "In LangGraph, what are 'Nodes' and 'Edges'?",
        "options": [
            "A) Nodes are models; Edges are prompts",
            "B) Nodes are functions performing work; Edges define transitions between nodes (fixed or conditional)",
            "C) Nodes are databases; Edges are queries",
            "D) Nodes are users; Edges are messages",
        ],
        "answer": "B",
        "explanation": "Nodes = Python functions that do work (LLM calls, tool execution). Edges = transitions that wire nodes together (fixed sequence or conditional routing).",
    },
    {
        "q": "What is 'Checkpointing' in LangGraph?",
        "options": [
            "A) Saving model weights during training",
            "B) Taking snapshots of graph state for persistence, crash recovery, and time-travel debugging",
            "C) Checking code for errors",
            "D) Verifying API responses",
        ],
        "answer": "B",
        "explanation": "Checkpointing saves state snapshots at each step. This enables: crash recovery (resume from last good state), time-travel (replay past states), and persistence across sessions.",
    },
    {
        "q": "How does LangGraph implement human-in-the-loop?",
        "options": [
            "A) By sending emails to the user",
            "B) Using 'interrupts' that pause execution, wait for human feedback, then resume from the same point",
            "C) By requiring the user to restart the program",
            "D) Through a polling mechanism",
        ],
        "answer": "B",
        "explanation": "LangGraph uses interrupts to pause the graph mid-execution. The human can approve, edit, or reject. Execution resumes exactly where it stopped, thanks to checkpointing.",
    },
    {
        "q": "A developer uses LLMChain in their LangChain application. What's the recommended modern approach?",
        "options": [
            "A) LLMChain is the current best practice",
            "B) Replace with LCEL: prompt | llm | StrOutputParser()",
            "C) Use SequentialChain instead",
            "D) Use AgentExecutor",
        ],
        "answer": "B",
        "explanation": "LLMChain is legacy/deprecated. The modern equivalent is LCEL: prompt | llm | parser. This provides streaming, batching, and async automatically.",
    },
    {
        "q": "You have 500 documents that don't fit in a single prompt. Which strategy should you use?",
        "options": [
            "A) Stuff — put them all in one prompt",
            "B) Map-Reduce — process each separately, then combine results",
            "C) Ignore the extra documents",
            "D) Increase the temperature",
        ],
        "answer": "B",
        "explanation": "Map-Reduce processes each document with the LLM individually (map), then combines all answers into a final response (reduce). Handles any number of documents.",
    },
    {
        "q": "A RAG system retrieves 4 very similar paragraphs that all say the same thing. What feature would help?",
        "options": [
            "A) Increasing the chunk_size",
            "B) Using MMR (Maximum Marginal Relevance) search instead of standard similarity",
            "C) Switching to a larger model",
            "D) Adding more documents to the vector store",
        ],
        "answer": "B",
        "explanation": "MMR balances relevance and diversity — it selects results that are both relevant to the query AND different from each other, reducing redundancy.",
    },
    {
        "q": "Which statement about LangSmith is FALSE?",
        "options": [
            "A) LangSmith is required to run LangChain applications",
            "B) LangSmith provides tracing and debugging",
            "C) LangSmith supports evaluation datasets",
            "D) LangSmith can monitor production applications",
        ],
        "answer": "A",
        "explanation": "TRAP: LangSmith is OPTIONAL. LangChain works perfectly without it. LangSmith adds observability but is not a runtime dependency.",
    },
    {
        "q": "A developer uses different embedding models for indexing (model A) and querying (model B). What happens?",
        "options": [
            "A) Better search results due to diversity",
            "B) Similarity search returns meaningless results because vectors are in incompatible spaces",
            "C) No effect — all embedding models produce the same vectors",
            "D) Faster query performance",
        ],
        "answer": "B",
        "explanation": "Different embedding models produce vectors in different dimensional spaces with different semantic mappings. Comparing them gives random/meaningless similarity scores.",
    },
    {
        "q": "In LangGraph, what are 'Conditional Edges'?",
        "options": [
            "A) Edges that only work on certain days",
            "B) Logic-based transitions that route to different nodes based on the current state",
            "C) Edges that require user permission",
            "D) Edges between nodes on different servers",
        ],
        "answer": "B",
        "explanation": "Conditional edges are if/else routing — they evaluate the current state and decide which node to transition to next. Essential for branching logic in agents.",
    },
    {
        "q": "What is the main advantage of LangChain's unified model interface for production?",
        "options": [
            "A) All models are the same quality",
            "B) You can switch LLM providers (OpenAI → Google → Anthropic) by changing one import line",
            "C) It makes all models free",
            "D) It guarantees the same output from all models",
        ],
        "answer": "B",
        "explanation": "The unified interface (invoke, stream, batch) means switching providers requires changing only the import and model name — no rewriting of chain/agent logic.",
    },
    {
        "q": "An agent enters an infinite loop, calling the same tool repeatedly. What's the BEST prevention?",
        "options": [
            "A) Use a larger model",
            "B) Set max_iterations on the agent to limit tool-calling cycles",
            "C) Remove all tools",
            "D) Set temperature to 0",
        ],
        "answer": "B",
        "explanation": "Always set max_iterations (or max_steps in LangGraph) to prevent infinite loops. This is a critical production safety measure for autonomous agents.",
    },
    {
        "q": "What is 'Hybrid Search' in the context of LangChain retrievers?",
        "options": [
            "A) Searching in two databases simultaneously",
            "B) Combining dense vector search (semantic) with sparse keyword search (BM25) via EnsembleRetriever",
            "C) Using two different LLMs for search",
            "D) Searching documents and images together",
        ],
        "answer": "B",
        "explanation": "Hybrid search combines dense retrieval (embedding similarity — good for meaning) with sparse retrieval (BM25 keywords — good for exact matches) for better overall results.",
    },
    {
        "q": "When building a production RAG system, what's the correct order of optimization priorities?",
        "options": [
            "A) Model → Embeddings → Retrieval → Prompting",
            "B) Retrieval quality → Chunking strategy → Prompt engineering → Model selection",
            "C) Prompt → Model → Fine-tuning → Deployment",
            "D) All optimizations should happen simultaneously",
        ],
        "answer": "B",
        "explanation": "Retrieval quality matters most (garbage in, garbage out). Then chunking (good chunks = good retrieval). Then prompt engineering. Then model choice.",
    },
    {
        "q": "What is SelfQueryRetriever?",
        "options": [
            "A) A retriever that queries itself recursively",
            "B) A retriever that uses an LLM to convert natural language into structured metadata filters",
            "C) A retriever that only searches its own documentation",
            "D) A retriever that generates its own queries randomly",
        ],
        "answer": "B",
        "explanation": "SelfQueryRetriever uses the LLM to parse the user's natural language query into structured filters (e.g., 'papers from 2024' → year=2024 filter + semantic search).",
    },
    {
        "q": "What is the relationship between LangGraph and create_react_agent()?",
        "options": [
            "A) They are unrelated",
            "B) create_react_agent() is a pre-built LangGraph graph that implements the ReAct pattern",
            "C) create_react_agent() is deprecated in favor of LangGraph",
            "D) LangGraph doesn't support agents",
        ],
        "answer": "B",
        "explanation": "create_react_agent() is a convenience function from LangGraph (langgraph.prebuilt) that creates a pre-configured ReAct agent as a LangGraph graph.",
    },
    {
        "q": "A developer wants to persist conversation history across server restarts. What's the BEST approach?",
        "options": [
            "A) ConversationBufferMemory (in-memory)",
            "B) LangGraph checkpointing with persistent storage (database-backed)",
            "C) Saving prompts to a text file",
            "D) Increasing the context window",
        ],
        "answer": "B",
        "explanation": "In-memory solutions are lost on restart. LangGraph checkpointing with a database backend (PostgreSQL, Redis) persists state across restarts and crashes.",
    },
    {
        "q": "What is the advantage of RunnableLambda in LCEL?",
        "options": [
            "A) It runs Lambda functions on AWS",
            "B) It wraps any Python function into an LCEL-compatible Runnable for custom transformations",
            "C) It provides serverless execution",
            "D) It handles authentication",
        ],
        "answer": "B",
        "explanation": "RunnableLambda converts any Python function into a Runnable, letting you insert custom logic (transformations, filtering, formatting) into LCEL chains.",
    },
    {
        "q": "Why does LangChain separate langchain-core from langchain-community?",
        "options": [
            "A) For licensing reasons",
            "B) To separate stable core interfaces from third-party integrations that may change independently",
            "C) To make the framework slower",
            "D) Because they use different programming languages",
        ],
        "answer": "B",
        "explanation": "langchain-core = stable interfaces and LCEL (rarely changes). langchain-community = third-party integrations that evolve with external APIs (changes frequently).",
    },
    {
        "q": "In a production RAG system, the LLM sometimes ignores the 'Answer only from context' instruction and hallucinates. What's the BEST multi-layered defense?",
        "options": [
            "A) Just use a larger model",
            "B) Combine: grounding instruction + fallback instruction + temperature=0 + output validation + citations",
            "C) Remove the system prompt entirely",
            "D) Use few-shot examples only",
        ],
        "answer": "B",
        "explanation": "Production defense is multi-layered: grounding rules + 'say I don't know' fallback + temperature=0 + post-processing validation + require citations from context.",
    },
    {
        "q": "What does Vertex AI Search provide that manual LangChain RAG doesn't?",
        "options": [
            "A) Nothing — they are identical",
            "B) A fully managed RAG service: automatic chunking, embedding, indexing, and grounding without custom code",
            "C) A cheaper LLM model",
            "D) Faster Python execution",
        ],
        "answer": "B",
        "explanation": "Vertex AI Search is a managed RAG solution — Google handles chunking, embedding, indexing, and grounding. No manual pipeline needed. LangChain gives you full control but requires building everything.",
    },
    {
        "q": "A LangGraph agent needs to send an email, but only after the user confirms. Which feature handles this?",
        "options": [
            "A) Conditional edge",
            "B) Human-in-the-loop interrupt — pause before the email tool, wait for approval",
            "C) Setting temperature to 0",
            "D) Using a fallback prompt",
        ],
        "answer": "B",
        "explanation": "LangGraph's human-in-the-loop uses interrupts to pause execution before sensitive operations. The human approves or rejects, and execution resumes from the checkpoint.",
    },
    {
        "q": "What is the ParentDocumentRetriever?",
        "options": [
            "A) A retriever that only searches parent directories",
            "B) A retriever that stores small chunks for search but returns the FULL parent document for context",
            "C) A retriever for parenting advice documents",
            "D) A retriever that inherits from a parent class",
        ],
        "answer": "B",
        "explanation": "ParentDocumentRetriever stores small chunks (for precise matching) but returns the larger parent document (for complete context) — best of both worlds.",
    },
    {
        "q": "When should you use LangGraph instead of simple LCEL chains?",
        "options": [
            "A) Always — LangGraph is always better",
            "B) When your workflow needs loops, conditional branching, persistent state, or human approval steps",
            "C) Only for simple Q&A",
            "D) Never — LCEL chains are always sufficient",
        ],
        "answer": "B",
        "explanation": "Use LCEL chains for simple, linear workflows (RAG, Q&A). Switch to LangGraph when you need cycles (retry on error), branching, persistent state, or human-in-the-loop.",
    },
    {
        "q": "A team uses LangChain for RAG, LangGraph for agents, and LangSmith for monitoring. Which statement is correct?",
        "options": [
            "A) Using all three is redundant",
            "B) This is the recommended production architecture — each serves a different purpose in the stack",
            "C) LangGraph makes LangChain unnecessary",
            "D) LangSmith makes LangGraph unnecessary",
        ],
        "answer": "B",
        "explanation": "This is the ideal setup: LangChain = core primitives and RAG, LangGraph = complex agent orchestration, LangSmith = observability across both. They are complementary layers.",
    },
]


def run_quiz(
    questions: list,
    difficulty: str,
    num_questions: int = 10,
    emoji: str = "🟢",
) -> tuple:
    """Run a quiz from the given question pool. Returns (score, total)."""
    selected = random.sample(questions, min(num_questions, len(questions)))
    score = 0
    total = len(selected)

    print(f"\n{'=' * 60}")
    print(f"{emoji} LANGCHAIN QUIZ — {difficulty.upper()} DIFFICULTY")
    print(f"   {total} questions | GenAI L2 Exam Prep")
    print(f"{'=' * 60}")

    for i, q in enumerate(selected, 1):
        print(f"\n{'─' * 50}")
        print(f"Q{i}/{total}: {q['q']}\n")
        for opt in q["options"]:
            print(f"   {opt}")

        while True:
            answer = input(f"\nYour answer (A/B/C/D): ").strip().upper()
            if answer in ("A", "B", "C", "D"):
                break
            print("   ⚠️  Please enter A, B, C, or D")

        if answer == q["answer"]:
            print(f"   ✅ Correct!")
            score += 1
        else:
            print(f"   ❌ Wrong! Correct answer: {q['answer']}")

        print(f"   📝 {q['explanation']}")

    return score, total


def main():
    print("\n" + "=" * 60)
    print("🧠 LANGCHAIN FRAMEWORK — PRACTICE QUIZ")
    print("=" * 60)
    total_q = len(EASY_QUESTIONS) + len(MEDIUM_QUESTIONS) + len(HARD_QUESTIONS)
    print(f"\nTotal questions available: {total_q}")
    print(f"  🟢 Easy:   {len(EASY_QUESTIONS)} questions")
    print(f"  🟡 Medium: {len(MEDIUM_QUESTIONS)} questions")
    print(f"  🔴 Hard:   {len(HARD_QUESTIONS)} questions")

    print("\nSelect quiz mode:")
    print("  1) 🟢 Easy only")
    print("  2) 🟡 Medium only")
    print("  3) 🔴 Hard only")
    print("  4) 🌈 Mixed (all difficulties)")
    print("  5) 🏆 Full exam simulation (all 75 questions)")

    while True:
        mode = input("\nYour choice (1-5): ").strip()
        if mode in ("1", "2", "3", "4", "5"):
            break
        print("   ⚠️  Please enter 1, 2, 3, 4, or 5")

    total_score = 0
    total_questions = 0

    if mode == "1":
        try:
            n = input("How many questions? (press Enter for 10): ").strip()
            num = int(n) if n else 10
        except ValueError:
            num = 10
        s, t = run_quiz(EASY_QUESTIONS, "Easy", num, "🟢")
        total_score += s
        total_questions += t

    elif mode == "2":
        try:
            n = input("How many questions? (press Enter for 10): ").strip()
            num = int(n) if n else 10
        except ValueError:
            num = 10
        s, t = run_quiz(MEDIUM_QUESTIONS, "Medium", num, "🟡")
        total_score += s
        total_questions += t

    elif mode == "3":
        try:
            n = input("How many questions? (press Enter for 10): ").strip()
            num = int(n) if n else 10
        except ValueError:
            num = 10
        s, t = run_quiz(HARD_QUESTIONS, "Hard", num, "🔴")
        total_score += s
        total_questions += t

    elif mode == "4":
        try:
            n = input("How many per difficulty? (press Enter for 5): ").strip()
            num = int(n) if n else 5
        except ValueError:
            num = 5
        for pool, diff, em in [
            (EASY_QUESTIONS, "Easy", "🟢"),
            (MEDIUM_QUESTIONS, "Medium", "🟡"),
            (HARD_QUESTIONS, "Hard", "🔴"),
        ]:
            s, t = run_quiz(pool, diff, num, em)
            total_score += s
            total_questions += t

    elif mode == "5":
        print("\n🏆 FULL EXAM SIMULATION — All 75 questions!")
        for pool, diff, em in [
            (EASY_QUESTIONS, "Easy", "🟢"),
            (MEDIUM_QUESTIONS, "Medium", "🟡"),
            (HARD_QUESTIONS, "Hard", "🔴"),
        ]:
            s, t = run_quiz(pool, diff, len(pool), em)
            total_score += s
            total_questions += t

    # Final Results
    pct = total_score / total_questions * 100 if total_questions > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"📊 FINAL SCORE: {total_score}/{total_questions} ({pct:.0f}%)")
    print(f"{'=' * 60}")

    if pct >= 90:
        print("🏆 Outstanding! You've mastered LangChain for the L2 exam!")
    elif pct >= 80:
        print("🎉 Excellent! Solid understanding. Review the few you missed.")
    elif pct >= 60:
        print("👍 Good progress! Re-read sections on topics you missed.")
    else:
        print("📚 Keep studying! Re-read 01_LangChain_Study_Guide.md and try again.")

    print("\nTip: Run again with different difficulty levels to test all areas!")


if __name__ == "__main__":
    main()
