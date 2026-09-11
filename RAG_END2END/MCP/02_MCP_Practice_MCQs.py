# 🧠 Model Context Protocol (MCP) — Practice MCQs (GenAI L2 Exam)
# =================================================================
# 75 Multiple Choice Questions in 3 Difficulty Levels:
#   🟢 EASY (25 questions)    — Definitions, basic concepts
#   🟡 MEDIUM (25 questions)  — Architecture, comparisons, workflows
#   🔴 HARD (25 questions)    — Security, scenarios, edge cases, traps
#
# Run this script to take an interactive self-assessment quiz.

import random
import sys

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🟢 EASY QUESTIONS (Definitions & Basic Concepts)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EASY_QUESTIONS = [
    {
        "q": "What does MCP stand for?",
        "options": [
            "A) Model Context Protocol",
            "B) Machine Communication Platform",
            "C) Multi-Channel Processor",
            "D) Model Configuration Pipeline",
        ],
        "answer": "A",
        "explanation": "MCP stands for Model Context Protocol — an open standard for connecting AI models to external tools and data sources.",
    },
    {
        "q": "Who originally created MCP?",
        "options": [
            "A) OpenAI",
            "B) Google",
            "C) Anthropic",
            "D) Meta",
        ],
        "answer": "C",
        "explanation": "MCP was created by Anthropic and later open-sourced. It is now governed by the Linux Foundation.",
    },
    {
        "q": "What messaging protocol does MCP use for communication?",
        "options": [
            "A) REST API",
            "B) GraphQL",
            "C) JSON-RPC 2.0",
            "D) gRPC",
        ],
        "answer": "C",
        "explanation": "MCP uses JSON-RPC 2.0 for all communication, enabling stateful, bidirectional messaging between clients and servers.",
    },
    {
        "q": "What are the THREE server-side primitives in MCP?",
        "options": [
            "A) Models, Embeddings, Vectors",
            "B) Tools, Resources, Prompts",
            "C) Chains, Agents, Memory",
            "D) Queries, Mutations, Subscriptions",
        ],
        "answer": "B",
        "explanation": "MCP servers expose three primitives: Tools (executable functions), Resources (read-only data), and Prompts (templated workflows).",
    },
    {
        "q": "What is an MCP 'Host'?",
        "options": [
            "A) The cloud server running the LLM",
            "B) The AI application the user interacts with (e.g., Claude Desktop, Cursor)",
            "C) The database storing tool configurations",
            "D) The network router connecting clients",
        ],
        "answer": "B",
        "explanation": "The Host is the AI application (like Claude Desktop, Cursor, or a custom app) that contains MCP clients and manages user interactions.",
    },
    {
        "q": "What is an MCP 'Client'?",
        "options": [
            "A) The end user of the AI application",
            "B) A component within the Host that maintains a 1:1 connection to a server",
            "C) The web browser used to access the AI",
            "D) A library for building user interfaces",
        ],
        "answer": "B",
        "explanation": "An MCP Client is a protocol-level component within the Host that maintains a direct, stateful, 1:1 connection to exactly one MCP Server.",
    },
    {
        "q": "What is an MCP 'Server'?",
        "options": [
            "A) A physical machine in a data center",
            "B) A program that exposes tools, resources, and prompts to MCP clients",
            "C) The LLM model itself",
            "D) A database management system",
        ],
        "answer": "B",
        "explanation": "An MCP Server is a lightweight program that exposes specific capabilities (tools, resources, prompts) to connected MCP clients.",
    },
    {
        "q": "What type of architecture does MCP use?",
        "options": [
            "A) Peer-to-peer",
            "B) Client-Server",
            "C) Microservices mesh",
            "D) Monolithic",
        ],
        "answer": "B",
        "explanation": "MCP uses a client-server architecture where clients (in hosts) connect to servers that expose tools, resources, and prompts.",
    },
    {
        "q": "In MCP, what are 'Tools'?",
        "options": [
            "A) Read-only data files",
            "B) User interface elements",
            "C) Executable functions that the AI model can invoke to perform actions",
            "D) System configuration settings",
        ],
        "answer": "C",
        "explanation": "Tools are model-controlled executable functions that can perform actions (query databases, send messages, create files) and may have side effects.",
    },
    {
        "q": "In MCP, what are 'Resources'?",
        "options": [
            "A) Executable functions with side effects",
            "B) Read-only data and context exposed by the server",
            "C) LLM model weights",
            "D) Network bandwidth allocations",
        ],
        "answer": "B",
        "explanation": "Resources are application-controlled, read-only data (files, database records, documents) that servers expose for the AI to consume as context.",
    },
    {
        "q": "In MCP, what are 'Prompts' (the primitive)?",
        "options": [
            "A) The same as prompt engineering techniques (zero-shot, few-shot)",
            "B) Pre-defined prompt templates and interaction workflows exposed by the server",
            "C) Random text generation",
            "D) Model training instructions",
        ],
        "answer": "B",
        "explanation": "In MCP, 'Prompts' are server-defined templated messages or workflows — NOT the same as prompting techniques like zero-shot or few-shot.",
    },
    {
        "q": "What is the relationship between an MCP Client and an MCP Server?",
        "options": [
            "A) One client connects to many servers simultaneously",
            "B) One client maintains a 1:1 connection to exactly one server",
            "C) Many clients share a single connection to one server",
            "D) Clients and servers are the same component",
        ],
        "answer": "B",
        "explanation": "Each MCP Client maintains a direct, stateful, 1:1 connection to exactly one MCP Server. A Host can have multiple clients for multiple servers.",
    },
    {
        "q": "What transport does MCP use for LOCAL communication?",
        "options": [
            "A) HTTP/SSE",
            "B) WebSocket",
            "C) STDIO (Standard Input/Output)",
            "D) FTP",
        ],
        "answer": "C",
        "explanation": "STDIO (stdin/stdout) is used for local communication where the server runs as a child process on the same machine as the client.",
    },
    {
        "q": "What transport does MCP use for REMOTE communication?",
        "options": [
            "A) STDIO",
            "B) Streamable HTTP with SSE (Server-Sent Events)",
            "C) SMTP",
            "D) Raw TCP sockets",
        ],
        "answer": "B",
        "explanation": "Streamable HTTP with SSE is used for remote communication, enabling scalable connections between distributed clients and servers.",
    },
    {
        "q": "MCP is best described as:",
        "options": [
            "A) A proprietary API by Anthropic",
            "B) An open standard protocol for AI-to-tool connectivity",
            "C) A machine learning framework",
            "D) A database query language",
        ],
        "answer": "B",
        "explanation": "MCP is an open standard protocol (now governed by Linux Foundation) that standardizes how AI applications connect to external tools and data.",
    },
    {
        "q": "What problem does MCP solve?",
        "options": [
            "A) The vanishing gradient problem",
            "B) The N×M integration problem (every AI app needing custom connectors for every tool)",
            "C) The cold start problem in recommendation systems",
            "D) The class imbalance problem in training data",
        ],
        "answer": "B",
        "explanation": "MCP reduces the N×M integration complexity (N models × M tools = N×M custom connectors) to N+M by providing a universal protocol.",
    },
    {
        "q": "Which of these is an example of an MCP Host?",
        "options": [
            "A) PostgreSQL database",
            "B) Claude Desktop or Cursor IDE",
            "C) GitHub API",
            "D) An MCP Server",
        ],
        "answer": "B",
        "explanation": "Hosts are AI applications like Claude Desktop, Cursor, Windsurf, or VS Code with Copilot — they contain MCP clients and face the user.",
    },
    {
        "q": "Which of these is an example of an MCP Server?",
        "options": [
            "A) Claude Desktop",
            "B) A user's web browser",
            "C) A filesystem server that reads/writes local files",
            "D) The LLM model itself",
        ],
        "answer": "C",
        "explanation": "MCP Servers are programs that expose capabilities — e.g., a filesystem server, GitHub server, database server, Slack server.",
    },
    {
        "q": "MCP communication sessions are:",
        "options": [
            "A) Stateless (like REST APIs)",
            "B) Stateful (context maintained across requests)",
            "C) One-way only (client to server)",
            "D) Broadcast-based (one to many)",
        ],
        "answer": "B",
        "explanation": "MCP sessions are stateful — context and state are maintained across requests during the connection lifecycle.",
    },
    {
        "q": "Who controls which 'Tools' to call in MCP?",
        "options": [
            "A) The user manually",
            "B) The AI model decides autonomously",
            "C) The server administrator",
            "D) A random selection algorithm",
        ],
        "answer": "B",
        "explanation": "Tools are model-controlled — the AI model autonomously decides when to call tools based on their names and descriptions.",
    },
    {
        "q": "Who controls which 'Resources' to include in context?",
        "options": [
            "A) The AI model",
            "B) The application / host",
            "C) The end user only",
            "D) The MCP server exclusively",
        ],
        "answer": "B",
        "explanation": "Resources are application-controlled — the host application decides which resources to include in the context.",
    },
    {
        "q": "Who controls when 'Prompts' (MCP primitive) are used?",
        "options": [
            "A) The AI model",
            "B) The application automatically",
            "C) The user (e.g., via slash commands or menus)",
            "D) The server's scheduler",
        ],
        "answer": "C",
        "explanation": "Prompts are user-controlled — they appear as slash commands or menu items that the user selects to trigger predefined workflows.",
    },
    {
        "q": "Can an MCP Resource modify data?",
        "options": [
            "A) Yes, resources can read and write",
            "B) No — resources are read-only with no side effects",
            "C) Yes, but only with admin permissions",
            "D) Only if the server allows it",
        ],
        "answer": "B",
        "explanation": "Resources are strictly read-only. Only Tools can have side effects (modify data, send messages, etc.).",
    },
    {
        "q": "What is the common analogy used to describe MCP?",
        "options": [
            "A) MCP is the 'Google Search of AI'",
            "B) MCP is the 'USB-C for AI' — a universal connector",
            "C) MCP is the 'blockchain of AI'",
            "D) MCP is the 'internet protocol of AI'",
        ],
        "answer": "B",
        "explanation": "MCP is commonly called 'USB-C for AI' — just as USB-C standardizes device connections, MCP standardizes AI-to-tool connections.",
    },
    {
        "q": "How many MCP Clients can a single Host contain?",
        "options": [
            "A) Exactly 1",
            "B) Exactly 2",
            "C) Multiple (one per connected server)",
            "D) None — hosts don't contain clients",
        ],
        "answer": "C",
        "explanation": "A Host can contain multiple MCP Clients — typically one client for each MCP Server it needs to connect to.",
    },
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🟡 MEDIUM QUESTIONS (Architecture, Comparisons, Workflows)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MEDIUM_QUESTIONS = [
    {
        "q": "During MCP initialization, what happens in the 'capability negotiation' phase?",
        "options": [
            "A) The client sends its API key to the server",
            "B) Both client and server exchange what features and protocol versions they support",
            "C) The server downloads the LLM model",
            "D) The user selects which tools to enable",
        ],
        "answer": "B",
        "explanation": "During initialization, client and server exchange capabilities (supported features, protocol version, server/client info) in a handshake.",
    },
    {
        "q": "What is the correct order of the MCP connection lifecycle?",
        "options": [
            "A) Connect → Use Tools → Negotiate → Disconnect",
            "B) Initialize (negotiate capabilities) → Active Session → Shutdown",
            "C) Authenticate → Discover → Execute → Log",
            "D) Handshake → Stream → Buffer → Close",
        ],
        "answer": "B",
        "explanation": "The lifecycle is: Initialize (capability negotiation) → Active Session (tool calls, resource reads) → Shutdown (graceful termination).",
    },
    {
        "q": "How does MCP reduce integration complexity from N×M to N+M?",
        "options": [
            "A) By using a single universal API key",
            "B) By providing a standard protocol so tools are built once and work with any MCP client",
            "C) By caching all responses",
            "D) By limiting the number of supported tools",
        ],
        "answer": "B",
        "explanation": "Build a tool as an MCP Server once → it works with any MCP-compatible client. Build a client once → it connects to any MCP server. This reduces N×M custom integrations to N+M standardized ones.",
    },
    {
        "q": "What makes MCP different from traditional REST APIs?",
        "options": [
            "A) MCP is faster",
            "B) MCP uses JSON-RPC 2.0 with stateful, bidirectional sessions (not stateless)",
            "C) MCP uses XML instead of JSON",
            "D) MCP doesn't support authentication",
        ],
        "answer": "B",
        "explanation": "MCP uses JSON-RPC 2.0 which is stateful and bidirectional, unlike REST which is stateless and primarily request-response.",
    },
    {
        "q": "In the MCP architecture, the Host is responsible for:",
        "options": [
            "A) Only running the LLM inference",
            "B) Managing multiple client connections, enforcing security, and handling user consent",
            "C) Only storing data",
            "D) Only displaying the user interface",
        ],
        "answer": "B",
        "explanation": "The Host manages client connections, enforces security policies, routes LLM interactions, and controls user consent for tool execution.",
    },
    {
        "q": "How does an LLM decide which MCP Tool to call?",
        "options": [
            "A) Random selection from available tools",
            "B) Based on the tool's name and description — the model matches them to the user's request",
            "C) The user must manually specify which tool to use",
            "D) Tools are called in alphabetical order",
        ],
        "answer": "B",
        "explanation": "The LLM uses the tool's name and description to understand what each tool does, then autonomously decides which tool(s) to call based on the user's request.",
    },
    {
        "q": "What is the key difference between MCP Tools and MCP Resources?",
        "options": [
            "A) Tools are free, Resources cost money",
            "B) Tools can have side effects and are model-controlled; Resources are read-only and application-controlled",
            "C) Tools are for text, Resources are for images",
            "D) No difference — they are interchangeable",
        ],
        "answer": "B",
        "explanation": "Tools = executable, model-controlled, can have side effects. Resources = read-only, application-controlled, no side effects.",
    },
    {
        "q": "When should you use STDIO transport vs HTTP/SSE transport?",
        "options": [
            "A) STDIO for everything, HTTP/SSE is deprecated",
            "B) STDIO for local servers on the same machine; HTTP/SSE for remote/cloud servers",
            "C) HTTP/SSE for local, STDIO for remote",
            "D) They are interchangeable with no functional difference",
        ],
        "answer": "B",
        "explanation": "STDIO is for local process communication (fast, no auth needed). HTTP/SSE is for remote/distributed servers (requires OAuth 2.1).",
    },
    {
        "q": "How does MCP compare to Function Calling in terms of model lock-in?",
        "options": [
            "A) Both have high lock-in",
            "B) MCP has no vendor lock-in (open standard); Function Calling is provider-specific (high lock-in)",
            "C) Function Calling has no lock-in; MCP is locked to Anthropic",
            "D) Neither has any lock-in",
        ],
        "answer": "B",
        "explanation": "MCP is vendor-neutral (open standard). Function Calling is provider-specific (OpenAI's format differs from Google's, etc.).",
    },
    {
        "q": "In MCP, what are 'Roots'?",
        "options": [
            "A) The root directory of the MCP installation",
            "B) Client-defined filesystem paths or URIs that set boundaries for server operations",
            "C) The root cause analysis tool",
            "D) The initial prompt sent to the LLM",
        ],
        "answer": "B",
        "explanation": "Roots are entry points (paths/URIs) shared by the client that define where the server can operate — a security boundary mechanism.",
    },
    {
        "q": "What happened to 'Sampling' in MCP?",
        "options": [
            "A) It was enhanced with new features",
            "B) It was deprecated as of the July 2026 protocol update",
            "C) It became the most important feature",
            "D) It was renamed to 'Inference'",
        ],
        "answer": "B",
        "explanation": "Sampling (which allowed servers to request LLM completions from the Host) was deprecated. Implementations now integrate directly with LLM APIs.",
    },
    {
        "q": "MCP 'Prompts' differ from prompt engineering techniques because:",
        "options": [
            "A) They are the same concept",
            "B) MCP Prompts are server-defined templates; prompt engineering is about designing effective inputs for LLMs",
            "C) MCP Prompts are always zero-shot",
            "D) Prompt engineering is part of the MCP specification",
        ],
        "answer": "B",
        "explanation": "MCP Prompts are reusable server-defined templates/workflows. Prompt engineering (zero-shot, CoT, etc.) is about designing effective LLM inputs — different concepts.",
    },
    {
        "q": "What JSON-RPC 2.0 message type does NOT expect a response?",
        "options": [
            "A) Request",
            "B) Response",
            "C) Notification",
            "D) Query",
        ],
        "answer": "C",
        "explanation": "Notifications are one-way messages that don't expect a response. Requests expect responses. This is part of the JSON-RPC 2.0 specification.",
    },
    {
        "q": "In production, MCP, Function Calling, and LangChain Tools are typically:",
        "options": [
            "A) Mutually exclusive — you can only use one",
            "B) Combined — LangChain orchestrates, MCP standardizes tools, Function Calling triggers invocations",
            "C) All deprecated in favor of custom APIs",
            "D) Only used in prototypes, not production",
        ],
        "answer": "B",
        "explanation": "In modern production, these three work together: LangChain for orchestration, MCP for standardized tool ecosystem, Function Calling as the LLM's native invocation mechanism.",
    },
    {
        "q": "What does 'listChanged' mean in MCP capability negotiation?",
        "options": [
            "A) The list of users has changed",
            "B) The server will notify the client when the list of tools/resources/prompts changes",
            "C) The client must refresh the page",
            "D) The server is shutting down",
        ],
        "answer": "B",
        "explanation": "When a server declares 'listChanged: true' for a capability, it means it will send notifications when available tools, resources, or prompts change.",
    },
    {
        "q": "Which of these is NOT a valid MCP Server?",
        "options": [
            "A) A filesystem server that reads local files",
            "B) A GitHub server that manages repos and issues",
            "C) An LLM model like GPT-4",
            "D) A database server that queries PostgreSQL",
        ],
        "answer": "C",
        "explanation": "An LLM is NOT an MCP Server. MCP Servers expose tools and data TO the LLM. The LLM runs inside the Host and uses MCP Clients to connect to Servers.",
    },
    {
        "q": "MCP Resources support subscriptions. This means:",
        "options": [
            "A) Users need a paid subscription to use resources",
            "B) Clients can subscribe to be notified when resource content changes",
            "C) Resources are only available monthly",
            "D) Servers subscribe to client updates",
        ],
        "answer": "B",
        "explanation": "Resource subscriptions allow clients to get real-time notifications when the content of a resource changes (e.g., a file is updated).",
    },
    {
        "q": "Which SDK is the most mature for building MCP servers?",
        "options": [
            "A) Python SDK",
            "B) TypeScript SDK",
            "C) Go SDK",
            "D) Rust SDK",
        ],
        "answer": "B",
        "explanation": "The TypeScript SDK (@modelcontextprotocol/sdk) is the primary and most mature SDK, maintained directly by the MCP core team.",
    },
    {
        "q": "An MCP Tool definition requires which three elements?",
        "options": [
            "A) URL, method, headers",
            "B) Name, description, and input schema (JSON Schema)",
            "C) Key, value, timestamp",
            "D) Endpoint, payload, timeout",
        ],
        "answer": "B",
        "explanation": "Tools are defined with a name (identifier), description (for the LLM to understand purpose), and inputSchema (JSON Schema defining parameters).",
    },
    {
        "q": "When the exam says 'dynamic tool discovery,' it refers to MCP's ability to:",
        "options": [
            "A) Generate new tools automatically using AI",
            "B) Allow clients to discover available tools at runtime via list/get methods",
            "C) Download tools from the internet during execution",
            "D) Dynamically change tool behavior based on user mood",
        ],
        "answer": "B",
        "explanation": "Dynamic discovery means clients query the server at runtime to find available tools, resources, and prompts — no hardcoding needed.",
    },
    {
        "q": "MCP is now governed by which organization?",
        "options": [
            "A) Anthropic exclusively",
            "B) The Linux Foundation",
            "C) OpenAI",
            "D) The W3C",
        ],
        "answer": "B",
        "explanation": "MCP was created by Anthropic but is now an open-source project governed by the Linux Foundation.",
    },
    {
        "q": "What is the purpose of 'Resource Indicators (RFC 8707)' in MCP's OAuth?",
        "options": [
            "A) To indicate which resources need backup",
            "B) To bind access tokens to specific MCP servers, preventing token replay attacks",
            "C) To indicate resource file sizes",
            "D) To count the number of resources available",
        ],
        "answer": "B",
        "explanation": "Resource Indicators ensure that a token issued for one MCP server cannot be replayed to access a different server — critical for multi-server security.",
    },
    {
        "q": "In MCP, the initialization handshake includes:",
        "options": [
            "A) Only the client sending its name",
            "B) Client sends 'initialize' → Server responds with capabilities → Client sends 'initialized' notification",
            "C) Server sends the entire tool catalog immediately",
            "D) Both sides exchange encryption keys",
        ],
        "answer": "B",
        "explanation": "The three-step handshake: (1) Client sends initialize with capabilities, (2) Server responds with its capabilities, (3) Client confirms with 'initialized' notification.",
    },
    {
        "q": "Why is MCP described as 'transport-agnostic'?",
        "options": [
            "A) Because it doesn't need any transport",
            "B) Because the same protocol works over different transports (STDIO, HTTP/SSE) without changes",
            "C) Because it creates its own transport layer",
            "D) Because it only works with one transport",
        ],
        "answer": "B",
        "explanation": "MCP is transport-agnostic — the same JSON-RPC 2.0 protocol works identically over STDIO (local) or HTTP/SSE (remote) without protocol changes.",
    },
    {
        "q": "What is the best practice for MCP server design?",
        "options": [
            "A) Build one mega-server that handles everything",
            "B) One server per domain — separate filesystem, GitHub, database servers",
            "C) Build servers that only expose resources, never tools",
            "D) Avoid exposing any tool descriptions",
        ],
        "answer": "B",
        "explanation": "Best practice is one server per domain/capability — this follows separation of concerns, makes servers reusable, and simplifies security.",
    },
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🔴 HARD QUESTIONS (Security, Scenarios, Edge Cases, Traps)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HARD_QUESTIONS = [
    {
        "q": "An enterprise has 5 AI models and 10 internal tools. Without MCP, how many custom integrations are needed? With MCP?",
        "options": [
            "A) Without: 15, With: 5",
            "B) Without: 50, With: 15",
            "C) Without: 50, With: 50",
            "D) Without: 10, With: 15",
        ],
        "answer": "B",
        "explanation": "Without MCP: 5 models × 10 tools = 50 custom integrations. With MCP: 5 MCP clients + 10 MCP servers = 15 standardized integrations.",
    },
    {
        "q": "A security audit reveals that an MCP server's access token issued for Server A is being used to access Server B. Which security mechanism SHOULD have prevented this?",
        "options": [
            "A) PKCE",
            "B) Resource Indicators (RFC 8707) — tokens bound to specific servers",
            "C) STDIO transport",
            "D) JSON-RPC 2.0 versioning",
        ],
        "answer": "B",
        "explanation": "Resource Indicators (RFC 8707) bind tokens to specific MCP servers. If properly implemented, a token for Server A would be rejected by Server B.",
    },
    {
        "q": "An MCP server running locally via STDIO needs to authenticate. What's the correct approach?",
        "options": [
            "A) Use OAuth 2.1 with PKCE",
            "B) STDIO doesn't use OAuth; it relies on process-level security and environment credentials",
            "C) Use API keys in the request headers",
            "D) No authentication is ever needed for MCP",
        ],
        "answer": "B",
        "explanation": "STDIO transport is local process communication — it doesn't use OAuth. Security comes from process isolation and environment-based credentials.",
    },
    {
        "q": "A developer wants to build a tool that the AI can call to delete customer records. In production MCP, what security measure is MANDATORY?",
        "options": [
            "A) No special measures needed — the AI is trustworthy",
            "B) Human-in-the-loop approval before executing destructive operations",
            "C) Using a smaller model to reduce errors",
            "D) Disabling the tool after business hours",
        ],
        "answer": "B",
        "explanation": "MCP mandates User Consent and Control — destructive operations MUST require explicit human approval (human-in-the-loop) before execution.",
    },
    {
        "q": "Which statement about MCP is FALSE?",
        "options": [
            "A) MCP is vendor-neutral and open source",
            "B) MCP uses JSON-RPC 2.0 for messaging",
            "C) MCP uses stateless REST-like communication",
            "D) MCP supports both local (STDIO) and remote (HTTP/SSE) transport",
        ],
        "answer": "C",
        "explanation": "TRAP: MCP does NOT use stateless REST. It uses JSON-RPC 2.0 which is stateful and bidirectional. This is a common exam trap.",
    },
    {
        "q": "An MCP client connects to a server and receives capabilities with 'tools: { listChanged: true }'. What does this mean?",
        "options": [
            "A) The server's tools have already changed and are invalid",
            "B) The server will proactively notify the client if the available tools change during the session",
            "C) The client must poll the server every second",
            "D) Tools can only be used once before they change",
        ],
        "answer": "B",
        "explanation": "'listChanged: true' means the server supports dynamic notification — it will push updates if tools are added, removed, or modified during the session.",
    },
    {
        "q": "Why did MCP deprecate the 'Sampling' primitive?",
        "options": [
            "A) It was never implemented",
            "B) Security concerns — it allowed servers to generate arbitrary LLM content via the host",
            "C) It was too slow",
            "D) It was renamed but not removed",
        ],
        "answer": "B",
        "explanation": "Sampling allowed servers to request LLM completions via the Host, which raised security concerns. Implementations now integrate directly with LLM APIs instead.",
    },
    {
        "q": "A company uses Claude Desktop (Host) with MCP clients connected to a filesystem server and a GitHub server. The user asks 'What are the open issues related to files changed this week?' What happens?",
        "options": [
            "A) Only the filesystem server responds",
            "B) The Host routes queries to both MCP clients — filesystem client gets changed files, GitHub client gets open issues, LLM synthesizes",
            "C) The user must make two separate requests",
            "D) MCP can't handle multi-server queries",
        ],
        "answer": "B",
        "explanation": "The Host coordinates: the filesystem MCP client retrieves changed files, the GitHub MCP client retrieves issues, and the LLM synthesizes the combined context into an answer.",
    },
    {
        "q": "What is the 'confused deputy' attack in the context of MCP?",
        "options": [
            "A) When the user is confused about which tool to use",
            "B) When an MCP server tricks the host into performing actions beyond its authorized scope",
            "C) When two servers conflict with each other",
            "D) When the LLM generates incorrect JSON",
        ],
        "answer": "B",
        "explanation": "A 'confused deputy' attack occurs when a malicious server manipulates the trusted host into performing unauthorized actions — MCP's scoped tokens and consent mechanisms defend against this.",
    },
    {
        "q": "In OAuth 2.1 for MCP, PKCE is:",
        "options": [
            "A) Optional for public clients only",
            "B) Mandatory for ALL authorization flows",
            "C) Only needed for mobile applications",
            "D) Replaced by API keys",
        ],
        "answer": "B",
        "explanation": "OAuth 2.1 mandates PKCE (Proof Key for Code Exchange) for ALL flows — not just public clients. This is a key security improvement over OAuth 2.0.",
    },
    {
        "q": "A team debates whether to use MCP or direct API integration for connecting their single AI model to a single database. What's the best advice?",
        "options": [
            "A) Always use MCP — it's the latest technology",
            "B) For a simple 1:1 integration, direct API or Function Calling is simpler; MCP shines with multiple models/tools",
            "C) Never use MCP — it's too complex",
            "D) Use both simultaneously for redundancy",
        ],
        "answer": "B",
        "explanation": "MCP's benefits (N+M reduction, reusability) shine with multiple models and tools. For a simple 1:1 case, the overhead may not be justified.",
    },
    {
        "q": "MCP security best practices state that clients should handle unrecognized capabilities by:",
        "options": [
            "A) Accepting them and trying to use them",
            "B) Refusing them — only process explicitly recognized and trusted capabilities",
            "C) Ignoring the entire server",
            "D) Reporting them to Anthropic",
        ],
        "answer": "B",
        "explanation": "Security best practice: refuse unrecognized capabilities. Default to deny, not allow. This prevents malicious servers from introducing unexpected behaviors.",
    },
    {
        "q": "What is the relationship between MCP Tools and the ReAct prompting pattern?",
        "options": [
            "A) They are unrelated",
            "B) ReAct provides the reasoning pattern (Thought-Action-Observation) that LLMs use to decide which MCP Tools to call",
            "C) MCP Tools replace ReAct",
            "D) ReAct is an MCP primitive",
        ],
        "answer": "B",
        "explanation": "ReAct (Reasoning + Acting) is the prompting pattern where the LLM reasons about what to do and takes actions. MCP Tools are the standardized actions the LLM can take.",
    },
    {
        "q": "An MCP server exposes both a 'query_database' Tool and a 'database_schema' Resource for the same database. Why both?",
        "options": [
            "A) Redundancy in case one fails",
            "B) The Resource (schema) provides read-only context; the Tool (query) performs the actual execution. Different purposes, different control models",
            "C) One is for production, one for development",
            "D) It's a design mistake — you only need one",
        ],
        "answer": "B",
        "explanation": "Resources provide context (schema for the LLM to understand table structure). Tools perform actions (execute queries). Different purposes: context vs. action.",
    },
    {
        "q": "Which scenario demonstrates when MCP is MOST valuable?",
        "options": [
            "A) A single chatbot using OpenAI's API for Q&A",
            "B) An enterprise with 3 AI models (Claude, GPT-4, Gemini) needing access to 8 internal tools (GitHub, Jira, DB, Slack, etc.)",
            "C) A student running a local Jupyter notebook",
            "D) A static website with no AI features",
        ],
        "answer": "B",
        "explanation": "MCP's value is maximized in multi-model, multi-tool environments. 3 models × 8 tools = 24 custom integrations without MCP, but only 11 (3+8) with MCP.",
    },
    {
        "q": "What is 'Enterprise-Managed Authorization (EMA)' in MCP?",
        "options": [
            "A) A simple API key system",
            "B) An authorization pattern focusing on non-human agent identity and conditional, ephemeral access",
            "C) A type of MCP server",
            "D) A deprecated feature",
        ],
        "answer": "B",
        "explanation": "EMA addresses enterprise needs for AI agent authorization — managing non-human identities, granting conditional and time-limited (ephemeral) access to resources.",
    },
    {
        "q": "How do MCP Resources differ from RAG (Retrieval-Augmented Generation) context?",
        "options": [
            "A) They are identical concepts",
            "B) MCP Resources are server-exposed data accessible via protocol; RAG context is retrieved via similarity search from vector databases",
            "C) RAG is part of MCP",
            "D) MCP Resources don't provide context to LLMs",
        ],
        "answer": "B",
        "explanation": "MCP Resources are protocol-level data exposed by servers. RAG retrieves context via embedding similarity from vector databases. Different mechanisms, both provide context to LLMs.",
    },
    {
        "q": "An MCP tool execution fails. According to best practices, the error should be:",
        "options": [
            "A) Silently ignored",
            "B) Returned as a clear, descriptive message that the LLM can understand and act upon",
            "C) Cause the entire session to crash",
            "D) Only logged to a file, never shown to the LLM",
        ],
        "answer": "B",
        "explanation": "Errors should be returned as clear, understandable messages so the LLM can reason about the failure and potentially retry or inform the user.",
    },
    {
        "q": "MCP uses OAuth 2.1 instead of OAuth 2.0 primarily because OAuth 2.1:",
        "options": [
            "A) Is faster",
            "B) Eliminates the implicit grant, mandates PKCE, and requires exact redirect URI matching — reducing attack surface",
            "C) Uses JSON instead of XML",
            "D) Doesn't require client registration",
        ],
        "answer": "B",
        "explanation": "OAuth 2.1 is more secure: no implicit grant, mandatory PKCE for all flows, exact redirect URI matching. These close known OAuth 2.0 vulnerabilities.",
    },
    {
        "q": "What is the principle of 'least privilege' in MCP context?",
        "options": [
            "A) Users should have admin access to everything",
            "B) MCP servers and clients should request and grant only the minimum permissions necessary for their function",
            "C) Only use the least expensive LLM model",
            "D) Minimize the number of MCP servers",
        ],
        "answer": "B",
        "explanation": "Least privilege means scoping tokens, roots, and capabilities to the minimum necessary — reducing the blast radius if a component is compromised.",
    },
    {
        "q": "A developer notices that MCP notifications don't have an 'id' field. Why?",
        "options": [
            "A) It's a bug in the specification",
            "B) Notifications are one-way messages that don't expect responses, so they don't need an ID for response matching",
            "C) IDs are only used in version 1.0",
            "D) The server generates IDs internally but doesn't share them",
        ],
        "answer": "B",
        "explanation": "In JSON-RPC 2.0, notifications are fire-and-forget — no response is expected, so no ID is needed to match request-response pairs.",
    },
    {
        "q": "Why should MCP Tool descriptions be clear and detailed?",
        "options": [
            "A) For documentation purposes only",
            "B) Because the LLM uses the description to understand what the tool does and decide when to call it",
            "C) To make the code look professional",
            "D) For regulatory compliance only",
        ],
        "answer": "B",
        "explanation": "The LLM reads tool descriptions to decide which tool to invoke. Poor descriptions → wrong tool selection → poor user experience or errors.",
    },
    {
        "q": "An MCP server claims it supports 'tools', 'resources', and a new 'quantum_compute' capability. A security-conscious client should:",
        "options": [
            "A) Accept all capabilities including quantum_compute",
            "B) Accept tools and resources (known), but refuse the unrecognized 'quantum_compute' capability",
            "C) Reject the entire server",
            "D) Ask the user which capabilities to accept",
        ],
        "answer": "B",
        "explanation": "Security best practice: accept only recognized capabilities, refuse unknown ones. The client can safely use tools and resources while ignoring the unrecognized capability.",
    },
    {
        "q": "Compare: Function Calling defines tools statically in each API call. MCP allows dynamic discovery at runtime. What is the practical advantage of MCP's approach?",
        "options": [
            "A) No advantage — static is always better",
            "B) When servers add new tools, clients discover them automatically without code changes",
            "C) Dynamic discovery is slower but more secure",
            "D) It allows tools to be written in any programming language",
        ],
        "answer": "B",
        "explanation": "Dynamic discovery means the tool ecosystem can evolve independently — servers add/remove tools and clients discover changes at runtime without redeployment.",
    },
    {
        "q": "What is the main security risk of MCP that the specification explicitly warns about?",
        "options": [
            "A) Data encryption is not supported",
            "B) MCP enables arbitrary code execution and data access, so user consent and control are critical",
            "C) MCP servers can access the internet without permission",
            "D) JSON-RPC 2.0 is inherently insecure",
        ],
        "answer": "B",
        "explanation": "MCP's power comes with risk: tools can execute code, access data, and make API calls. The spec emphasizes User Consent and Control as the primary defense.",
    },
]


def run_quiz(
    questions: list,
    difficulty: str,
    num_questions: int = 10,
    emoji: str = "🟢",
) -> int:
    """Run a quiz from the given question pool. Returns the score."""
    selected = random.sample(questions, min(num_questions, len(questions)))
    score = 0
    total = len(selected)

    print(f"\n{'=' * 60}")
    print(f"{emoji} MCP QUIZ — {difficulty.upper()} DIFFICULTY")
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
    print("🧠 MODEL CONTEXT PROTOCOL (MCP) — PRACTICE QUIZ")
    print("=" * 60)
    print(f"\nTotal questions available: {len(EASY_QUESTIONS) + len(MEDIUM_QUESTIONS) + len(HARD_QUESTIONS)}")
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
        print("🏆 Outstanding! You have mastered MCP for the L2 exam!")
    elif pct >= 80:
        print("🎉 Excellent! Strong understanding. Review the few you missed.")
    elif pct >= 60:
        print("👍 Good progress! Re-read sections on the topics you missed.")
    else:
        print("📚 Keep studying! Re-read 01_MCP_Study_Guide.md and try again.")

    print("\nTip: Run again with different difficulty levels to test all areas!")


if __name__ == "__main__":
    main()
