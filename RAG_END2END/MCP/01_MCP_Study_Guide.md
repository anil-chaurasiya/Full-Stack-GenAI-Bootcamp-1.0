# 📚 Model Context Protocol (MCP) — Complete Study Guide (GenAI L2 Exam)

> **Goal**: Master every MCP concept tested in the Google Cloud GenAI Level-2 exam.
> Covers architecture, primitives, transport, security, lifecycle, and real-world use cases.

---

## Table of Contents

1. [What is MCP?](#1-what-is-mcp)
2. [The Problem MCP Solves](#2-the-problem-mcp-solves)
3. [MCP Architecture — The Big Picture](#3-mcp-architecture--the-big-picture)
4. [Core Components Deep Dive](#4-core-components-deep-dive)
   - 4.1 Host
   - 4.2 Client
   - 4.3 Server
5. [MCP Primitives (Server-Side)](#5-mcp-primitives-server-side)
   - 5.1 Tools
   - 5.2 Resources
   - 5.3 Prompts
6. [MCP Primitives (Client-Side)](#6-mcp-primitives-client-side)
   - 6.1 Roots
   - 6.2 Sampling (Deprecated)
7. [Transport Layer](#7-transport-layer)
   - 7.1 STDIO
   - 7.2 Streamable HTTP (SSE)
8. [Communication Protocol — JSON-RPC 2.0](#8-communication-protocol--json-rpc-20)
9. [Connection Lifecycle](#9-connection-lifecycle)
10. [Capability Negotiation](#10-capability-negotiation)
11. [Security & Authorization](#11-security--authorization)
12. [MCP vs Function Calling vs LangChain Tools](#12-mcp-vs-function-calling-vs-langchain-tools)
13. [MCP SDKs & Ecosystem](#13-mcp-sdks--ecosystem)
14. [MCP in Production — Best Practices](#14-mcp-in-production--best-practices)
15. [Real-World MCP Use Cases](#15-real-world-mcp-use-cases)
16. [Key Terminology Glossary](#16-key-terminology-glossary)
17. [Common Exam Patterns & Traps](#17-common-exam-patterns--traps)

---

## 1. What is MCP?

**Model Context Protocol (MCP)** is an **open standard protocol** created by **Anthropic** that provides a universal, standardized way for AI applications to connect to external data sources, tools, and services.

> **Exam Definition**: MCP is an open-source protocol that standardizes how AI models and applications interact with external tools, data sources, and services through a client-server architecture using JSON-RPC 2.0 messaging.

### Key Facts for Exam ⚡

| Fact | Detail |
|------|--------|
| **Created by** | Anthropic (open-sourced) |
| **Governed by** | Linux Foundation (as of 2025) |
| **Protocol** | JSON-RPC 2.0 |
| **Architecture** | Client-Server |
| **License** | Open Source |
| **Transport** | STDIO (local), Streamable HTTP/SSE (remote) |
| **Purpose** | Standardize LLM ↔ Tool/Data integration |

**Think of MCP as "USB-C for AI"** — just as USB-C provides a universal connector for devices, MCP provides a universal connector for AI applications to access tools and data.

---

## 2. The Problem MCP Solves

### The N×M Integration Problem

Without MCP, every AI application needs a **custom integration** for every tool/data source:

```
WITHOUT MCP (N×M Problem):
┌──────────┐     ┌──────────┐
│ Claude   │────▶│ GitHub   │  ← custom connector
│          │────▶│ Slack    │  ← custom connector
│          │────▶│ Database │  ← custom connector
└──────────┘     └──────────┘
┌──────────┐     ┌──────────┐
│ GPT-4    │────▶│ GitHub   │  ← DIFFERENT custom connector
│          │────▶│ Slack    │  ← DIFFERENT custom connector
│          │────▶│ Database │  ← DIFFERENT custom connector
└──────────┘     └──────────┘

Total integrations needed: N models × M tools = 6 custom connectors
```

### MCP Solution: N+M

```
WITH MCP (N+M Solution):
┌──────────┐                   ┌──────────────┐
│ Claude   │──┐                │ GitHub       │
└──────────┘  │    ┌───────┐   │ MCP Server   │
              ├───▶│  MCP  │──▶└──────────────┘
┌──────────┐  │    │Protocol│   ┌──────────────┐
│ GPT-4    │──┘    │       │──▶│ Slack        │
└──────────┘       │       │   │ MCP Server   │
                   │       │   └──────────────┘
                   │       │   ┌──────────────┐
                   └───────┘──▶│ Database     │
                               │ MCP Server   │
                               └──────────────┘

Total integrations: N + M = 2 clients + 3 servers = 5
```

### Exam Key ⚡
- MCP reduces the integration complexity from **O(N×M)** to **O(N+M)**.
- Build a tool **once** as an MCP Server → works with **any** MCP-compatible client.
- Build an MCP Client **once** → can connect to **any** MCP Server.

---

## 3. MCP Architecture — The Big Picture

```
┌─────────────────────────────────────────────────────────────┐
│                        MCP HOST                              │
│  (AI Application: Claude Desktop, Cursor, Custom App)        │
│                                                              │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐ │
│  │  MCP Client 1  │  │  MCP Client 2  │  │  MCP Client 3  │ │
│  │  (1:1 with     │  │  (1:1 with     │  │  (1:1 with     │ │
│  │   Server 1)    │  │   Server 2)    │  │   Server 3)    │ │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘ │
│          │                   │                   │           │
└──────────┼───────────────────┼───────────────────┼───────────┘
           │ STDIO             │ HTTP/SSE          │ STDIO
           ▼                   ▼                   ▼
   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
   │ MCP Server 1 │   │ MCP Server 2 │   │ MCP Server 3 │
   │ (Filesystem) │   │ (GitHub API) │   │ (Database)   │
   │              │   │              │   │              │
   │ Exposes:     │   │ Exposes:     │   │ Exposes:     │
   │ • Tools      │   │ • Tools      │   │ • Tools      │
   │ • Resources  │   │ • Resources  │   │ • Resources  │
   │ • Prompts    │   │ • Prompts    │   │ • Prompts    │
   └──────────────┘   └──────────────┘   └──────────────┘
```

### The Three Layers

| Layer | Component | Role |
|-------|-----------|------|
| **Application** | Host | The AI application the user interacts with |
| **Connection** | Client | Maintains a stateful 1:1 connection to one server |
| **Service** | Server | Exposes tools, resources, and prompts |

### Exam Key ⚡
- **1 Host** can contain **many Clients**.
- Each **Client** connects to exactly **1 Server** (1:1 relationship).
- A **Host** manages access control and user consent.
- Communication is **bidirectional** and **stateful** (not stateless REST).

---

## 4. Core Components Deep Dive

### 4.1 Host

**Definition**: The AI application that the user directly interacts with. It contains one or more MCP Clients.

**Examples**: Claude Desktop, Cursor IDE, Windsurf, VS Code with Copilot, custom AI applications.

**Responsibilities**:
- Manage multiple client connections
- Enforce **security policies** and **user consent**
- Control which servers can access what
- Route LLM interactions and tool calls
- Provide the UI for human-in-the-loop approval

### 4.2 Client

**Definition**: A protocol-level component within the Host that maintains a **direct, stateful, 1:1 connection** to a single MCP Server.

**Responsibilities**:
- Establish and maintain the connection
- Send requests to the server (list tools, call tools, read resources)
- Receive notifications and responses
- Handle capability negotiation during initialization

**Exam Key**: The Client is NOT the same as the Host. A Host can have many Clients. Each Client talks to exactly one Server.

### 4.3 Server

**Definition**: A lightweight program that exposes specific capabilities (tools, resources, prompts) to MCP Clients.

**Examples**: 
- Filesystem Server — reads/writes local files
- GitHub Server — interacts with GitHub API
- PostgreSQL Server — queries databases
- Slack Server — sends/reads messages
- Web Search Server — performs internet searches

**Responsibilities**:
- Expose tools, resources, and prompts
- Handle tool execution requests
- Return results in standardized format
- Declare capabilities during initialization

---

## 5. MCP Primitives (Server-Side)

Servers expose **three types of primitives** to provide context and capabilities to AI models:

### Overview Table

| Primitive | Control | Description | Analogy |
|-----------|---------|-------------|---------|
| **Tools** | Model-controlled | Functions the AI can invoke to perform actions | "What the AI can DO" |
| **Resources** | Application-controlled | Read-only data for context | "What the AI can READ" |
| **Prompts** | User-controlled | Templated workflows/interactions | "What the AI can SUGGEST" |

---

### 5.1 Tools ⚙️

**Definition**: Executable functions that the LLM can decide to invoke to perform actions or retrieve dynamic information.

**Key Characteristics**:
- **Model-controlled** — the AI model decides when and how to call them
- Can have **side effects** (write to DB, send emails, create files)
- Require **human approval** in production (human-in-the-loop)
- Defined with a **name**, **description**, and **input schema** (JSON Schema)

**Example Tool Definition**:
```json
{
  "name": "query_database",
  "description": "Execute a read-only SQL query on the customer database",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "The SQL query to execute"
      }
    },
    "required": ["query"]
  }
}
```

**Common MCP Tools**:
- `read_file` / `write_file` — File operations
- `search_web` — Internet search
- `query_database` — Database queries
- `create_github_issue` — GitHub operations
- `send_slack_message` — Messaging

**Exam Key ⚡**: Tools are the most important primitive. They are what make MCP servers actionable. The LLM autonomously decides which tool to call based on the tool's name and description.

---

### 5.2 Resources 📄

**Definition**: Read-only data and context that servers expose for the AI to consume. Resources provide information but do NOT perform actions.

**Key Characteristics**:
- **Application-controlled** — the host application decides which resources to include
- **Read-only** — no side effects
- Identified by **URI** (e.g., `file:///path/to/doc.txt`, `db://customers/schema`)
- Can be **static** (fixed content) or **dynamic** (generated on demand)
- Support **subscriptions** — clients can get notified when resources change

**Example Resource**:
```json
{
  "uri": "file:///project/README.md",
  "name": "Project README",
  "description": "Main documentation for the project",
  "mimeType": "text/markdown"
}
```

**Resource vs Tool**:
| Aspect | Resource | Tool |
|--------|----------|------|
| Side effects | None (read-only) | Can have side effects |
| Control | Application decides | Model decides |
| Purpose | Provide context | Perform actions |
| Analogy | Reading a file | Running a command |

---

### 5.3 Prompts 💬

**Definition**: Pre-defined prompt templates and interaction workflows that servers expose to help users interact with the AI effectively.

**Key Characteristics**:
- **User-controlled** — typically selected by the user from a menu or slash command
- Templated messages with **arguments**
- Can include **multi-step workflows**
- Appear as slash commands or menu items in the host UI

**Example Prompt**:
```json
{
  "name": "explain_code",
  "description": "Explain a code snippet in plain English",
  "arguments": [
    {
      "name": "language",
      "description": "Programming language",
      "required": true
    },
    {
      "name": "code",
      "description": "The code to explain",
      "required": true
    }
  ]
}
```

**Exam Key ⚡**: Prompts are NOT the same as "prompting techniques" (zero-shot, few-shot, etc.). In MCP, "Prompts" are **server-defined templates** that standardize common interaction patterns.

---

## 6. MCP Primitives (Client-Side)

### 6.1 Roots 🌲

**Definition**: Entry points (typically filesystem paths or URIs) that the client shares with the server to define the boundaries of where the server can operate.

**Example**: A client might set roots to `["file:///home/user/project"]`, telling the server to only access files within that project directory.

**Purpose**: Security boundary — limits the scope of server operations.

### 6.2 Sampling 🔄 (Deprecated)

**Definition**: A mechanism that allowed MCP servers to request an LLM completion from the Host.

**Status**: **DEPRECATED** as of the July 2026 protocol update (version 2026-07-28).

**Why Deprecated**: Implementations are now encouraged to integrate directly with LLM provider APIs instead of routing through the MCP protocol.

**Exam Tip**: If the exam asks about Sampling, know that it existed but has been deprecated. Focus on Tools, Resources, and Prompts instead.

---

## 7. Transport Layer

MCP is **transport-agnostic** but supports two primary transport mechanisms:

### 7.1 STDIO (Standard Input/Output)

```
┌────────────┐  stdin/stdout  ┌────────────┐
│ MCP Client │ ◀════════════▶ │ MCP Server │
│ (in Host)  │   (local pipe) │ (process)  │
└────────────┘                └────────────┘
```

| Aspect | Detail |
|--------|--------|
| **Use Case** | Local communication |
| **How It Works** | Server runs as a child process; communicates via stdin/stdout |
| **Latency** | Very low (no network) |
| **Security** | Inherits process permissions; no OAuth needed |
| **Examples** | File system access, local database, local code execution |
| **Best For** | Desktop apps, IDEs, local development |

### 7.2 Streamable HTTP (SSE — Server-Sent Events)

```
┌────────────┐    HTTP/SSE     ┌────────────┐
│ MCP Client │ ◀══════════════▶│ MCP Server │
│ (in Host)  │   (network)    │ (remote)   │
└────────────┘                └────────────┘
```

| Aspect | Detail |
|--------|--------|
| **Use Case** | Remote communication |
| **How It Works** | HTTP requests + SSE for server-to-client streaming |
| **Latency** | Network-dependent |
| **Security** | Requires OAuth 2.1 authentication |
| **Examples** | Cloud APIs, SaaS integrations, shared services |
| **Best For** | Production, enterprise, distributed systems |

### Exam Key ⚡
- **STDIO** = local, fast, simple, no auth needed
- **SSE/HTTP** = remote, scalable, requires OAuth 2.1
- MCP is **transport-agnostic** — the same protocol works over both
- Choose STDIO for development, HTTP/SSE for production

---

## 8. Communication Protocol — JSON-RPC 2.0

All MCP messages use **JSON-RPC 2.0** format:

### Request
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "query_database",
    "arguments": {
      "query": "SELECT COUNT(*) FROM users"
    }
  }
}
```

### Response
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Count: 42,857 users"
      }
    ]
  }
}
```

### Notification (No Response Expected)
```json
{
  "jsonrpc": "2.0",
  "method": "notifications/resources/updated",
  "params": {
    "uri": "file:///project/data.csv"
  }
}
```

### Key Facts
- **Stateful sessions** — not stateless like REST
- **Bidirectional** — both client and server can send messages
- **Three message types**: Request (expects response), Response, Notification (no response)
- **Each request has a unique `id`** for matching responses

---

## 9. Connection Lifecycle

```
CLIENT                                SERVER
  │                                      │
  │──── initialize (capabilities) ──────▶│
  │◀─── initialize response ────────────│
  │──── initialized notification ──────▶│
  │                                      │
  │         ═══ ACTIVE SESSION ═══       │
  │                                      │
  │──── tools/list ─────────────────────▶│
  │◀─── tools list response ────────────│
  │──── tools/call ─────────────────────▶│
  │◀─── tool result ────────────────────│
  │──── resources/read ─────────────────▶│
  │◀─── resource content ──────────────│
  │                                      │
  │         ═══ SHUTDOWN ═══             │
  │                                      │
  │──── shutdown request ───────────────▶│
  │◀─── shutdown response ──────────────│
  │──── exit notification ──────────────▶│
  │                                      │
```

### Phases

1. **Initialization**: Client sends `initialize` with its capabilities → Server responds with its capabilities → Client sends `initialized` notification.
2. **Active Session**: Client can list tools, call tools, read resources, use prompts. Server can send notifications.
3. **Shutdown**: Graceful termination with `shutdown` request and `exit` notification.

### Exam Key ⚡
- Initialization includes **capability negotiation** — both sides declare what they support.
- The session is **stateful** — context is maintained across requests.
- Either side can send **notifications** during the active session.

---

## 10. Capability Negotiation

During initialization, client and server exchange their capabilities:

```json
// Client → Server: "Here's what I support"
{
  "method": "initialize",
  "params": {
    "protocolVersion": "2025-03-26",
    "capabilities": {
      "roots": { "listChanged": true },
      "sampling": {}
    },
    "clientInfo": {
      "name": "MyChatApp",
      "version": "1.0.0"
    }
  }
}

// Server → Client: "Here's what I offer"
{
  "result": {
    "protocolVersion": "2025-03-26",
    "capabilities": {
      "tools": { "listChanged": true },
      "resources": { "subscribe": true, "listChanged": true },
      "prompts": { "listChanged": true }
    },
    "serverInfo": {
      "name": "GitHubServer",
      "version": "2.1.0"
    }
  }
}
```

### Key Principles
- Both sides declare what they can do
- Clients should **refuse unrecognized capabilities** (security best practice)
- `listChanged` means the server will notify when the list of tools/resources/prompts changes
- Version negotiation ensures **backward compatibility**

---

## 11. Security & Authorization

### Security Model Overview

```
┌────────────────────────────────────────────────┐
│              SECURITY LAYERS                    │
├────────────────────────────────────────────────┤
│                                                │
│  1. TRANSPORT SECURITY                         │
│     • STDIO: Process-level isolation           │
│     • HTTP: TLS + OAuth 2.1                    │
│                                                │
│  2. AUTHORIZATION (OAuth 2.1)                  │
│     • PKCE mandatory for all flows             │
│     • Scoped access tokens                     │
│     • Resource indicators (RFC 8707)           │
│                                                │
│  3. USER CONSENT                               │
│     • Human-in-the-loop for tool execution     │
│     • Explicit approval for data access        │
│     • Clear UI for permission management       │
│                                                │
│  4. LEAST PRIVILEGE                            │
│     • Roots limit filesystem scope             │
│     • Scoped tokens limit API access           │
│     • Refuse unrecognized capabilities         │
│                                                │
└────────────────────────────────────────────────┘
```

### OAuth 2.1 in MCP

| Step | What Happens |
|------|-------------|
| **1. Metadata Discovery** | Client discovers server's auth endpoints via OAuth 2.0 Protected Resource Metadata |
| **2. Client Registration** | Client obtains client ID (dynamic registration or pre-registration) |
| **3. Authorization + PKCE** | Client performs auth flow with mandatory PKCE |
| **4. Token Issuance** | Server issues scoped access token |
| **5. Resource Indicators** | Token bound to specific MCP server (RFC 8707) — prevents token replay |

### Security Principles for Exam ⚡

1. **User Consent is Mandatory** — Hosts must get explicit user approval before tool execution
2. **Least Privilege** — Request only necessary permissions
3. **STDIO doesn't need OAuth** — it's local, process-level security
4. **HTTP/SSE MUST use OAuth 2.1** — with PKCE mandatory
5. **No Implicit Grant** — OAuth 2.1 eliminates the implicit flow
6. **Token Scoping** — Tokens are limited to specific resources/actions
7. **Treat MCP as an Untrusted Boundary** — validate all inputs, sanitize outputs

---

## 12. MCP vs Function Calling vs LangChain Tools

This is a **high-priority exam topic** — understanding the differences and when to use each.

### Comparison Table

| Dimension | Function Calling | LangChain Tools | MCP |
|-----------|:---:|:---:|:---:|
| **Type** | Proprietary API Feature | Framework Abstraction | Open Standard Protocol |
| **Created By** | OpenAI, Google, etc. | LangChain Inc. | Anthropic → Linux Foundation |
| **Scope** | Single model ↔ single tool | Multi-model orchestration | Universal tool ecosystem |
| **Model Lock-in** | 🔴 High (provider-specific) | 🟡 Low (adapter-based) | 🟢 None (vendor-neutral) |
| **Setup Complexity** | 🟢 Simple | 🟡 Medium | 🔴 Higher (server setup) |
| **Discovery** | Static (defined in API call) | Static (coded in agent) | Dynamic (runtime discovery) |
| **Reusability** | Low (rewrite per model) | Medium (within LangChain) | High (any MCP client) |
| **Best For** | Simple prototypes | Complex agent workflows | Enterprise multi-tool systems |
| **State** | Stateless | Managed by framework | Stateful sessions |

### How They Work Together

```
┌────────────────────────────────────────────────────────────┐
│                    PRODUCTION STACK                         │
│                                                            │
│  ┌─────────────┐                                          │
│  │ LangChain   │ ← Orchestration layer (chains, memory)   │
│  │ Agent       │                                          │
│  └──────┬──────┘                                          │
│         │                                                  │
│  ┌──────▼──────┐                                          │
│  │    MCP      │ ← Standardized tool/data connectivity     │
│  │  Protocol   │                                          │
│  └──────┬──────┘                                          │
│         │                                                  │
│  ┌──────▼──────┐                                          │
│  │  Function   │ ← LLM's native mechanism to trigger      │
│  │  Calling    │   tool invocations                       │
│  └─────────────┘                                          │
└────────────────────────────────────────────────────────────┘
```

### Decision Guide

```
Simple prototype with one LLM?          → Function Calling
Complex agent with memory & chains?     → LangChain Tools
Enterprise with many tools & models?    → MCP
Need all three benefits?                → Combine them!
```

---

## 13. MCP SDKs & Ecosystem

### Official SDKs

| Language | SDK | Use Case |
|----------|-----|----------|
| **TypeScript** | `@modelcontextprotocol/sdk` | Primary SDK, most mature |
| **Python** | `mcp` | Data science, ML pipelines |
| **Java/Kotlin** | `mcp-java-sdk` | Enterprise applications |
| **C#** | `mcp-csharp-sdk` | .NET ecosystem |
| **Go** | `mcp-go` | Cloud-native services |
| **Rust** | `mcp-rust-sdk` | Performance-critical |
| **Swift** | `mcp-swift-sdk` | Apple platforms |

### Python MCP Server Example

```python
from mcp.server import Server
from mcp.types import Tool, TextContent

# Create an MCP server
server = Server("my-server")

# Define a tool
@server.tool()
async def get_weather(city: str) -> list[TextContent]:
    """Get current weather for a city."""
    # ... fetch weather data ...
    return [TextContent(type="text", text=f"Weather in {city}: 72°F, Sunny")]

# Run with STDIO transport
if __name__ == "__main__":
    import asyncio
    from mcp.server.stdio import stdio_server
    
    asyncio.run(stdio_server(server))
```

### Popular MCP Servers

| Server | What It Does |
|--------|-------------|
| `filesystem` | Read/write local files |
| `github` | Interact with GitHub repos, issues, PRs |
| `postgres` / `sqlite` | Query databases |
| `slack` | Send/read Slack messages |
| `brave-search` | Web search |
| `puppeteer` | Browser automation |
| `memory` | Persistent key-value store |

### MCP-Compatible Hosts

- **Claude Desktop** (Anthropic)
- **Cursor** (IDE)
- **Windsurf** (IDE)
- **VS Code + GitHub Copilot**
- **Sourcegraph Cody**
- **Custom applications** (via SDK)

---

## 14. MCP in Production — Best Practices

### Security Checklist ✅

1. **Always use OAuth 2.1** for remote (HTTP/SSE) connections
2. **Implement human-in-the-loop** — get user approval before executing tools with side effects
3. **Scope tokens narrowly** — minimum necessary permissions
4. **Set roots appropriately** — limit filesystem access to project directories
5. **Validate all tool inputs** — treat everything from the server as untrusted
6. **Log all tool calls** — auditability is critical for enterprise
7. **Use TLS for HTTP transport** — never transmit over plain HTTP

### Design Checklist ✅

1. **One server per domain** — filesystem server, GitHub server, database server (not one mega-server)
2. **Clear tool descriptions** — the LLM uses descriptions to decide which tool to call
3. **Idempotent tools when possible** — safe to retry without side effects
4. **Error handling** — return clear error messages the LLM can understand
5. **Rate limiting** — protect backend services from excessive calls
6. **Graceful degradation** — handle server unavailability without crashing

---

## 15. Real-World MCP Use Cases

### Use Case 1: AI-Powered IDE (Cursor/Windsurf)
```
User types code → IDE (Host) → MCP Client → Filesystem Server (reads project files)
                                           → GitHub Server (fetches issues)
                                           → Database Server (queries schema)
                              → LLM generates contextual code
```

### Use Case 2: Enterprise Knowledge Assistant
```
Employee asks question → Chat App (Host) → MCP Client → Confluence Server (docs)
                                                       → Jira Server (tickets)
                                                       → Database Server (metrics)
                                         → LLM provides grounded answer
```

### Use Case 3: Automated DevOps Agent
```
Alert triggers → Agent (Host) → MCP Client → Monitoring Server (get metrics)
                                            → Kubernetes Server (check pods)
                                            → Slack Server (notify team)
                               → LLM diagnoses issue and takes action
```

---

## 16. Key Terminology Glossary

| Term | Definition |
|------|-----------|
| **MCP** | Model Context Protocol — open standard for LLM ↔ tool communication |
| **Host** | The AI application containing MCP clients (e.g., Claude Desktop) |
| **Client** | Component within the host that maintains a 1:1 connection to a server |
| **Server** | Program that exposes tools, resources, and prompts |
| **Tools** | Model-controlled executable functions with potential side effects |
| **Resources** | Application-controlled read-only data for context |
| **Prompts** | User-controlled templated workflows |
| **Roots** | Client-defined filesystem/URI boundaries for server operations |
| **Sampling** | (Deprecated) Mechanism for servers to request LLM completions |
| **JSON-RPC 2.0** | The messaging protocol used for all MCP communication |
| **STDIO** | Transport for local server communication via stdin/stdout |
| **SSE** | Server-Sent Events — HTTP-based transport for remote servers |
| **Capability Negotiation** | Handshake where client and server declare supported features |
| **OAuth 2.1** | Authorization standard required for HTTP/SSE transport |
| **PKCE** | Proof Key for Code Exchange — mandatory in OAuth 2.1 |
| **N×M Problem** | The integration explosion MCP solves (N models × M tools) |
| **Primitive** | A core building block in MCP (Tools, Resources, Prompts) |
| **Stateful Session** | MCP connections maintain state across requests (unlike REST) |

---

## 17. Common Exam Patterns & Traps

### ❓ Frequently Tested Concepts

1. **"What problem does MCP solve?"**  
   → The N×M integration problem — standardizes AI ↔ tool connectivity.

2. **"What is the relationship between Host, Client, and Server?"**  
   → Host contains many Clients. Each Client connects to exactly 1 Server (1:1).

3. **"What are the three server-side primitives?"**  
   → Tools (model-controlled), Resources (app-controlled), Prompts (user-controlled).

4. **"When would you use STDIO vs HTTP/SSE?"**  
   → STDIO for local servers, HTTP/SSE for remote/cloud servers.

5. **"How does MCP differ from function calling?"**  
   → MCP is an open standard; function calling is provider-specific. MCP supports dynamic discovery; function calling uses static definitions.

6. **"What messaging protocol does MCP use?"**  
   → JSON-RPC 2.0.

7. **"What authorization does MCP require?"**  
   → OAuth 2.1 with mandatory PKCE for HTTP/SSE transport. STDIO uses process-level security.

### ⚠️ Common Traps

| Trap | Why It's Wrong | Correct Answer |
|------|---------------|----------------|
| "MCP is proprietary to Anthropic" | MCP is open source, governed by Linux Foundation | Open standard, vendor-neutral |
| "Client and Host are the same thing" | Client is a component WITHIN the Host | Host contains 1+ Clients; each Client connects to 1 Server |
| "Resources can modify data" | Resources are READ-ONLY | Only Tools can have side effects |
| "MCP uses REST APIs" | MCP uses JSON-RPC 2.0 (stateful, bidirectional) | Not REST — it's stateful with bidirectional messaging |
| "STDIO needs OAuth" | STDIO is local process communication | Only HTTP/SSE requires OAuth 2.1 |
| "Tools are controlled by the user" | Tools are MODEL-controlled | Model decides when to call tools; User controls Prompts |
| "MCP replaces function calling" | They serve different purposes and can be combined | MCP standardizes the tool ecosystem; function calling is the LLM mechanism |
| "Sampling is a core MCP feature" | Sampling has been deprecated | Focus on Tools, Resources, Prompts |

---

## Quick Revision Flowchart 🧭

```
"Should I use MCP for this project?"

┌─────────────────────────────┐
│ Do you need AI ↔ tool       │
│ integration?                │
└──────────┬──────────────────┘
           │ Yes
           ▼
┌─────────────────────────────┐
│ Will you have MULTIPLE      │──── No ──── Function Calling
│ models or tools?            │             (simpler for 1:1)
└──────────┬──────────────────┘
           │ Yes
           ▼
┌─────────────────────────────┐
│ Need standardization and    │──── No ──── LangChain Tools
│ vendor-neutral reusability? │             (framework-level)
└──────────┬──────────────────┘
           │ Yes
           ▼
      ✅ USE MCP
```

---

> **Final Tip for L2 Exam**: MCP is about **standardization** and **interoperability**. If the exam describes a scenario with multiple AI models needing to access the same tools, MCP is almost always the right answer. Remember: **Host → Client → Server**, **Tools/Resources/Prompts**, **JSON-RPC 2.0**, **STDIO (local) / HTTP+SSE (remote)**.

---

*Study Material Created for GenAI L2 Exam Preparation*  
*Path: `RAG_END2END/MCP/`*
