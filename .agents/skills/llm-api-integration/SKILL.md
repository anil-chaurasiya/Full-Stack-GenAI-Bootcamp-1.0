---
name: llm-api-integration
description: Integrate and use LLM APIs from various providers (OpenAI, Anthropic, Google, Groq, OpenRouter, Ollama, HuggingFace) using LangChain wrappers. Handle API keys, model selection, streaming, and error handling.
---

# LLM API Integration Skill

Use this skill when the user wants to connect to, switch between, or troubleshoot LLM provider APIs.

## Supported Providers

### Free Tier Providers (Prefer These for Demos)

#### Groq
```python
from langchain_groq import ChatGroq

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)
response = llm.invoke("Hello!")
```

#### Google Generative AI
```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
response = llm.invoke("Hello!")
```

#### HuggingFace
```python
from langchain_huggingface import HuggingFaceEndpoint

llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2")
response = llm.invoke("Hello!")
```

#### OpenRouter
```python
from langchain_openrouter import ChatOpenRouter

llm = ChatOpenRouter(model="meta-llama/llama-3.1-8b-instruct:free")
response = llm.invoke("Hello!")
```

#### Ollama (Local)
```python
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.1", temperature=0.7)
response = llm.invoke("Hello!")
```

### Paid Providers

#### OpenAI
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
response = llm.invoke("Hello!")
```

#### Anthropic
```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0.7)
response = llm.invoke("Hello!")
```

## Environment Setup

All providers read API keys from environment variables automatically:
```python
import os
from dotenv import load_dotenv
load_dotenv()

# Keys are auto-detected by LangChain:
# GROQ_API_KEY, GOOGLE_API_KEY, OPENAI_API_KEY,
# ANTHROPIC_API_KEY, HUGGINGFACEHUB_API_TOKEN, OPENROUTER_API_KEY
```

## Common Patterns

### Streaming
```python
for chunk in llm.stream("Tell me a story"):
    print(chunk.content, end="", flush=True)
```

### With Prompt Templates
```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    ("human", "{input}")
])

chain = prompt | llm
response = chain.invoke({"input": "Explain transformers"})
```

### Error Handling
```python
try:
    response = llm.invoke("Hello!")
except Exception as e:
    print(f"LLM API error: {e}")
    # Fallback to a different provider
    fallback_llm = ChatGroq(model="llama-3.1-8b-instant")
    response = fallback_llm.invoke("Hello!")
```

## Provider Selection Guide

| Need                  | Recommended Provider        |
|-----------------------|-----------------------------|
| Free & fast           | Groq (Llama 3.1)           |
| Free & multimodal     | Google (Gemini)             |
| Best quality (paid)   | OpenAI (GPT-4o) / Anthropic (Claude) |
| Local/offline         | Ollama                      |
| Model variety         | OpenRouter / HuggingFace   |
