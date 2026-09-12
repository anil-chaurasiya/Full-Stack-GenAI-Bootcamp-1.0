# Project-Level Agent Rules — Full-Stack GenAI Bootcamp

## General Behavior
- Always load API keys from environment variables (`.env`), never hardcode them.
- When generating Python code, follow PEP 8 and use type hints.
- Prefer LangChain abstractions when building LLM/RAG pipelines — this is a LangChain-heavy project.
- When creating new class or assignment folders, follow the existing naming conventions (e.g., `Class-{number}-{DD}-{Month}-{Year}-{topic-slug}`).

## Notebook Guidelines
- When editing or creating Jupyter notebooks, include explanatory markdown cells.
- Keep code cells focused — one concept per cell.
- Add section headers using markdown `##` in notebooks.

## Security
- Never print or log API keys.
- Always validate that `.env` is in `.gitignore` before any git operations.
- Use `os.getenv("KEY_NAME")` or `dotenv` for secret management.

## Dependencies
- Check `requirements.txt` and `requirements_vectordb.txt` before suggesting new packages.
- If a new dependency is needed, add it to the appropriate requirements file.

## RAG Pipeline Conventions
- Use LangChain document loaders for parsing (PyPDF, Docx, etc.).
- Prefer ChromaDB for local development, Pinecone/Qdrant for production demos.
- Always specify embedding models explicitly — don't rely on defaults.
- Use `RecursiveCharacterTextSplitter` as the default chunking strategy unless a specific alternative is warranted.

## LLM Provider Usage
- Default to free-tier providers (Groq, Google, HuggingFace) for demos.
- Only use paid providers (OpenAI, Anthropic) when specifically requested.
- Always wrap LLM calls in try/except for graceful error handling.
