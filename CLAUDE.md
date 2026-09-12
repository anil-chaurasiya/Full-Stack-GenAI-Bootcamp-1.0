# CLAUDE.md — Full-Stack GenAI Bootcamp 1.0

## Project Overview

This repository is a **Full-Stack Generative AI Bootcamp** learning repository containing class materials, assignments, Jupyter notebooks, and Python scripts covering the end-to-end GenAI stack — from encoding & embeddings to transformers, LLM fine-tuning, RAG pipelines, vector databases, and prompt engineering.

**Mentor:** [Sunny Savita](https://sunnysavita10.github.io/)

---

## Repository Structure

```
Full-Stack-GenAI-Bootcamp-1.0/
├── Class-XX-**/              # Numbered class folders with notes, notebooks & code
├── Assignment-XX-**/         # Numbered assignment folders
├── GenAI-Interview-Questions/ # Interview prep material
├── Resumes/                  # Resume templates
├── .env                      # API keys (DO NOT COMMIT — in .gitignore)
├── .env.example              # Template for required API keys
├── requirements.txt          # Core Python dependencies
├── requirements_vectordb.txt # Vector DB specific dependencies
├── main.py                   # Entry-point script
└── CLAUDE.md                 # This file
```

### Class Folder Naming Convention

Folders follow this pattern:
```
Class-{number}-{DD}-{Month}-{Year}-{topic-slug}
```
Example: `Class-37-08-Aug-2026-prompting`

### Assignment Folder Naming Convention

```
Assignment-{number}-{topic-slug}
```

---

## Tech Stack & Dependencies

### Core ML/AI
- **PyTorch** (≥2.4.0), **Torchvision** (≥0.19.0)
- **Transformers** (≥4.41.0), **Datasets** (≥2.19.0)
- **Sentence-Transformers** (2.7.0), **Gensim** (4.3.2)
- **scikit-learn** (1.4.2), **NumPy** (1.26.4), **SciPy** (1.11.4)

### LangChain Ecosystem
- `langchain`, `langchain-community`
- Provider integrations: `langchain-openai`, `langchain-anthropic`, `langchain-google-genai`, `langchain-groq`, `langchain-openrouter`, `langchain-ollama`, `langchain-huggingface`

### Vector Databases
- **ChromaDB** (`langchain_chroma`)
- **Pinecone** (`pinecone`, `langchain_pinecone`)
- **Qdrant** (`langchain_qdrant`)
- **FAISS** (covered in class materials)

### Document Parsing
- `pypdf`, `python-docx`, `docx2txt`, `beautifulsoup4`
- `unstructured`, `docling`, `langchain-docling`
- `pandas`, `openpyxl`, `tabulate`

### Other
- `nltk`, `pyyaml`, `rank-bm25`, `mteb`, `jq`, `ipykernel`, `pillow`

---

## Environment Setup

### Required API Keys (see `.env.example`)

| Key                      | Provider       | Cost   |
|--------------------------|----------------|--------|
| `HUGGINGFACEHUB_API_TOKEN` | Hugging Face  | Free   |
| `GOOGLE_API_KEY`         | Google AI      | Free   |
| `OPENAI_API_KEY`         | OpenAI         | Paid   |
| `ANTHROPIC_API_KEY`      | Anthropic      | Paid   |
| `GROQ_API_KEY`           | Groq           | Free   |
| `OPENROUTER_API_KEY`     | OpenRouter     | Free   |

### Setup Steps
```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

# 2. Install dependencies
pip install -r requirements.txt
pip install -r requirements_vectordb.txt

# 3. Configure environment
cp .env.example .env
# Edit .env and add your API keys
```

---

## Code Conventions

### General
- **Language:** Python 3.10+
- **Notebooks:** Jupyter (`.ipynb`) for class materials and exploratory work
- **Scripts:** Standard `.py` files for utilities and applications
- **Package manager:** pip (with `requirements.txt`)

### Style Guidelines
- Follow **PEP 8** for Python code
- Use **type hints** where practical
- Use **docstrings** (Google style) for functions and classes
- Keep imports organized: stdlib → third-party → local
- Use `python-dotenv` or `os.getenv()` for loading API keys — **never hardcode secrets**

### Notebook Conventions
- Include markdown cells explaining each step
- Clear outputs before committing (when reasonable)
- Use descriptive variable names

---

## Key Topics Covered (by Class)

| Classes   | Topic Area                                      |
|-----------|--------------------------------------------------|
| 1–2       | System Setup & GenAI Introduction                |
| 3–8       | Encoding & Embeddings (OHE, BOW, TF-IDF, Word2Vec, SOTA) |
| 9–14      | Transformer Architecture (Attention, Multi-head, Decoder) |
| 15        | LLM / SLM / MMLLM Understanding                 |
| 16        | LLM Access via APIs                              |
| 18–28     | Fine-Tuning (HF, Unsloth, LoRA, QLoRA, RLHF, DPO) |
| 29–36     | RAG Pipelines (Parsing, Vector DBs, Chunking, Retrieval) |
| 37+       | Prompt Engineering                               |

---

## Important Notes

- **Never commit `.env`** — it contains secrets and is in `.gitignore`
- **Large PDF files** exist in some class folders (handwritten notes) — be mindful of repo size
- **GPU recommended** for fine-tuning classes (18–28); use Google Colab if no local GPU
- When working with vector databases, install from `requirements_vectordb.txt` separately
- Some notebooks may require additional downloads (NLTK data, model weights, etc.)

---

## Commands Reference

```bash
# Run the main script
python main.py

# Launch Jupyter
jupyter notebook

# Install a specific provider
pip install langchain-anthropic

# Run a specific notebook
jupyter notebook Class-37-08-Aug-2026-prompting/prompting.ipynb
```
