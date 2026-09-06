"""
Module 01: RAG Fundamentals — Qdrant Edition (Debuggable Python Script)
======================================================================
This script mirrors the workflow from 01_RAG_Fundamentals.py but uses
Qdrant (in-memory) as the vector database instead of FAISS.

You can run this directly in VS Code, debug with breakpoints (F5),
or run individual cells using VS Code's Interactive Window (# %%).
"""

import sys
import os
import warnings
from dotenv import load_dotenv

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# %% [1] Setup: Load environment variables
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
print("✅ Environment loaded")
if groq_api_key:
    print(f"🔑 GROQ_API_KEY is found (starts with: {groq_api_key[:8]}...)")
else:
    print("⚠️ GROQ_API_KEY is not set in your .env file!")


# %% [2] Stage 1: LOAD — Create sample documents
from langchain_core.documents import Document

documents = [
    Document(
        page_content="RAG stands for Retrieval-Augmented Generation. It combines retrieval with LLM generation.",
        metadata={"source": "doc1"}
    ),
    Document(
        page_content="Vector databases store embeddings and enable fast similarity search. Examples include FAISS, ChromaDB, and Pinecone.",
        metadata={"source": "doc2"}
    ),
    Document(
        page_content="Fine-tuning modifies model weights on custom data. It's different from RAG which retrieves information at inference time.",
        metadata={"source": "doc3"}
    ),
    Document(
        page_content="Chunking splits documents into smaller pieces. Common strategies include fixed-size, recursive, and semantic chunking.",
        metadata={"source": "doc4"}
    ),
    Document(
        page_content="Embeddings convert text into numerical vectors. Similar texts produce vectors that are close in vector space.",
        metadata={"source": "doc5"}
    ),
]

print(f"\n✅ Loaded {len(documents)} documents")
print(f"📄 Sample document: {documents[0].page_content[:80]}...\n")


# %% [3] Stage 2: EMBED + STORE — Create embeddings and store in Qdrant vector DB
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

print("⏳ Initializing HuggingFace embeddings (all-MiniLM-L6-v2)...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("⏳ Creating Qdrant vector store (in-memory)...")
vectorstore = QdrantVectorStore.from_documents(
    documents=documents,
    embedding=embeddings,
    location=":memory:",
    collection_name="rag_fundamentals",
)

print(f"✅ Created Qdrant vector store with {len(documents)} vectors")
print(f"📐 Embedding dimension: {len(embeddings.embed_query('test'))}\n")


# %% [4] Stage 3: RETRIEVE — Find relevant documents for a query
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

query = "What is the difference between RAG and fine-tuning?"
retrieved_docs = retriever.invoke(query)

print(f"🔍 Query: {query}")
print(f"📄 Retrieved {len(retrieved_docs)} documents:\n")
for i, doc in enumerate(retrieved_docs, 1):
    print(f"  [{i}] {doc.page_content}")
    print(f"      Source: {doc.metadata['source']}\n")


# %% [5] Stage 4: GENERATE — Use LLM with retrieved context
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Supported active models on your Groq key:
# - "openai/gpt-oss-20b"
# - "qwen/qwen3.6-27b"
# - "llama-3.1-8b-instant" (if your key/region allows)
MODEL_NAME = "openai/gpt-oss-20b"

print(f"⏳ Initializing ChatGroq with model: {MODEL_NAME}...")
llm = ChatGroq(model=MODEL_NAME)

# Create RAG prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. Answer the question based ONLY on the provided context. "
        "If the context doesn't contain the answer, say 'I don't have enough information.'"
    ),
    ("human", "Context:\n{context}\n\nQuestion: {question}")
])

# Build the chain using LCEL
chain = prompt | llm | StrOutputParser()

# Format context from retrieved docs
context = "\n".join([doc.page_content for doc in retrieved_docs])

# Generate answer
try:
    print("⏳ Invoking RAG chain...")
    answer = chain.invoke({"context": context, "question": query})
    print(f"\n❓ Question: {query}\n")
    print(f"🤖 Answer: {answer}\n")
except Exception as e:
    print(f"\n⚠️ Error: {e}")
    print("Make sure GROQ_API_KEY is set in your .env file and the model name is accessible.")
