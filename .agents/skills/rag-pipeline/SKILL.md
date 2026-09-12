---
name: rag-pipeline
description: Build and debug Retrieval-Augmented Generation (RAG) pipelines using LangChain, vector databases (ChromaDB, Pinecone, Qdrant, FAISS), document loaders, text splitters, and retrievers.
---

# RAG Pipeline Skill

Use this skill when the user wants to build, modify, or debug a RAG (Retrieval-Augmented Generation) pipeline.

## Core Components

### 1. Document Loading
Use LangChain document loaders based on file type:
- **PDF**: `PyPDFLoader` or `UnstructuredPDFLoader`
- **DOCX**: `Docx2txtLoader` or `UnstructuredWordDocumentLoader`
- **HTML**: `BSHTMLLoader` (BeautifulSoup)
- **CSV/Excel**: `CSVLoader`, `UnstructuredExcelLoader`
- **JSON**: `JSONLoader`
- **Advanced**: `DoclingLoader` from `langchain-docling`

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("path/to/document.pdf")
documents = loader.load()
```

### 2. Text Splitting / Chunking
Default to `RecursiveCharacterTextSplitter`:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = text_splitter.split_documents(documents)
```

### 3. Embeddings
Prefer these embedding models (in order of recommendation):
1. **HuggingFace** (free): `HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")`
2. **Google** (free): `GoogleGenerativeAIEmbeddings(model="models/embedding-001")`
3. **OpenAI** (paid): `OpenAIEmbeddings(model="text-embedding-3-small")`

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
```

### 4. Vector Store
Choose based on use case:

| Store    | Use Case               | Install                        |
|----------|------------------------|--------------------------------|
| ChromaDB | Local dev, prototyping | `pip install langchain_chroma` |
| FAISS    | Fast local search      | `pip install faiss-cpu`        |
| Pinecone | Cloud/production       | `pip install pinecone langchain_pinecone` |
| Qdrant   | Cloud/production       | `pip install langchain_qdrant` |

```python
from langchain_chroma import Chroma

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
```

### 5. Retriever
```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)
```

### 6. Chain Assembly
```python
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

llm = ChatGroq(model="llama-3.1-8b-instant")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

result = qa_chain.invoke({"query": "What is RAG?"})
```

## Environment Setup
Always load keys from `.env`:
```python
from dotenv import load_dotenv
load_dotenv()
```

## Debugging Tips
- If retrieval returns irrelevant results, try adjusting `chunk_size` and `chunk_overlap`
- Use `search_type="mmr"` for more diverse results
- Check embedding dimensionality matches between your model and vector store
- Use `retriever.get_relevant_documents("test query")` to debug retrieval independently
