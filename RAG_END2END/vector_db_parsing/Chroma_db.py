#!/usr/bin/env python
# coding: utf-8

# | Feature                | FAISS   | ChromaDB |
# | ---------------------- | ------- | -------- |
# | Flat index             | ✅       | ❌        |
# | IVF                    | ✅       | ❌        |
# | HNSW                   | ✅       | ✅        |
# | PQ                     | ✅       | ❌        |
# | Manual index selection | ✅       | ❌        |
# | Metadata filtering     | Limited | ✅        |
# | Persistence            | Manual  | Built-in |
# | LangChain integration  | ✅       | ✅        |
# 

# Final understanding
# FAISS = vector index/search engine
# 
# You manually manage:
# - index type
# - dimension
# - metric
# - docstore
# - ID mapping
# - persistence
#   
# Chroma = vector database
# It manages:
# - vectors
# - documents
# - metadata
# - IDs
# - collections
# - index
# - persistence
# 

# FAISS is a low-level, high-performance library for dense-vector similarity search and clustering. It gives developers direct control over index structures such as Flat, IVF, HNSW and product-quantized indexes, and it offers strong CPU and GPU capabilities. However, FAISS is not a complete vector database: document storage, metadata management, filtering, CRUD APIs, collections, persistence orchestration and server infrastructure generally need to be handled separately.
# 
# Chroma is a retrieval database/search infrastructure designed for AI applications. It stores embeddings together with documents, metadata and IDs, and provides collections, persistence, metadata filtering, full-text and sparse retrieval, CRUD operations and client-server or hosted deployment options. In current Chroma, single-node vector search uses HNSW, while its broader schema and cloud architecture also support other retrieval indexes such as SPANN and sparse/full-text indexes.
# 
# Therefore, FAISS is preferable when low-level index control, custom ANN algorithms, compression or GPU optimization is the priority. Chroma is preferable when building a complete RAG application that needs database-style storage, filtering, updates and operational simplicity.

# | Feature         | FAISS                 | Chroma |
# | --------------- | --------------------- | ------ |
# | Store vectors   | ✅                     | ✅      |
# | Store documents | ❌                     | ✅      |
# | Store metadata  | ❌                     | ✅      |
# | Collections     | ❌                     | ✅      |
# | CRUD            | Limited               | ✅      |
# | Filtering       | ❌ (native)            | ✅      |
# | Persistence     | Basic index save/load | ✅      |
# | Client APIs     | ❌                     | ✅      |
# | Server mode     | ❌                     | ✅      |
# 

# Tumhare code me FAISS ke saath document aur metadata dono store ho rahe the, but crucial point ye hai:
# 
# Unhe native FAISS store nahi kar raha tha; LangChain ka FAISS wrapper store kar raha tha.

# LangChain FAISS VectorStore
# │
# ├── FAISS index
# │     └── Numerical embedding vectors
# │
# ├── InMemoryDocstore
# │     └── LangChain Document objects
# │         ├── page_content
# │         └── metadata
# │
# └── index_to_docstore_id
#       └── FAISS position ko Document ID se map karta hai

# vector_store = FAISS(
#     embedding_function=embeddings,
#     index=faiss_index,
#     docstore=InMemoryDocstore(),
#     index_to_docstore_id={}
# )

# Chroma me document, metadata, ID aur embedding same database collection ke records hain:
# 
# Chroma collection

# collection.add(
#     ids=["chunk-1"],
#     documents=["Llama 2 is a family of language models."],
#     metadatas=[
#         {
#             "source": "llama2.pdf",
#             "page": 5
#         }
#     ],
#     embeddings=[[0.1, 0.2, 0.3]]
# )

# | Component        | LangChain + FAISS             | Chroma                          |
# | ---------------- | ----------------------------- | ------------------------------- |
# | Embeddings       | Native FAISS index            | Chroma vector index             |
# | Documents        | LangChain `Docstore`          | Chroma collection               |
# | Metadata         | LangChain `Document.metadata` | Chroma collection record        |
# | Mapping          | `index_to_docstore_id`        | Internally managed              |
# | Save             | FAISS file + pickle           | Database persistence            |
# | Metadata filters | Wrapper/application handling  | Native database filtering       |
# | Collections      | Not native to FAISS           | Native                          |
# | CRUD             | Wrapper/index-dependent       | Native record operations        |
# | Server/cloud     | Separate system required      | Supported database architecture |
# 

# https://www.trychroma.com/

# In[3]:


from dotenv import load_dotenv
import os

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# In[4]:


os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
from langchain_openai import OpenAIEmbeddings
embeddings=OpenAIEmbeddings(model="text-embedding-3-large")


# In[9]:


# --------------------------------------------------
# 3. Load PDF
# --------------------------------------------------
file_path = r"..\\data\\llama2-research-paper.pdf"
loader = PyPDFLoader(file_path)
pages = loader.load()
print("Total pages:", len(pages))


# In[10]:


# --------------------------------------------------
# 4. Create chunks
# --------------------------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)
chunks = text_splitter.split_documents(pages)
print("Total chunks:", len(chunks))


# In[11]:


# --------------------------------------------------
# 5. Create Chroma vector store
# --------------------------------------------------
vector_store = Chroma(
    collection_name="llama2_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_db_llama2",
    collection_metadata={
        "hnsw:space": "cosine"
    }
)


# In[ ]:


{
    "id": "doc-101",
    "embedding": [0.12, -0.45, 0.78, ...],
    "document": "Employees receive 20 days of annual leave.",
    "metadata": {
        "source": "hr_policy.pdf",
        "page": 5,
        "department": "HR"
    }
}


# In[12]:


# --------------------------------------------------
# 6. Add documents
# --------------------------------------------------

document_ids = vector_store.add_documents(
    documents=chunks
)

print("Documents added:", len(document_ids))

print(
    "Total documents stored:",
    vector_store._collection.count()
)


# In[14]:


# --------------------------------------------------
# 7. Create retriever
# --------------------------------------------------

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 5
    }
)


# In[15]:


# --------------------------------------------------
# 8. Test retriever
# --------------------------------------------------

query = "What is the architecture of Llama 2?"

retrieved_documents = retriever.invoke(query)

for i, document in enumerate(
    retrieved_documents,
    start=1
):
    print(f"\n--- Retrieved document {i} ---")
    print(document.page_content[:500])
    print("Metadata:", document.metadata)


# In[ ]:


# --------------------------------------------------
# 9. Prompt
# --------------------------------------------------

prompt = ChatPromptTemplate.from_template(
    """
    You are a question-answering assistant.

    Answer the question only from the provided context.

    If the context does not contain the answer, say:
    "I do not have enough information in the provided document."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
)


# --------------------------------------------------
# 10. Format documents
# --------------------------------------------------

def format_docs(docs):
    return "\n\n".join(
        f"""
        Source: {doc.metadata.get("source")}
        Page: {doc.metadata.get("page")}

        {doc.page_content}
        """
        for doc in docs
    )


# --------------------------------------------------
# 11. LLM
# --------------------------------------------------

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)


# --------------------------------------------------
# 12. RAG chain
# --------------------------------------------------

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | model
    | StrOutputParser()
)


# --------------------------------------------------
# 13. Ask question
# --------------------------------------------------

answer = rag_chain.invoke(
    "What is the architecture of Llama 2?"
)

print("\nFinal answer:\n")
print(answer)


# In[ ]:


from langchain_chroma import Chroma

loaded_vector_store = Chroma(
    collection_name="llama2_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_db_llama2"
)


# In[ ]:


loaded_retriever = loaded_vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)


# In[ ]:


docs = loaded_retriever.invoke(
    "What is Llama 2?"
)

for doc in docs:
    print(doc.page_content[:500])


# In[ ]:


vector_store.persist()


# In[ ]:


retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 5,
        "filter": {
            "page": 10
        }
    }
)


# In[ ]:


results = vector_store.similarity_search(
    query="What is reinforcement learning?",
    k=5,
    filter={
        "page": 10
    }
)

