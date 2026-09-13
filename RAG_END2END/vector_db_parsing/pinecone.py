#!/usr/bin/env python
# coding: utf-8

# | Feature          | FAISS                          | Chroma                             | Pinecone                                             |
# | ---------------- | ------------------------------ | ---------------------------------- | ---------------------------------------------------- |
# | Product type     | Vector-search library          | Vector database                    | Managed vector database                              |
# | Runs locally     | Yes                            | Yes                                | No normal local database mode                        |
# | Managed cloud    | No                             | Yes                                | Yes, primary mode                                    |
# | Vector index     | Directly controlled            | Database-managed                   | Fully managed                                        |
# | Documents        | LangChain/external store       | Native collection record           | Metadata/document fields through records/integration |
# | Metadata         | External/wrapper               | Native                             | Native                                               |
# | Metadata filters | Not native FAISS DB filtering  | Native                             | Native                                               |
# | CRUD             | Limited/index-dependent        | Native                             | Native                                               |
# | Persistence      | Manual save/load               | Automatic persistent client/server | Managed                                              |
# | Collections      | No native collection primitive | Native collections                 | Indexes and namespaces                               |
# | Scaling          | You design it                  | Local/server/cloud options         | Automatically managed infrastructure                 |
# | Server API       | Build yourself                 | Available                          | Built-in                                             |
# | GPU/index tuning | Strong direct control          | Abstracted                         | Fully abstracted                                     |
# | Cost             | Infrastructure only            | Local free/cloud paid              | Cloud usage-based                                    |
# | Best for         | Research/local/custom ANN      | Local RAG and flexible deployments | Production cloud-scale retrieval                     |
# 

# In[ ]:


import os
import time
from uuid import uuid4
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document


# In[2]:


# ---------------------------------------------------
# 1. Load environment variables
# ---------------------------------------------------

load_dotenv()

google_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

if not google_api_key:
    raise ValueError("OPENAI_API_KEY is missing")

if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY is missing")


# In[3]:


# ---------------------------------------------------
# 2. Embedding model
# ---------------------------------------------------
from langchain_openai import OpenAIEmbeddings
embeddings=OpenAIEmbeddings(model="text-embedding-3-large")
dimension = len(
    embeddings.embed_query("dimension check")
)
print("Embedding dimension:", dimension)


# In[4]:


# ---------------------------------------------------
# 3. Pinecone client
# ---------------------------------------------------
pc = Pinecone(
    api_key=pinecone_api_key
)


# In[6]:


index_name = "langchain-llama-index"


# In[7]:


# ---------------------------------------------------
# 4. Create index
# ---------------------------------------------------

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ),
    )


# In[9]:


description = pc.describe_index(index_name)
description.status["ready"]


# In[8]:


# ---------------------------------------------------
# 5. Wait for index readiness
# ---------------------------------------------------
while True:
    description = pc.describe_index(index_name)
    if description.status["ready"]:
        break
    time.sleep(2)


# In[10]:


# ---------------------------------------------------
# 6. Connect to index
# ---------------------------------------------------
index = pc.Index(index_name)


# In[11]:


# ---------------------------------------------------
# 7. LangChain Pinecone vector store
# ---------------------------------------------------

vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    namespace="demo-documents"
)


# In[12]:


# ---------------------------------------------------
# 8. Create documents
# ---------------------------------------------------

documents = [
    Document(
        page_content=(
            "I had chocolate chip pancakes and "
            "scrambled eggs for breakfast this morning."
        ),
        metadata={"source": "tweet"},
    ),
    Document(
        page_content=(
            "The weather forecast for tomorrow is cloudy "
            "and overcast, with a high of 62 degrees."
        ),
        metadata={"source": "news"},
    ),
    Document(
        page_content=(
            "Building an exciting new project with "
            "LangChain - come check it out!"
        ),
        metadata={"source": "tweet"},
    ),
    Document(
        page_content=(
            "Robbers broke into the city bank and "
            "stole $1 million in cash."
        ),
        metadata={"source": "news"},
    ),
    Document(
        page_content=(
            "LangGraph is the best framework for building "
            "stateful, agentic applications!"
        ),
        metadata={"source": "tweet"},
    ),
]

ids = [
    str(uuid4())
    for _ in documents
]


# In[13]:


# ---------------------------------------------------
# 9. Add documents
# ---------------------------------------------------

inserted_ids = vector_store.add_documents(
    documents=documents,
    ids=ids
)

print("Inserted IDs:", inserted_ids)


# In[14]:


# ---------------------------------------------------
# 10. Similarity search with metadata filter
# ---------------------------------------------------

results = vector_store.similarity_search(
    query=(
        "LangChain provides abstractions "
        "for working with LLMs"
    ),
    k=2,
    filter={
        "source": "tweet"
    }
)

for result in results:
    print(result.page_content)
    print(result.metadata)


# In[16]:


# ---------------------------------------------------
# 11. Retriever with threshold and filter
# ---------------------------------------------------

retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 1,
        "score_threshold": 0.4,
        "filter": {
            "source": "news"
        }
    }
)

retrieved_docs = retriever.invoke(
    "Stealing money from a bank is a crime"
)

print(retrieved_docs)


# In[ ]:


# ---------------------------------------------------
# 12. Delete a document
# ---------------------------------------------------

vector_store.delete(
    ids=[ids[-1]]
)

