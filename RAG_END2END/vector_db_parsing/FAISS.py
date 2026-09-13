#!/usr/bin/env python
# coding: utf-8

# In[55]:


from dotenv import load_dotenv
import os
load_dotenv()


# In[56]:


os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


# In[ ]:


os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
from langchain_openai import OpenAIEmbeddings
embeddings=OpenAIEmbeddings(model="text-embedding-3-large")


# In[60]:


# openai_embeddings_model.embed_query("What is the capital of France?")


# In[11]:


from langchain_google_genai import GoogleGenerativeAIEmbeddings


# In[12]:


embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")


# In[13]:


embeddings.embed_query("Hello world")


# In[14]:


from sklearn.metrics.pairwise import cosine_similarity


# In[15]:


documents=["Washington is the capital of USA",
           "Donald Trump is a president of USA",
           "Narendra Modi is a prime minister of India."]


# In[16]:


my_query  = "Who is a president of USA?"


# In[17]:


embedded_docs = embeddings.embed_documents(documents)
embedded_query = embeddings.embed_query(my_query)


# In[18]:


cosine_similarity([embedded_query], embedded_docs)


# In[19]:


from sklearn.metrics.pairwise import euclidean_distances


# In[20]:


euclidean_distances([embedded_query], embedded_docs)


# In[21]:


import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS


# In[22]:


len(embeddings.embed_query("Hello world"))


# In[23]:


index = faiss.IndexFlatL2(3072)  # 3072 is the dimension of the embeddings


# In[24]:


#this is my vector store 
# inside this vectorstrore the data is being store inside the index
# as of now it is inmemory but we can also store it in disk as well
vector_store = FAISS(
    embedding_function = embeddings,
    index = index,
    docstore = InMemoryDocstore(),
    index_to_docstore_id={}

)


# In[25]:


vector_store.add_texts(["Washington is the capital of USA",
           "Donald Trump is a president of USA",
           "Narendra Modi is a prime minister of India."])


# In[26]:


vector_store.index_to_docstore_id


# In[27]:


faiss_index_id = 1


# In[28]:


docstore_id =vector_store.index_to_docstore_id[faiss_index_id]


# In[29]:


vector_store.docstore.search(docstore_id)


# In[30]:


vector_store.docstore.search(docstore_id).page_content


# In[31]:


vector_store.docstore.search(docstore_id).metadata


# In[32]:


vector_store.index.reconstruct(faiss_index_id)


# In[33]:


vector_store.index.reconstruct(faiss_index_id).shape


# In[34]:


vector_store.similarity_search("Who is a president of USA?", k=1)


# In[35]:


vector_store.similarity_search("Who is a president of USA?", k=2)


# In[37]:


# langchain ->documents -> embeddings -> vectorstore -> index -> faiss


# In[38]:


from uuid import uuid4
from langchain_core.documents import Document

document_1 = Document(
    page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
    metadata={"source": "tweet"},
)

document_2 = Document(
    page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
    metadata={"source": "news"},
)

document_3 = Document(
    page_content="Building an exciting new project with LangChain - come check it out!",
    metadata={"source": "tweet"},
)

document_4 = Document(
    page_content="Robbers broke into the city bank and stole $1 million in cash.",
    metadata={"source": "news"},
)

document_5 = Document(
    page_content="Wow! That was an amazing movie. I can't wait to see it again.",
    metadata={"source": "tweet"},
)

document_6 = Document(
    page_content="Is the new iPhone worth the price? Read this review to find out.",
    metadata={"source": "website"},
)

document_7 = Document(
    page_content="The top 10 soccer players in the world right now.",
    metadata={"source": "website"},
)

document_8 = Document(
    page_content="LangGraph is the best framework for building stateful, agentic applications!",
    metadata={"source": "tweet"},
)

document_9 = Document(
    page_content="The stock market is down 500 points today due to fears of a recession.",
    metadata={"source": "news"},
)

document_10 = Document(
    page_content="I have a bad feeling I am going to get deleted :(",
    metadata={"source": "tweet"},
)


# In[39]:


documents = [
    document_1,
    document_2,
    document_3,
    document_4,
    document_5,
    document_6,
    document_7,
    document_8,
    document_9,
    document_10,
]


# In[40]:


vector_store.add_documents(documents=documents)


# In[41]:


vector_store.similarity_search("LangChain provides abstractions to make working with LLMs easy",
    k=5)


# In[42]:


vector_store.similarity_search("LangChain provides abstractions to make working with LLMs easy",
    k=3,
    filter={"source": "tweet"})


# In[43]:


vector_store.similarity_search("LangChain provides abstractions to make working with LLMs easy",
    k=3,
    filter={"source": "news"})


# In[44]:


#disk persistence of the vectorstore
vector_store.save_local("faiss_index")


# In[45]:


FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)


# | Family | Main idea                 |
# | ------ | ------------------------- |
# | Flat   | Compare with every vector |
# | IVF    | Search selected clusters  |
# | HNSW   | Navigate through a graph  |
# 

# But technically, they are not the only three FAISS categories.
# 
# FAISS also includes:
# 
# PQ
# SQ
# LSH
# Binary indexes
# Residual quantization
# Combined indexes such as IVFPQ
# 
# The most important correction is:
# 
# PQ is not necessarily a replacement for IVF or HNSW. PQ is mainly a vector compression technique.

# The easiest way to understand FAISS is to separate three things:
# 
# 1. How vectors are searched
# 2. How similarity is calculated
# 3. How vectors are stored
# 4. 
# The name can be broken into:
# 
# Index + Flat + IP
# Index: a structure used to store and search vectors
# Flat: checks every vector directly; no clustering or graph
# IP: Inner Product similarity
# 
# Full form of IP
# IP = Inner Product
# 
# Inner Product is also commonly called the dot product.
# 
# For normalized vectors:
# 
# Inner Product ≈ Cosine Similarity
# 
# That is why we normalize embeddings when using IndexFlatIP for cosine similarity.

# Here, HNSW and IVF define the search strategy, while Flat defines how the actual vectors are stored and compared.
# 
# IndexHNSWFlat
# HNSW = search using a graph
# Flat = store complete original vectors
# 
# HNSW first navigates through a graph to find nearby candidates.
# 
# After finding candidates, it compares their full original vectors.
# 
# Therefore:
# 
# IndexHNSWFlat means HNSW graph search with full, uncompressed vectors.
# 
# IndexIVFFlat
# IVF = divide vectors into clusters
# Flat = store complete original vectors inside each cluster
# 
# IVF first finds the most relevant clusters.
# 
# Then it compares the query against the complete vectors stored inside those clusters.
# 
# Therefore:
# 
# IndexIVFFlat means cluster-based search with full, uncompressed vectors inside each cluster.

# In[46]:


# import faiss
# import numpy as np
# faiss_index = faiss.IndexFlatIP(768)


# In[47]:


import faiss
import numpy as np

dimension = 768
M = 32  # Har vector ke approximate graph connections

faiss_index = faiss.IndexHNSWFlat(
    dimension,
    M,
    faiss.METRIC_INNER_PRODUCT
)

# Higher value = better recall, but slower indexing
faiss_index.hnsw.efConstruction = 200

# Higher value = better recall, but slower search
faiss_index.hnsw.efSearch = 64


# In[48]:


from langchain_community.document_loaders import PyPDFLoader


# In[49]:


file_path = "data\\llama2-research-paper.pdf"


# In[50]:


loader = PyPDFLoader(file_path)
# The notebook used ``async for`` at the top level. A normal script uses the
# synchronous loader API so it can run directly from the command line.
pages = loader.load()


# In[51]:


pages


# In[52]:


from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = text_splitter.split_documents(pages)

print("Total pages:", len(pages))
print("Total chunks:", len(chunks))

print(chunks[0].page_content)
print(chunks[0].metadata)


# In[53]:


vector_store = FAISS(
    embedding_function=embeddings,
    index=faiss_index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
    normalize_L2=False
)


# In[54]:


ids = vector_store.add_documents(chunks)

print("Documents added:", len(ids))
print("Total vectors:", vector_store.index.ntotal)


# In[ ]:


retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 5
    }
)


# In[ ]:


query = "What is the architecture of Llama 2?"

retrieved_documents = retriever.invoke(query)

for i, document in enumerate(retrieved_documents, start=1):
    print(f"\n--- Result {i} ---")
    print(document.page_content[:500])
    print("Metadata:", document.metadata)


# In[ ]:


#disk persistence of the vectorstore
vector_store.save_local("faiss_index_llama2")


# In[ ]:


from langchain import hub
prompt = hub.pull("rlm/rag-prompt")

import pprint
pprint.pprint(prompt.messages)


# In[ ]:


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# In[ ]:


from langchain_core.runnables import RunnablePassthrough


# In[ ]:


from langchain_google_genai import ChatGoogleGenerativeAI
model=ChatGoogleGenerativeAI(model='gemini-1.5-flash')


# In[ ]:


# Chain
from langchain_core.output_parsers import StrOutputParser
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)


# In[ ]:


rag_chain.invoke("What is Task Decomposition?")


# In[62]:


from dotenv import load_dotenv
import os
import faiss

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# In[63]:


dimension = len(embeddings.embed_query("dimension test"))
print("Embedding dimension:", dimension)


# In[64]:


file_path = r"data\llama2-research-paper.pdf"

loader = PyPDFLoader(file_path)
pages = loader.load()

print("Total pages:", len(pages))


# In[66]:


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = text_splitter.split_documents(pages)

print("Total chunks:", len(chunks))


# In[67]:


dimension


# In[68]:


M = 32


# In[69]:


faiss_index = faiss.IndexHNSWFlat(
    dimension,
    M,
    faiss.METRIC_L2
)

faiss_index.hnsw.efConstruction = 200
faiss_index.hnsw.efSearch = 64


# In[70]:


vector_store = FAISS(
    embedding_function=embeddings,
    index=faiss_index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
    normalize_L2=False
)


# In[71]:


document_ids = vector_store.add_documents(chunks)

print("Documents added:", len(document_ids))
print("Total vectors:", vector_store.index.ntotal)


# In[72]:


retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)


# In[73]:


retriever.invoke("What is the architecture of Llama 2?")


# In[74]:


# --------------------------------------------------
# 9. Create local RAG prompt
# --------------------------------------------------

prompt = ChatPromptTemplate.from_template(
    """
    You are a question-answering assistant.

    Answer the question only from the supplied context.
    If the answer is not available in the context, say:
    "I do not have enough information in the provided document."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
)


# In[75]:


def format_docs(docs):
    return "\n\n".join(
        f"Source: {doc.metadata}\n{doc.page_content}"
        for doc in docs
    )


# In[76]:


# --------------------------------------------------
# 10. LLM
# --------------------------------------------------

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)


# In[78]:


# --------------------------------------------------
# 11. RAG chain
# --------------------------------------------------

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | model
    | StrOutputParser()
)


# In[79]:


# --------------------------------------------------
# 12. Ask a question related to the PDF
# --------------------------------------------------

answer = rag_chain.invoke(
    "What is the architecture of Llama 2?"
)

print(answer)


# In[80]:


# --------------------------------------------------
# 13. Save vector store
# --------------------------------------------------

vector_store.save_local("faiss_index_llama2")


# In[ ]:




