"""
Module 07: RAG Chain — End to End (Debuggable Python Script)
============================================================
This script mirrors the workflow from 07_RAG_Chain_End2End.ipynb.
Demonstrates the full 6-stage RAG pipeline using modern LangChain (LCEL).

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

# %% [1] Setup: Load environment variables & paths
load_dotenv()

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

print("✅ Environment loaded")


# %% [2] STEP 1: LOAD — Document Ingestion
# ═══════════════════════════════════════════════════════════
from langchain_community.document_loaders import TextLoader

sample_txt_path = os.path.join(DATA_DIR, "sample.txt")
if not os.path.exists(sample_txt_path):
    with open(sample_txt_path, "w", encoding="utf-8") as f:
        f.write(
            "RAG (Retrieval-Augmented Generation) combines information retrieval with LLM generation.\n\n"
            "Large Language Models (LLMs) are deep learning models trained on vast corpora of text.\n\n"
            "Machine Learning is the broader field, whereas Deep Learning uses multi-layered neural networks."
        )

loader = TextLoader(sample_txt_path, encoding="utf-8")
documents = loader.load()

print(f"\n📄 Step 1 — LOAD: {len(documents)} document(s) loaded")


# %% [3] STEP 2: CHUNK — Split into smaller pieces
# ═══════════════════════════════════════════════════════════
# Modern LangChain standard: import from langchain_text_splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],
)

chunks = text_splitter.split_documents(documents)
print(f"📝 Step 2 — CHUNK: {len(documents)} doc → {len(chunks)} chunks")


# %% [4] STEP 3 & 4: EMBED + STORE — Create vector store
# ═══════════════════════════════════════════════════════════
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print(f"🔢 Step 3 — EMBED: Using all-MiniLM-L6-v2 ({len(embeddings.embed_query('test'))}d)")
    print(f"🗄️ Step 4 — STORE: {len(chunks)} vectors stored in FAISS")

    # %% [5] STEP 5: RETRIEVE — Create retriever
    # ═══════════════════════════════════════════════════════════
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    test_query = "What is RAG?"
    test_results = retriever.invoke(test_query)
    print(f"\n🔍 Step 5 — RETRIEVE: Top {len(test_results)} results for '{test_query}'")
    for i, doc in enumerate(test_results, 1):
        print(f"   [{i}] {doc.page_content[:80]}...")

    # %% [6] STEP 6: GENERATE — Modern LCEL Chain
    # ═══════════════════════════════════════════════════════════
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough, RunnableParallel

    # Prompt with grounding rules
    rag_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant that answers questions based on the provided context.\n"
            "Rules:\n"
            "- Only use information from the context below\n"
            "- If the context doesn't contain the answer, say 'I don't have enough information'\n"
            "- Be concise and accurate"
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        ),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Note: If langchain_groq is installed and GROQ_API_KEY is present, we invoke live LLM.
    try:
        from langchain_groq import ChatGroq

        llm = ChatGroq(model="openai/gpt-oss-20b")

        # Pure modern LCEL chain:
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | rag_prompt
            | llm
            | StrOutputParser()
        )

        print("\n✅ Step 6 — Modern LCEL RAG Chain built!")
        print("🎯 Flow: Input Question → (Retriever | format_docs) + Question → Prompt → LLM → OutputParser")

        # Test the chain
        print(f"\n❓ Asking: '{test_query}'")
        answer = rag_chain.invoke(test_query)
        print(f"🤖 Answer: {answer}")

        # LCEL with Source Attribution (RunnableParallel)
        rag_chain_with_sources = RunnableParallel(
            {
                "answer": (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | rag_prompt
                    | llm
                    | StrOutputParser()
                ),
                "sources": retriever,
            }
        )

        result = rag_chain_with_sources.invoke("What is machine learning?")
        print(f"\n🤖 Answer with Sources: {result['answer']}")
        print(f"📚 Sources count: {len(result['sources'])}")

    except ImportError:
        print("\nℹ️ langchain_groq not installed. Run: pip install langchain-groq")
    except Exception as e:
        print(f"\nℹ️ LLM generation info: {e}")

except ImportError:
    print("⚠️ Required packages not installed. Run: pip install langchain-huggingface sentence-transformers faiss-cpu")


# %% [7] Summary & Modern LangChain Standards
print("\n" + "=" * 60)
print("✅ Module 7 Complete!")
print("=" * 60)
print("""
Modern LangChain Architecture:
1. Pure LCEL (LangChain Expression Language) replaces legacy RetrievalQA
2. Pipe operator (|) composes runnables: format → prompt → llm → parser
3. RunnablePassthrough() forwards the user question
4. RunnableParallel() enables returning answer + source documents simultaneously
5. Always import splitters from langchain_text_splitters
""")
