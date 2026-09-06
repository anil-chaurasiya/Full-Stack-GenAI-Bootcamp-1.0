"""
Module 08: Prompting for RAG (Debuggable Python Script)
=======================================================
This script mirrors the workflow from 08_Prompting_for_RAG.ipynb.
Covers Zero-shot, Few-shot, Chain-of-Thought, Output Parsers, and Prompt Management.

You can run this directly in VS Code, debug with breakpoints (F5),
or run individual cells using VS Code's Interactive Window (# %%).
"""

import sys
import os
import json
import warnings
from dotenv import load_dotenv

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# %% [1] Setup: Load environment & paths
load_dotenv()

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

print("✅ Setup complete")


# %% [2] Zero-Shot Prompting
zero_shot_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a GenAI expert. Give concise, exam-ready answers."),
    ("human", "{question}"),
])

print("\n📋 Zero-Shot Prompt Template created:")
print(f"   Input variables: {zero_shot_prompt.input_variables}")


# %% [3] Few-Shot Prompting
examples = [
    {
        "input": "What is FAISS?",
        "output": "FAISS (Facebook AI Similarity Search) is a library for efficient similarity search of dense vectors, developed by Meta AI.",
    },
    {
        "input": "What is ChromaDB?",
        "output": "ChromaDB is an open-source embedding database designed for AI applications, offering easy local setup with auto-persistence.",
    },
]

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

final_few_shot = ChatPromptTemplate.from_messages([
    ("system", "You are a GenAI expert. Answer in the same concise style as the examples."),
    few_shot_prompt,
    ("human", "{input}"),
])

print("\n📋 Few-Shot Prompt Template created:")
print(f"   Number of examples: {len(examples)}")


# %% [4] Chain-of-Thought (CoT) Prompting
cot_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a logical reasoning expert. Think step by step before giving your final answer."),
    (
        "human",
        "Question: {question}\n\n"
        "Let's think through this step by step:\n"
        "1. First, identify the key concepts\n"
        "2. Then, analyze the relationships\n"
        "3. Finally, state your conclusion\n\n"
        "Step-by-step reasoning:"
    ),
])

print("\n📋 Chain-of-Thought Prompt Template created")


# %% [5] Ideal RAG Prompt Template (Grounding + Fallback)
rag_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant that answers questions based on the provided context.\n\n"
        "RULES:\n"
        "1. ONLY use information from the context below to answer\n"
        "2. If the context doesn't contain the answer, say 'Based on the provided documents, I don't have enough information.'\n"
        "3. Do NOT make up information\n"
        "4. Be concise but thorough"
    ),
    (
        "human",
        "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    ),
])

print("\n📋 Production RAG Prompt Template created:")
print(f"   Variables: {rag_prompt.input_variables}")


# %% [6] Structured Output: JsonOutputParser
json_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a data extraction assistant.\n"
        "Always respond in valid JSON format with these fields:\n"
        '{{"answer": "your answer", "confidence": "high/medium/low", "key_concepts": ["list", "of", "concepts"]}}'
    ),
    ("human", "{question}"),
])

print("\n📋 JSON Prompt Template created")


# %% [7] External Prompt Management (JSON File)
prompts_dict = {
    "rag_qa": {
        "system": "Answer questions based only on the provided context. Be concise.",
        "template": "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
    },
    "summarize": {
        "system": "You are a summarization expert.",
        "template": "Summarize the following text in {num_sentences} sentences:\n\n{text}",
    },
    "extract": {
        "system": "Extract structured information from text. Respond in JSON.",
        "template": "Text: {text}\n\nExtract the following fields: {fields}",
    },
}

prompts_file_path = os.path.join(DATA_DIR, "prompts.json")
with open(prompts_file_path, "w", encoding="utf-8") as f:
    json.dump(prompts_dict, f, indent=2)

print(f"\n✅ Prompts persisted to {prompts_file_path}")

# Load from file & reconstruct prompt template
with open(prompts_file_path, "r", encoding="utf-8") as f:
    loaded_prompts = json.load(f)

qa_config = loaded_prompts["rag_qa"]
loaded_rag_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_config["system"]),
    ("human", qa_config["template"]),
])

print(f"📋 Loaded prompt from file: {loaded_rag_prompt.input_variables}")


# %% [8] Summary
print("\n" + "=" * 60)
print("✅ Module 8 Complete!")
print("=" * 60)
print("""
Prompt Engineering Best Practices:
1. Always import ChatPromptTemplate and parsers from langchain_core
2. Add explicit grounding rules to prevent hallucination
3. Include a fallback instruction ("If you don't know, say so")
4. Use temperature=0 for factual RAG tasks
5. Persist prompt templates in external JSON/YAML for version control
""")
