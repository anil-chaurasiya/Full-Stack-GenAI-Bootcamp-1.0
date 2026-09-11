"""
Prompting Techniques — Hands-On Practice (GenAI L2 Exam)
=========================================================
This script demonstrates ALL prompting techniques with runnable code.
Use VS Code's Interactive Window (# %%) to run cell by cell, or debug with F5.

Prerequisites:
    pip install langchain-core langchain-groq python-dotenv
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

# %% [1] Setup
load_dotenv()

from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

print("✅ Setup complete — All imports loaded")
print("=" * 60)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TECHNIQUE 1: ZERO-SHOT PROMPTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# %% [2] Zero-Shot Prompting
print("\n📌 TECHNIQUE 1: Zero-Shot Prompting")
print("-" * 40)

zero_shot_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a GenAI expert preparing students for the L2 certification exam."),
    ("human", "{question}"),
])

print(f"Template input variables: {zero_shot_prompt.input_variables}")
print(f"Number of messages: {len(zero_shot_prompt.messages)}")

# Preview the formatted prompt
formatted = zero_shot_prompt.format_messages(
    question="What is the difference between fine-tuning and prompt engineering?"
)
for msg in formatted:
    print(f"  [{msg.type}] {msg.content[:80]}...")

print("""
📝 EXAM NOTE:
   - Zero-shot = NO examples provided
   - Model uses ONLY its pre-trained (parametric) knowledge
   - Best for: simple classification, summarization, general Q&A
   - Lowest token cost of all techniques
""")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TECHNIQUE 2: FEW-SHOT PROMPTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# %% [3] Few-Shot Prompting
print("\n📌 TECHNIQUE 2: Few-Shot Prompting")
print("-" * 40)

# Define examples — quality matters more than quantity!
examples = [
    {
        "input": "What is FAISS?",
        "output": "FAISS (Facebook AI Similarity Search) is a library by Meta AI for "
                  "efficient dense vector similarity search. It supports GPU acceleration "
                  "and various index types (Flat, IVF, HNSW).",
    },
    {
        "input": "What is ChromaDB?",
        "output": "ChromaDB is an open-source embedding database for AI applications. "
                  "It provides easy local setup with auto-persistence and supports "
                  "metadata filtering for hybrid search.",
    },
    {
        "input": "What is LangChain?",
        "output": "LangChain is a framework for building LLM-powered applications. "
                  "It provides abstractions for chains, agents, memory, retrievers, "
                  "and prompt templates via its Expression Language (LCEL).",
    },
]

# Create the example prompt template
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])

# Create the few-shot prompt
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# Wrap in the final template with system message
final_few_shot = ChatPromptTemplate.from_messages([
    ("system", "You are a GenAI expert. Answer in the same concise, "
               "structured style as the examples provided."),
    few_shot_prompt,
    ("human", "{input}"),
])

print(f"Number of examples: {len(examples)}")
print(f"Final template variables: {final_few_shot.input_variables}")

# Preview
formatted = final_few_shot.format_messages(input="What is Pinecone?")
print(f"Total messages after formatting: {len(formatted)}")
for msg in formatted:
    print(f"  [{msg.type}] {msg.content[:70]}...")

print("""
📝 EXAM NOTES:
   - Few-shot = 2-10 examples in the prompt
   - This is IN-CONTEXT LEARNING — no weight updates!
   - Quality of examples > Quantity of examples
   - Examples teach the model FORMAT + STYLE + DOMAIN knowledge
   - Popularized by GPT-3 paper (Brown et al., 2020)
   - Higher token cost than zero-shot
""")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TECHNIQUE 3: CHAIN-OF-THOUGHT (CoT) PROMPTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# %% [4] Chain-of-Thought (CoT) — Few-Shot Version
print("\n📌 TECHNIQUE 3a: Chain-of-Thought (Few-Shot CoT)")
print("-" * 40)

cot_examples = [
    {
        "input": "Roger has 5 tennis balls. He buys 2 more cans of 3 tennis balls each. "
                 "How many tennis balls does he have now?",
        "output": "Let me solve this step by step:\n"
                  "1. Roger starts with 5 tennis balls.\n"
                  "2. He buys 2 cans, each containing 3 balls = 2 × 3 = 6 balls.\n"
                  "3. Total = 5 + 6 = 11 tennis balls.\n"
                  "**Answer: 11 tennis balls**",
    },
    {
        "input": "A store had 23 apples. They used 20 to make juice and bought 6 more. "
                 "How many apples do they have?",
        "output": "Let me solve this step by step:\n"
                  "1. The store starts with 23 apples.\n"
                  "2. They used 20 for juice: 23 - 20 = 3 apples remaining.\n"
                  "3. They bought 6 more: 3 + 6 = 9 apples.\n"
                  "**Answer: 9 apples**",
    },
]

cot_example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])

cot_few_shot = FewShotChatMessagePromptTemplate(
    example_prompt=cot_example_prompt,
    examples=cot_examples,
)

cot_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a math tutor. Solve problems step by step, showing all reasoning."),
    cot_few_shot,
    ("human", "{input}"),
])

print(f"CoT examples: {len(cot_examples)} (each shows reasoning steps)")
formatted = cot_prompt.format_messages(
    input="If there are 3 cars in the parking lot and 2 more arrive, how many are there?"
)
print(f"Formatted messages: {len(formatted)}")

print("""
📝 EXAM NOTES:
   - CoT = examples that SHOW the reasoning process step-by-step
   - Introduced by Wei et al. (2022)
   - Best for: math, logic, multi-step reasoning
   - Works best with LARGE models (>100B parameters)
   - The key difference from regular few-shot: examples include REASONING STEPS
""")


# %% [5] Zero-Shot Chain-of-Thought
print("\n📌 TECHNIQUE 3b: Zero-Shot Chain-of-Thought")
print("-" * 40)

zero_shot_cot_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a logical reasoning expert."),
    ("human", """{question}

Let's think step by step."""),
])

print("The magic phrase: 'Let's think step by step.'")
formatted = zero_shot_cot_prompt.format_messages(
    question="A bat and ball cost $1.10. The bat costs $1 more than the ball. "
             "How much does the ball cost?"
)
for msg in formatted:
    print(f"  [{msg.type}] {msg.content[:80]}...")

print("""
📝 EXAM NOTES:
   - Zero-Shot CoT = NO examples, just add "Let's think step by step."
   - Introduced by Kojima et al. (2022)
   - This SINGLE PHRASE improves accuracy by 10-40% on reasoning tasks!
   - Simplest way to activate reasoning without curating examples
   - KEY EXAM DISTINCTION:
     • CoT (Few-Shot) = examples with reasoning
     • Zero-Shot CoT = no examples, just the magic phrase
""")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TECHNIQUE 4: SELF-CONSISTENCY PROMPTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# %% [6] Self-Consistency (Conceptual Demo)
print("\n📌 TECHNIQUE 4: Self-Consistency Prompting")
print("-" * 40)

print("""
How Self-Consistency Works:
┌───────────────────────────────────────────┐
│  Same Question → LLM (temperature > 0)    │
│                                           │
│  Path 1: Reasoning... → Answer: 42        │
│  Path 2: Reasoning... → Answer: 42        │
│  Path 3: Reasoning... → Answer: 38        │
│  Path 4: Reasoning... → Answer: 42        │
│  Path 5: Reasoning... → Answer: 42        │
│                                           │
│  Majority Vote → Final Answer: 42 ✅      │
└───────────────────────────────────────────┘
""")

# Pseudocode implementation
self_consistency_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a math expert. Solve this step by step."),
    ("human", "{question}"),
])

print("Implementation approach (pseudocode):")
print("""
    answers = []
    for i in range(N):
        response = llm.invoke(prompt, temperature=0.7)  # Non-zero!
        answer = extract_final_answer(response)
        answers.append(answer)
    
    final_answer = most_common(answers)  # Majority vote
""")

print("""
📝 EXAM NOTES:
   - Builds on CoT — generates MULTIPLE reasoning paths
   - Uses MAJORITY VOTING to select final answer
   - Requires temperature > 0 (for diversity in reasoning paths)
   - Introduced by Wang et al. (2022)
   - Trade-off: Higher accuracy but higher compute cost (N × inferences)
""")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TECHNIQUE 5: ROLE / PERSONA PROMPTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# %% [7] Role / Persona Prompting
print("\n📌 TECHNIQUE 5: Role / Persona Prompting")
print("-" * 40)

# Example 1: Technical Expert
expert_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a senior ML engineer with 15 years of experience at Google.
You specialize in:
- Large-scale distributed training
- Transformer architectures
- Production ML systems

When answering, always:
1. Give a precise technical answer
2. Mention relevant papers/tools
3. Highlight common pitfalls"""),
    ("human", "{question}"),
])

# Example 2: Exam Coach
exam_coach_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a GenAI L2 certification exam coach.
Your responses should:
- Be exam-focused and concise
- Highlight what Google Cloud expects
- Use bullet points for key concepts
- Include memory aids and mnemonics when helpful"""),
    ("human", "{question}"),
])

print("Created 2 persona prompts:")
print("  1. Senior ML Engineer (technical depth)")
print("  2. Exam Coach (exam-focused study aid)")

print("""
📝 EXAM NOTES:
   - Role/Persona is set in the SYSTEM message
   - Influences tone, vocabulary, depth, and perspective
   - Can be COMBINED with any other technique (CoT + Role, Few-Shot + Role)
   - Google Vertex AI uses 'System Instruction' for this
""")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TECHNIQUE 6: STRUCTURED OUTPUT PROMPTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# %% [8] Structured Output Prompting
print("\n📌 TECHNIQUE 6: Structured Output Prompting")
print("-" * 40)

# JSON Output
json_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a data extraction assistant.
Always respond in valid JSON format with exactly these fields:
{{
    "answer": "your concise answer",
    "confidence": "high | medium | low",
    "key_concepts": ["concept1", "concept2", "concept3"],
    "exam_relevance": "high | medium | low"
}}"""),
    ("human", "{question}"),
])

# Markdown Table Output
table_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a comparison expert.
Always respond with a markdown table comparing the items.
Include columns: Feature, Option A, Option B, Winner.
End with a one-line summary recommendation."""),
    ("human", "Compare {item_a} vs {item_b} for {use_case}"),
])

print(f"JSON prompt variables: {json_prompt.input_variables}")
print(f"Table prompt variables: {table_prompt.input_variables}")

# LangChain Output Parsers
print("\nLangChain Output Parsers:")
print("  • StrOutputParser()  → Returns raw string")
print("  • JsonOutputParser() → Parses JSON from response")
print("  • PydanticOutputParser() → Validates against Pydantic model")

# Parser chain example (conceptual)
print("\nChain Pattern:")
print("  chain = prompt | llm | JsonOutputParser()")
print("  result = chain.invoke({'question': '...'})")

print("""
📝 EXAM NOTES:
   - Structured output = tell the model EXACTLY what format to use
   - Critical for production systems (downstream code needs parsable output)
   - Gemini supports NATIVE JSON MODE via response_mime_type="application/json"
   - LangChain parsers: StrOutputParser, JsonOutputParser, PydanticOutputParser
   - Always provide the EXACT SCHEMA in the prompt
""")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TECHNIQUE 7: GROUNDED / CONTEXTUAL PROMPTING (RAG)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# %% [9] RAG / Grounded Prompting
print("\n📌 TECHNIQUE 7: Grounded / Contextual Prompting (RAG)")
print("-" * 40)

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant that answers questions based on the provided context.

RULES:
1. ONLY use information from the context below to answer.
2. If the context doesn't contain the answer, say:
   "Based on the provided documents, I don't have enough information to answer this."
3. Do NOT make up or hallucinate information.
4. Cite which part of the context supports your answer.
5. Be concise but thorough."""),
    ("human", """Context:
---
{context}
---

Question: {question}

Answer:"""),
])

print(f"RAG prompt variables: {rag_prompt.input_variables}")

# Preview with sample data
formatted = rag_prompt.format_messages(
    context="ChromaDB is an open-source vector database. It supports local persistence "
            "and metadata filtering. ChromaDB uses HNSW indexing internally.",
    question="What indexing method does ChromaDB use?"
)
for msg in formatted:
    print(f"  [{msg.type}] {msg.content[:80]}...")

print("""
📝 EXAM NOTES — RAG Prompting Checklist:
   ✅ Grounding instruction: "Answer ONLY from context"
   ✅ Fallback: "If not in context, say you don't know"
   ✅ Anti-hallucination: "Do NOT make up information"
   ✅ Context placeholder: {context}
   ✅ Question placeholder: {question}
   ✅ Role in system message
   
   KEY EXAM POINT:
   RAG REDUCES but does NOT ELIMINATE hallucination.
   The model can still misinterpret context or ignore instructions.
""")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TECHNIQUE 8: ReAct (REASONING + ACTING)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# %% [10] ReAct Prompting Pattern
print("\n📌 TECHNIQUE 8: ReAct (Reasoning + Acting)")
print("-" * 40)

react_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant with access to tools.

Available tools:
- search(query): Search the web for information
- calculator(expression): Evaluate math expressions  
- lookup(term): Look up a term in the knowledge base

Follow this pattern for EVERY response:
Thought: [What do I need to figure out?]
Action: [tool_name(arguments)]
Observation: [What did the tool return?]
... (repeat Thought/Action/Observation as needed)
Thought: I now have enough information.
Final Answer: [Your complete answer]"""),
    ("human", "{question}"),
])

print("ReAct Pattern: Thought → Action → Observation → (repeat) → Final Answer")
print(f"Variables: {react_prompt.input_variables}")

print("""
📝 EXAM NOTES:
   - ReAct = Reasoning + Acting — introduced by Yao et al. (2022)
   - The foundation of LANGCHAIN AGENTS and VERTEX AI EXTENSIONS
   - Enables LLMs to USE TOOLS (search, code execution, APIs)
   - Pattern: Thought → Action → Observation → loop
   - KEY: Reduces hallucination by grounding in REAL tool outputs
   - LangChain implementation: AgentExecutor, create_react_agent()
""")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TECHNIQUE 9: PROMPT CHAINING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# %% [11] Prompt Chaining
print("\n📌 TECHNIQUE 9: Prompt Chaining")
print("-" * 40)

# Step 1: Extract key entities
step1_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an entity extraction expert. Extract all named entities."),
    ("human", "Extract entities from:\n{text}\n\nEntities (JSON list):"),
])

# Step 2: Classify entities
step2_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a classification expert."),
    ("human", "Classify these entities by type (person, org, location, tech):\n{entities}"),
])

# Step 3: Generate summary
step3_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a report writer."),
    ("human", "Write a 3-sentence summary using these classified entities:\n{classified}"),
])

print("3-Step Prompt Chain:")
print("  Step 1: Extract entities from text")
print("  Step 2: Classify entities by type")
print("  Step 3: Generate summary report")
print("\nLCEL Chain Pattern:")
print("  chain = step1 | llm | parser | step2 | llm | parser | step3 | llm | parser")

print("""
📝 EXAM NOTES:
   - Prompt Chaining = output of one prompt feeds into the next
   - Each step is simpler and more reliable than one mega-prompt
   - Implemented with LCEL (LangChain Expression Language)
   - Can use DIFFERENT models for different steps (cost optimization)
   - Related to Vertex AI Pipelines and agentic workflows
""")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TECHNIQUE 10: INSTRUCTION PROMPTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# %% [12] Instruction Prompting
print("\n📌 TECHNIQUE 10: Instruction Prompting")
print("-" * 40)

instruction_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a technical writer.

INSTRUCTIONS:
1. Summarize the text in EXACTLY 3 bullet points
2. Each bullet must be ONE sentence only
3. Use simple, non-technical language
4. Do NOT include personal opinions
5. Start each bullet with a verb (e.g., "Explains...", "Describes...")"""),
    ("human", "Summarize this:\n{text}"),
])

print(f"Variables: {instruction_prompt.input_variables}")
print("Key: EXPLICIT, NUMBERED instructions with clear constraints")

print("""
📝 EXAM NOTES:
   - Instruction prompting = giving EXPLICIT step-by-step instructions
   - The foundation of instruction-tuned models (FLAN, InstructGPT, Gemini)
   - Instruction-tuned models follow instructions MUCH better than base models
   - Clear instructions > vague instructions (always be specific)
   - Use numbered lists for multi-step instructions
""")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PROMPT MANAGEMENT — EXTERNAL STORAGE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# %% [13] Prompt Management with JSON
print("\n📌 BONUS: External Prompt Management")
print("-" * 40)

import json

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define a prompt registry
prompt_registry = {
    "zero_shot_qa": {
        "system": "You are a GenAI expert. Answer concisely.",
        "template": "{question}",
        "technique": "zero-shot",
    },
    "rag_qa": {
        "system": "Answer ONLY from context. If unsure, say you don't know.",
        "template": "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
        "technique": "grounded",
    },
    "cot_reasoning": {
        "system": "You are a logical expert. Think step by step.",
        "template": "{question}\n\nLet's think step by step:",
        "technique": "zero-shot-cot",
    },
    "structured_extraction": {
        "system": "Extract info and respond in JSON: "
                  '{{\"answer\": \"\", \"confidence\": \"\", \"sources\": []}}',
        "template": "Text: {text}\n\nExtract: {fields}",
        "technique": "structured-output",
    },
}

# Save to file
prompts_path = os.path.join(CURRENT_DIR, "prompt_registry.json")
with open(prompts_path, "w", encoding="utf-8") as f:
    json.dump(prompt_registry, f, indent=2)
print(f"✅ Saved {len(prompt_registry)} prompts to: prompt_registry.json")

# Load and reconstruct
with open(prompts_path, "r", encoding="utf-8") as f:
    loaded = json.load(f)

for name, config in loaded.items():
    prompt = ChatPromptTemplate.from_messages([
        ("system", config["system"]),
        ("human", config["template"]),
    ])
    print(f"  ✅ Loaded '{name}' ({config['technique']}) → vars: {prompt.input_variables}")

print("""
📝 WHY EXTERNAL PROMPT MANAGEMENT?
   - Version control: Track prompt changes like code
   - A/B testing: Swap prompts without changing code
   - Reuse: Share prompts across services
   - Audit: Log which prompt version produced which output
""")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PARAMETERS DEMO
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# %% [14] Prompt Parameters Cheatsheet
print("\n📌 PARAMETERS CHEATSHEET")
print("=" * 60)

parameters_guide = """
┌─────────────────┬──────────┬─────────────────────────────────────┐
│ Parameter       │ Default  │ When to Change                      │
├─────────────────┼──────────┼─────────────────────────────────────┤
│ temperature     │ 1.0      │ → 0 for factual/RAG tasks           │
│                 │          │ → 0.7+ for creative/brainstorming    │
│                 │          │ → >0 for self-consistency (diversity) │
├─────────────────┼──────────┼─────────────────────────────────────┤
│ top_p           │ 1.0      │ → 0.9 for balanced output            │
│                 │          │ → Don't vary with temperature         │
├─────────────────┼──────────┼─────────────────────────────────────┤
│ top_k           │ varies   │ → 1 for most deterministic            │
│                 │          │ → 40+ for more creative               │
├─────────────────┼──────────┼─────────────────────────────────────┤
│ max_tokens      │ varies   │ → Set based on expected output length │
├─────────────────┼──────────┼─────────────────────────────────────┤
│ stop_sequences  │ []       │ → ["\\n\\n"] to stop at double newline │
├─────────────────┼──────────┼─────────────────────────────────────┤
│ frequency_pen.  │ 0        │ → 0.5+ to reduce repetition          │
└─────────────────┴──────────┴─────────────────────────────────────┘
"""
print(parameters_guide)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FINAL SUMMARY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# %% [15] Summary & Quick Reference
print("\n" + "=" * 60)
print("✅ ALL PROMPTING TECHNIQUES DEMONSTRATED!")
print("=" * 60)

print("""
┌─────────────────────────────────────────────────────────────┐
│           QUICK REFERENCE — DECISION FLOWCHART              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Simple task, no examples needed?    → ZERO-SHOT            │
│  Need consistent format/style?       → FEW-SHOT             │
│  Need step-by-step reasoning?        → CoT or ZERO-SHOT CoT │
│  Need highest accuracy for math?     → SELF-CONSISTENCY      │
│  Need to use external tools?         → ReAct                 │
│  Need specific output format?        → STRUCTURED OUTPUT     │
│  Have retrieved documents?           → RAG/GROUNDED          │
│  Complex multi-step workflow?        → PROMPT CHAINING       │
│  Need specific expertise/tone?       → ROLE/PERSONA          │
│  Need to explore solution space?     → TREE-OF-THOUGHT       │
│                                                             │
└─────────────────────────────────────────────────────────────┘

🎯 L2 EXAM TIP: Focus on WHEN to use each technique, not just WHAT it is.
   Google tests practical decision-making, not just definitions.
""")
