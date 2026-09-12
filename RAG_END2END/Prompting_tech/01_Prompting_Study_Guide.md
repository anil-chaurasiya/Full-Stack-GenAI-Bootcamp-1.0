# 📚 Prompting Techniques — Complete Study Guide (GenAI L2 Exam)

> **Goal**: Master every prompting concept tested in the Google Cloud GenAI Level-2 exam.
> Read this document end-to-end, then practice with the accompanying notebook.

---

## Table of Contents

1. [What is Prompting?](#1-what-is-prompting)
2. [Why Prompting Matters](#2-why-prompting-matters)
3. [Anatomy of a Prompt](#3-anatomy-of-a-prompt)
4. [Types of Prompting Techniques](#4-types-of-prompting-techniques)
   - 4.1 Zero-Shot Prompting
   - 4.2 One-Shot Prompting
   - 4.3 Few-Shot Prompting
   - 4.4 Chain-of-Thought (CoT) Prompting
   - 4.5 Zero-Shot Chain-of-Thought
   - 4.6 Self-Consistency Prompting
   - 4.7 Tree-of-Thought (ToT) Prompting
   - 4.8 ReAct (Reasoning + Acting)
   - 4.9 Role / Persona Prompting
   - 4.10 Structured Output Prompting
   - 4.11 Instruction Prompting
   - 4.12 Contextual / Grounded Prompting (RAG)
   - 4.13 Multi-Turn / Conversational Prompting
   - 4.14 Prompt Chaining
   - 4.15 Retrieval-Augmented Prompting
   - 4.16 Directional Stimulus Prompting
   - 4.17 Generated Knowledge Prompting
5. [Prompt Engineering Best Practices](#5-prompt-engineering-best-practices)
6. [Prompt Parameters & Their Impact](#6-prompt-parameters--their-impact)
7. [Prompting in RAG Pipelines](#7-prompting-in-rag-pipelines)
8. [Comparison Table — All Techniques at a Glance](#8-comparison-table--all-techniques-at-a-glance)
9. [Common Exam Patterns & Traps](#9-common-exam-patterns--traps)
10. [Key Terminology Glossary](#10-key-terminology-glossary)

---

## 1. What is Prompting?

**Prompting** is the process of providing an input (called a **prompt**) to a Large Language Model (LLM) to guide it toward producing a desired output.

> **Exam Definition**: A prompt is the natural language instruction or question given to a generative AI model that steers its response. Prompt engineering is the systematic design and optimization of prompts to elicit accurate, relevant, and useful outputs from LLMs.

### Key Points for Exam

- A prompt is the **only interface** between a human and an LLM at inference time.
- No model retraining is required — prompting works at **inference time** only.
- Prompt engineering is considered a form of **"in-context learning"** because the model learns from the information provided within the prompt itself, without updating its weights.
- It is the **most cost-effective** way to customize LLM behavior (compared to fine-tuning or training from scratch).

---

## 2. Why Prompting Matters

| Factor | Without Good Prompting | With Good Prompting |
|--------|----------------------|---------------------|
| Accuracy | Vague, hallucinated answers | Precise, grounded responses |
| Format | Unstructured, inconsistent | Structured, predictable |
| Relevance | Off-topic drift | Focused on task |
| Safety | May produce harmful content | Guided by guardrails |
| Cost | Wasted tokens on retries | Efficient first-pass results |

### Exam Relevance
- Google Cloud emphasizes **responsible AI** — good prompting is a key guardrail.
- Prompt engineering is the **first optimization step** before considering fine-tuning.
- Understanding prompting is essential for building **Vertex AI** applications and **RAG** pipelines.

---

## 3. Anatomy of a Prompt

A well-structured prompt typically contains some or all of these components:

```
┌──────────────────────────────────────────────┐
│  1. ROLE / PERSONA (System Instruction)      │
│     "You are a senior data scientist..."     │
├──────────────────────────────────────────────┤
│  2. CONTEXT / BACKGROUND                     │
│     Retrieved documents, domain knowledge    │
├──────────────────────────────────────────────┤
│  3. TASK / INSTRUCTION                       │
│     "Summarize the following...",            │
│     "Answer based on context only..."        │
├──────────────────────────────────────────────┤
│  4. EXAMPLES (Few-Shot)                      │
│     Input → Output pairs                     │
├──────────────────────────────────────────────┤
│  5. INPUT DATA                               │
│     The actual query or text to process      │
├──────────────────────────────────────────────┤
│  6. OUTPUT FORMAT SPECIFICATION              │
│     "Respond in JSON", "Use bullet points"   │
├──────────────────────────────────────────────┤
│  7. CONSTRAINTS / GUARDRAILS                 │
│     "Do NOT hallucinate",                    │
│     "If unsure, say 'I don't know'"          │
└──────────────────────────────────────────────┘
```

### Exam Tip ⚡
Not every prompt needs all 7 components. The **minimum effective prompt** is just a Task + Input. But production-grade prompts (especially for RAG) use 5–7 of these components.

---

## 4. Types of Prompting Techniques

---

### 4.1 Zero-Shot Prompting

**Definition**: Asking the model to perform a task **without any examples**. The model relies entirely on its pre-trained knowledge.

**When to Use**: Simple, well-understood tasks; when the model already knows the domain.

**Example**:
```
Classify the sentiment of this review as Positive, Negative, or Neutral:
"The product arrived on time and works perfectly!"
```

**Strengths**: Simple, fast, no example curation needed.  
**Weaknesses**: May produce inconsistent formats; struggles with complex or niche tasks.

**Exam Key**: Zero-shot = no examples. The model uses **only** its parametric knowledge.

---

### 4.2 One-Shot Prompting

**Definition**: Providing **exactly one example** to demonstrate the expected output format.

**Example**:
```
Classify sentiment:
Review: "Terrible service" → Negative

Review: "Great product, fast delivery!" → ?
```

**Exam Key**: One-shot is a special case of few-shot with k=1.

---

### 4.3 Few-Shot Prompting

**Definition**: Providing **multiple examples** (typically 2–10) to guide the model's behavior and output format.

**When to Use**: When you need consistent formatting, domain-specific behavior, or complex classification.

**Example**:
```
Convert the following to SQL:

Question: "How many users signed up last month?"
SQL: SELECT COUNT(*) FROM users WHERE signup_date >= DATE_SUB(CURDATE(), INTERVAL 1 MONTH);

Question: "Show all orders above $100"
SQL: SELECT * FROM orders WHERE amount > 100;

Question: "List products that are out of stock"
SQL:
```

**Strengths**: Significantly improves consistency and accuracy for structured tasks.  
**Weaknesses**: Uses more tokens (higher cost); example quality directly affects output quality.

**Exam Key Points**:
- Few-shot is a form of **in-context learning (ICL)** — no weight updates.
- The examples are **not** used for training — they only exist in the context window.
- More examples ≠ always better. **Quality > Quantity**.
- Few-shot prompting was introduced/popularized by the **GPT-3 paper** (Brown et al., 2020).

---

### 4.4 Chain-of-Thought (CoT) Prompting

**Definition**: Instructing the model to **show its reasoning process step by step** before arriving at a final answer.

**When to Use**: Math problems, logical reasoning, multi-step analysis, complex Q&A.

**Example (Few-Shot CoT)**:
```
Q: Roger has 5 tennis balls. He buys 2 cans of 3. How many does he have now?
A: Roger starts with 5. He buys 2 cans × 3 = 6 balls. Total = 5 + 6 = 11. 
   The answer is 11.

Q: The cafeteria had 23 apples. They used 20 and bought 6 more. How many now?
A:
```

**Key Insight**: By showing the intermediate reasoning steps in the examples, the model learns to **decompose** complex problems.

**Strengths**: Dramatically improves accuracy on reasoning tasks; makes errors traceable.  
**Weaknesses**: Uses more tokens; slower inference; reasoning can be wrong (hallucinated reasoning).

**Exam Key Points**:
- Introduced by **Wei et al. (2022)** — "Chain-of-Thought Prompting Elicits Reasoning in LLMs".
- Works best with **large models** (>100B parameters). Small models may not benefit.
- CoT is most effective for **arithmetic, commonsense, and symbolic reasoning**.

---

### 4.5 Zero-Shot Chain-of-Thought

**Definition**: Triggering step-by-step reasoning **without providing any examples**, using a simple trigger phrase.

**The Magic Phrase**: `"Let's think step by step."`

**Example**:
```
Q: A bat and a ball cost $1.10 in total. The bat costs $1 more than the ball.
   How much does the ball cost?

Let's think step by step.
```

**Exam Key Points**:
- Introduced by **Kojima et al. (2022)** — "Large Language Models are Zero-Shot Reasoners".
- Simply appending **"Let's think step by step"** improves accuracy by 10–40% on reasoning benchmarks.
- This is the **simplest** way to activate reasoning without curating examples.

---

### 4.6 Self-Consistency Prompting

**Definition**: Generating **multiple reasoning paths** (using CoT) and then selecting the most common/consistent answer via **majority voting**.

**How It Works**:
```
Step 1: Generate N different CoT reasoning paths (with temperature > 0)
Step 2: Extract the final answer from each path
Step 3: Take the majority vote → that's the final answer
```

**Exam Key Points**:
- Introduced by **Wang et al. (2022)**.
- Builds on top of CoT prompting.
- Trades **compute cost** (multiple generations) for **accuracy**.
- Uses **sampling** (temperature > 0) to get diverse reasoning paths.
- The intuition: complex problems may have multiple valid reasoning paths, but the correct answer is reached most often.

---

### 4.7 Tree-of-Thought (ToT) Prompting

**Definition**: An extension of CoT that explores **multiple reasoning branches** (like a tree), evaluates each branch, and backtracks from dead ends.

**How It Works**:
```
Problem → Generate multiple "thoughts" (branches)
        → Evaluate each branch (Is this promising?)
        → Explore further or backtrack
        → Select the best path
```

**Exam Key Points**:
- Introduced by **Yao et al. (2023)**.
- Best for **planning, strategy, and puzzle-solving** tasks.
- More computationally expensive than CoT or Self-Consistency.
- Uses **breadth-first search (BFS)** or **depth-first search (DFS)** over thought branches.

---

### 4.8 ReAct (Reasoning + Acting)

**Definition**: A prompting framework where the model **alternates between reasoning** (thinking about what to do) **and acting** (calling tools, searching, executing code).

**Pattern**:
```
Thought: I need to find the current weather in New York.
Action: search("weather New York today")
Observation: It's 72°F and sunny.
Thought: Now I have the information to answer.
Answer: The current weather in New York is 72°F and sunny.
```

**Exam Key Points**:
- Introduced by **Yao et al. (2022)** — "ReAct: Synergizing Reasoning and Acting in LLMs".
- Core framework behind **LangChain Agents** and **Vertex AI Extensions**.
- Enables LLMs to use **external tools** (search, calculators, APIs, databases).
- The **Thought-Action-Observation** loop is the defining pattern.
- Reduces hallucination by grounding answers in real tool outputs.

---

### 4.9 Role / Persona Prompting

**Definition**: Assigning a **specific role, expertise, or persona** to the model via the system instruction.

**Example**:
```
System: You are a board-certified cardiologist with 20 years of experience. 
        Explain medical concepts in patient-friendly language.

User: What causes high blood pressure?
```

**Exam Key Points**:
- Used in the **system instruction** (system message) of chat models.
- Influences the model's **tone, vocabulary, depth, and perspective**.
- Can be combined with any other technique (CoT, Few-Shot, etc.).

---

### 4.10 Structured Output Prompting

**Definition**: Instructing the model to respond in a **specific format** (JSON, XML, CSV, Markdown table, etc.).

**Example**:
```
Extract the following from the email and respond in JSON:
{
  "sender": "",
  "date": "",
  "subject": "",
  "action_items": []
}

Email: ...
```

**Exam Key Points**:
- Critical for **production pipelines** where downstream code parses the output.
- Use LangChain's `JsonOutputParser`, `PydanticOutputParser`, or `StrOutputParser`.
- Google's Gemini supports **native JSON mode** via response_mime_type.
- Always provide the **exact schema** you expect in the prompt.

---

### 4.11 Instruction Prompting

**Definition**: Giving the model **explicit, detailed instructions** about what to do, without examples.

**Example**:
```
Summarize the following article in exactly 3 bullet points.
Each bullet should be one sentence.
Do not include any opinions.
Focus only on factual claims.

Article: ...
```

**Exam Key Points**:
- The foundation of **instruction-tuned** models (FLAN, InstructGPT, Gemini).
- **Instruction-tuned models** follow instructions much better than base models.
- Clear, specific instructions > vague, general ones.

---

### 4.12 Contextual / Grounded Prompting (RAG)

**Definition**: Providing **retrieved context** in the prompt so the model answers based on **external knowledge** rather than its parametric memory.

**Example**:
```
System: Answer ONLY based on the context provided. If the answer is not in 
        the context, say "I don't have enough information."

Context:
{retrieved_documents}

Question: {user_question}
Answer:
```

**Exam Key Points**:
- The **core prompting pattern** behind Retrieval-Augmented Generation (RAG).
- Reduces **hallucination** by grounding answers in retrieved evidence.
- Must include **fallback instructions** ("If not in context, say so").
- The quality of retrieved context directly affects answer quality (**garbage in, garbage out**).

---

### 4.13 Multi-Turn / Conversational Prompting

**Definition**: Building on **previous messages** in a conversation to maintain context and refine responses.

**Example**:
```
User: What is photosynthesis?
AI: Photosynthesis is the process by which plants convert sunlight...

User: How does it differ in C3 vs C4 plants?
AI: In C3 plants, CO₂ is fixed directly by RuBisCO...

User: Which is more efficient in hot climates?
AI: C4 plants are more efficient in hot, arid climates because...
```

**Exam Key Points**:
- Models maintain context through **conversation history** (message list).
- Each turn adds to the **context window** — can hit token limits.
- Techniques to manage: **summarization**, **sliding window**, **memory modules**.

---

### 4.14 Prompt Chaining

**Definition**: Breaking a complex task into **sequential subtasks**, where the output of one prompt becomes the input to the next.

**Example**:
```
Prompt 1: Extract all entities from this document → [entities]
Prompt 2: For each entity, find related facts → [facts]  
Prompt 3: Synthesize into a summary report → [report]
```

**Exam Key Points**:
- Used in **LangChain LCEL chains** and **Vertex AI pipelines**.
- Each step can use a different model or prompt strategy.
- Improves reliability for complex, multi-step tasks.
- Related to **agentic workflows** and **orchestration**.

---

### 4.15 Retrieval-Augmented Prompting

**Definition**: Dynamically **retrieving relevant examples** for few-shot prompting based on the input query, rather than using static examples.

**How It Works**:
```
User Query → Similarity Search in Example Store → Top-K Similar Examples
→ Inject as Few-Shot Examples → Generate Response
```

**Exam Key Points**:
- Also called **Dynamic Few-Shot** prompting.
- Uses a **vector store** of examples instead of hard-coded ones.
- LangChain provides `SemanticSimilarityExampleSelector` for this.
- Combines the benefits of few-shot prompting with relevance-aware retrieval.

---

### 4.16 Directional Stimulus Prompting

**Definition**: Providing a **hint or nudge** (a "directional stimulus") to guide the model toward a specific aspect of the answer.

**Example**:
```
Summarize this article. 
Hint: Focus on the economic impact and policy recommendations.

Article: ...
```

**Exam Key Points**:
- The "hint" acts as a **steering signal** without being a full instruction.
- Useful for controlling which **aspects** of a topic the model emphasizes.

---

### 4.17 Generated Knowledge Prompting

**Definition**: First asking the model to **generate relevant background knowledge**, then using that knowledge to answer the actual question.

**Example**:
```
Step 1: "Generate 5 key facts about quantum computing."
Step 2: "Using the facts above, explain why quantum supremacy matters."
```

**Exam Key Points**:
- A **two-step** prompting technique.
- Useful when the model's knowledge needs to be **activated** before reasoning.
- Introduced by **Liu et al. (2022)**.

---

## 5. Prompt Engineering Best Practices

### The Golden Rules ⭐

| # | Best Practice | Why It Matters |
|---|--------------|----------------|
| 1 | **Be specific and explicit** | Vague prompts → vague answers |
| 2 | **Use delimiters** (```, """, `<xml>`) | Prevents prompt injection; separates sections clearly |
| 3 | **Specify the output format** | Enables downstream parsing; consistent UX |
| 4 | **Include negative instructions** | "Do NOT hallucinate", "Do NOT include personal opinions" |
| 5 | **Provide a fallback** | "If you don't know, say 'I don't know'" |
| 6 | **Use system instructions** | Sets consistent behavior across all user turns |
| 7 | **Iterate and test** | Test with diverse inputs; edge cases reveal weaknesses |
| 8 | **Keep prompts focused** | One task per prompt; avoid kitchen-sink prompts |
| 9 | **Use temperature wisely** | 0 for factual tasks, 0.7–1.0 for creative tasks |
| 10 | **Version control prompts** | Store in JSON/YAML; track changes like code |

### Exam-Specific Tips ⚡

- **Prompt injection**: Delimiter usage and input validation are the primary defenses.
- **Hallucination reduction**: Grounding (RAG) + explicit constraints + fallback instructions.
- **Token efficiency**: Concise prompts cost less; few-shot costs more than zero-shot.
- **Model selection**: Simpler prompts may work with smaller models; complex CoT needs large models.

---

## 6. Prompt Parameters & Their Impact

| Parameter | Range | Effect | Exam Relevance |
|-----------|-------|--------|----------------|
| **Temperature** | 0.0 – 2.0 | Controls randomness. 0 = deterministic, 1+ = creative | Use 0 for factual RAG, higher for brainstorming |
| **Top-P (Nucleus Sampling)** | 0.0 – 1.0 | Limits token selection to cumulative probability P | Alternative to temperature for controlling diversity |
| **Top-K** | 1 – ∞ | Limits selection to top K most likely tokens | Lower K = more focused, higher K = more diverse |
| **Max Output Tokens** | Model-dependent | Maximum tokens in the response | Controls cost and response length |
| **Stop Sequences** | Strings | Model stops generating when it produces these | Prevents runaway generation |
| **Frequency Penalty** | -2.0 – 2.0 | Penalizes repeated tokens | Reduces repetition |
| **Presence Penalty** | -2.0 – 2.0 | Penalizes tokens that already appeared | Encourages topic diversity |

### Exam Key ⚡

- **Temperature = 0**: Best for factual, deterministic tasks (RAG, classification, extraction).
- **Temperature > 0**: Needed for self-consistency prompting (diverse reasoning paths).
- **Top-P and Temperature** are usually not varied simultaneously — pick one.
- **Max tokens** does NOT affect the quality of reasoning, only the length cutoff.

---

## 7. Prompting in RAG Pipelines

RAG (Retrieval-Augmented Generation) heavily relies on prompt engineering. Here's how prompting fits into the RAG workflow:

```
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐
│  User Query  │────▶│  Retriever    │────▶│  Retrieved Docs   │
└─────────────┘     │  (Vector DB)  │     │  (Context)        │
                    └──────────────┘     └────────┬───────────┘
                                                  │
                                                  ▼
                                    ┌──────────────────────────┐
                                    │  PROMPT TEMPLATE          │
                                    │  ┌────────────────────┐  │
                                    │  │ System: Role +      │  │
                                    │  │         Guardrails  │  │
                                    │  ├────────────────────┤  │
                                    │  │ Context: {docs}     │  │
                                    │  ├────────────────────┤  │
                                    │  │ Question: {query}   │  │
                                    │  └────────────────────┘  │
                                    └──────────┬───────────────┘
                                               │
                                               ▼
                                    ┌──────────────────────┐
                                    │  LLM → Answer         │
                                    └──────────────────────┘
```

### RAG Prompt Must-Haves (Exam Checklist) ✅

1. **Grounding instruction**: "Answer ONLY based on the provided context"
2. **Fallback instruction**: "If the answer is not in the context, say so"
3. **Anti-hallucination rule**: "Do NOT make up information"
4. **Context placeholder**: `{context}` for retrieved documents
5. **Question placeholder**: `{question}` for user query
6. **Role assignment**: Domain expert persona in system message

---

## 8. Comparison Table — All Techniques at a Glance

| Technique | Examples Needed? | Reasoning Shown? | Tool Use? | Best For | Token Cost |
|-----------|:---:|:---:|:---:|---------|:---:|
| Zero-Shot | ❌ | ❌ | ❌ | Simple tasks, classification | Low |
| One-Shot | 1 | ❌ | ❌ | Format demonstration | Low |
| Few-Shot | 2–10 | ❌ | ❌ | Consistent formatting, niche tasks | Medium |
| CoT (Few-Shot) | 2–5 (with reasoning) | ✅ | ❌ | Math, logic, multi-step | Medium |
| Zero-Shot CoT | ❌ | ✅ | ❌ | Quick reasoning boost | Low |
| Self-Consistency | ❌ (uses CoT internally) | ✅ (multiple) | ❌ | High-accuracy reasoning | High |
| Tree-of-Thought | ❌ | ✅ (branching) | ❌ | Planning, puzzles | Very High |
| ReAct | ❌ | ✅ | ✅ | Agent tasks, tool use | High |
| Role/Persona | ❌ | ❌ | ❌ | Tone/expertise control | Low |
| Structured Output | ❌ | ❌ | ❌ | JSON/XML extraction | Low |
| RAG/Grounded | ❌ | ❌ | ❌ | Knowledge-grounded Q&A | Medium |
| Prompt Chaining | ❌ | Varies | Varies | Complex multi-step workflows | High |

---

## 9. Common Exam Patterns & Traps

### ❓ Frequently Tested Concepts

1. **"Which prompting technique is best for reducing hallucination?"**
   → **Grounded/Contextual Prompting (RAG)** — because answers are based on retrieved evidence.

2. **"What does 'in-context learning' mean?"**
   → The model learns from examples provided **in the prompt** at inference time. No weight updates.

3. **"When should you use fine-tuning vs. prompt engineering?"**
   → Prompt engineering first. Fine-tune only if prompt engineering doesn't achieve the quality bar, or if you need to encode domain-specific behavior that's too complex for prompts.

4. **"What is the key difference between CoT and Zero-Shot CoT?"**
   → CoT uses **examples with reasoning steps**. Zero-Shot CoT uses only the trigger phrase **"Let's think step by step"** — no examples.

5. **"What prompting technique do LangChain agents use?"**
   → **ReAct** — Thought-Action-Observation loop.

6. **"How do you ensure consistent output format?"**
   → **Structured Output Prompting** + **Output Parsers** (JsonOutputParser, PydanticOutputParser).

7. **"What is Self-Consistency?"**
   → Generate **multiple CoT paths** → **majority vote** on the final answer.

8. **"Prompt Chaining vs. Single Prompt?"**
   → Chaining is better for complex multi-step tasks because each step is simpler and more reliable.

### ⚠️ Common Traps

| Trap | Why It's Wrong | Correct Answer |
|------|---------------|----------------|
| "Few-shot trains the model" | Few-shot is in-context learning, NOT training | No weight updates occur |
| "More examples always help" | Too many examples waste tokens and may confuse | Quality > Quantity; 3–5 is usually optimal |
| "Temperature=0 means no output" | Temperature=0 means deterministic (greedy decoding) | It always picks the most likely token |
| "CoT works for all model sizes" | CoT benefits emerge primarily in large models (>100B) | Small models may not reason well with CoT |
| "RAG eliminates hallucination" | RAG reduces but doesn't eliminate hallucination | Model can still misinterpret or ignore context |

---

## 10. Key Terminology Glossary

| Term | Definition |
|------|-----------|
| **Prompt** | The input text provided to an LLM to elicit a response |
| **Prompt Engineering** | The practice of designing and optimizing prompts for better LLM outputs |
| **In-Context Learning (ICL)** | Learning from examples within the prompt at inference time (no training) |
| **System Instruction** | The "system" message that sets the model's role, behavior, and constraints |
| **Grounding** | Anchoring model responses to specific provided evidence/context |
| **Hallucination** | When a model generates plausible but factually incorrect information |
| **Token** | The basic unit of text processed by an LLM (~4 characters in English) |
| **Context Window** | The maximum number of tokens the model can process in one request |
| **Inference** | The process of generating output from a trained model (no learning) |
| **Temperature** | Parameter controlling randomness/creativity in generation |
| **Top-P / Nucleus Sampling** | Sampling from the smallest set of tokens whose cumulative probability ≥ P |
| **Guardrails** | Constraints in prompts to prevent unwanted outputs (safety, accuracy) |
| **Output Parser** | Code that converts model output into structured data (JSON, objects) |
| **LCEL** | LangChain Expression Language — declarative way to compose chains |
| **Agentic Prompting** | Prompting that enables models to plan, use tools, and act autonomously |

---

## Quick Revision Flowchart 🧭

```
                    ┌─────────────────────┐
                    │  What is the task?   │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
         Simple Q&A     Needs Reasoning    Needs Tools
              │                │                │
         Zero-Shot         CoT / ZS-CoT      ReAct
              │                │                │
     Need consistency?   Need accuracy?    Need multi-step?
         │        │         │                   │
        No       Yes     Self-Consistency   Prompt Chaining
         │        │                             │
     Done!    Few-Shot                      Agentic Loop
```

---

> **Final Tip for L2 Exam**: Google emphasizes **practical application** — know WHEN to use each technique, not just WHAT they are. Focus on the decision-making process: "Given this scenario, which prompting approach is most appropriate?"

---

*Study Material Created for GenAI L2 Exam Preparation*  
*Path: `RAG_END2END/Prompting_tech/`*
