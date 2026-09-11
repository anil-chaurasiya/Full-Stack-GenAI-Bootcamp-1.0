# 🧠 Prompting Techniques — Practice MCQs (GenAI L2 Exam)
# =========================================================
# 50 Multiple Choice Questions covering all prompting concepts.
# Run this script to take a self-assessment quiz.

import random

QUESTIONS = [
    # ── BASIC CONCEPTS ──────────────────────────────────────
    {
        "q": "What is 'prompting' in the context of Generative AI?",
        "options": [
            "A) Training the model on new data",
            "B) Providing input text to guide an LLM's response at inference time",
            "C) Fine-tuning the model weights",
            "D) Updating the model's vocabulary",
        ],
        "answer": "B",
        "explanation": "Prompting is the process of providing input to an LLM at inference time to guide its output. No weight updates occur.",
    },
    {
        "q": "What is 'prompt engineering'?",
        "options": [
            "A) Building the model architecture",
            "B) The systematic design and optimization of prompts to improve LLM outputs",
            "C) Training prompts as learnable parameters",
            "D) Converting prompts to embeddings",
        ],
        "answer": "B",
        "explanation": "Prompt engineering is the practice of designing, testing, and optimizing prompts to elicit better responses from LLMs.",
    },
    {
        "q": "Why is prompt engineering considered 'in-context learning'?",
        "options": [
            "A) Because the model's weights are updated during prompting",
            "B) Because the model learns from examples provided in the prompt without weight updates",
            "C) Because the context window stores training data",
            "D) Because the model downloads new knowledge",
        ],
        "answer": "B",
        "explanation": "In-context learning means the model 'learns' from information in the prompt (examples, instructions) at inference time without any parameter updates.",
    },
    {
        "q": "Which is the MOST cost-effective way to customize LLM behavior?",
        "options": [
            "A) Training from scratch",
            "B) Fine-tuning",
            "C) Prompt engineering",
            "D) Reinforcement learning from human feedback (RLHF)",
        ],
        "answer": "C",
        "explanation": "Prompt engineering requires no training, no data collection, and no compute for model updates. It's the cheapest and fastest way to customize behavior.",
    },

    # ── ZERO-SHOT ───────────────────────────────────────────
    {
        "q": "In zero-shot prompting, how many examples are provided to the model?",
        "options": [
            "A) 1",
            "B) 2-5",
            "C) 0 — no examples at all",
            "D) As many as fit in the context window",
        ],
        "answer": "C",
        "explanation": "Zero-shot means zero examples. The model relies entirely on its pre-trained knowledge to understand and complete the task.",
    },
    {
        "q": "When is zero-shot prompting MOST appropriate?",
        "options": [
            "A) Complex mathematical reasoning",
            "B) Simple, well-understood tasks like basic classification",
            "C) Tasks requiring very specific output formats",
            "D) Multi-step planning problems",
        ],
        "answer": "B",
        "explanation": "Zero-shot works best for straightforward tasks where the model's pre-training knowledge is sufficient.",
    },

    # ── FEW-SHOT ────────────────────────────────────────────
    {
        "q": "What is few-shot prompting?",
        "options": [
            "A) Training the model on a few data points",
            "B) Providing 2-10 examples in the prompt to guide format and behavior",
            "C) Using a small model for inference",
            "D) Running the model for only a few iterations",
        ],
        "answer": "B",
        "explanation": "Few-shot prompting provides multiple input-output examples in the prompt to teach the model the expected format and reasoning style.",
    },
    {
        "q": "Which paper popularized few-shot prompting?",
        "options": [
            "A) Attention Is All You Need (2017)",
            "B) BERT: Pre-training of Bidirectional Transformers (2018)",
            "C) Language Models are Few-Shot Learners — GPT-3 (Brown et al., 2020)",
            "D) Chain-of-Thought Prompting (Wei et al., 2022)",
        ],
        "answer": "C",
        "explanation": "The GPT-3 paper by Brown et al. (2020) demonstrated that large language models can perform few-shot learning through examples in the prompt.",
    },
    {
        "q": "In few-shot prompting, does the model's weights get updated?",
        "options": [
            "A) Yes, through backpropagation",
            "B) Yes, through gradient-free optimization",
            "C) No — it's in-context learning with no weight updates",
            "D) Only the embedding layer is updated",
        ],
        "answer": "C",
        "explanation": "Few-shot prompting is in-context learning. The examples exist only in the prompt; no model parameters are modified.",
    },
    {
        "q": "What matters MORE in few-shot prompting — quality or quantity of examples?",
        "options": [
            "A) Quantity — always use as many examples as possible",
            "B) Quality — well-chosen examples are more impactful than many poor ones",
            "C) Neither — the model ignores examples",
            "D) Both are equally unimportant",
        ],
        "answer": "B",
        "explanation": "Quality > Quantity. 3 well-crafted, diverse examples often outperform 10 repetitive or low-quality ones.",
    },

    # ── CHAIN-OF-THOUGHT ────────────────────────────────────
    {
        "q": "What is Chain-of-Thought (CoT) prompting?",
        "options": [
            "A) Connecting multiple LLMs in a chain",
            "B) Instructing the model to show step-by-step reasoning before the final answer",
            "C) Using a chain of vector databases",
            "D) Linking prompts through a database",
        ],
        "answer": "B",
        "explanation": "CoT prompting guides the model to decompose problems into intermediate reasoning steps before arriving at the final answer.",
    },
    {
        "q": "Who introduced Chain-of-Thought prompting?",
        "options": [
            "A) Brown et al. (2020)",
            "B) Vaswani et al. (2017)",
            "C) Wei et al. (2022)",
            "D) Radford et al. (2019)",
        ],
        "answer": "C",
        "explanation": "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models was introduced by Wei et al. (2022).",
    },
    {
        "q": "CoT prompting works best with which type of models?",
        "options": [
            "A) Small models (< 1B parameters)",
            "B) Medium models (1-10B parameters)",
            "C) Large models (> 100B parameters)",
            "D) Any size model equally",
        ],
        "answer": "C",
        "explanation": "CoT benefits emerge primarily in large models (>100B parameters). Smaller models may not have sufficient reasoning capability to benefit from CoT.",
    },
    {
        "q": "What types of tasks benefit MOST from CoT prompting?",
        "options": [
            "A) Simple classification and sentiment analysis",
            "B) Arithmetic, commonsense reasoning, and symbolic reasoning",
            "C) Text summarization",
            "D) Language translation",
        ],
        "answer": "B",
        "explanation": "CoT is most effective for tasks requiring multi-step reasoning: math problems, logical reasoning, and commonsense inference.",
    },

    # ── ZERO-SHOT CoT ───────────────────────────────────────
    {
        "q": "What is the 'magic phrase' for Zero-Shot Chain-of-Thought?",
        "options": [
            "A) 'Give me the answer'",
            "B) 'Let's think step by step.'",
            "C) 'Show your work'",
            "D) 'Explain in detail'",
        ],
        "answer": "B",
        "explanation": "Kojima et al. (2022) showed that simply appending 'Let's think step by step.' to a prompt improves reasoning accuracy by 10-40%.",
    },
    {
        "q": "What is the KEY difference between CoT and Zero-Shot CoT?",
        "options": [
            "A) CoT uses a different model",
            "B) CoT provides examples with reasoning; Zero-Shot CoT uses only a trigger phrase",
            "C) Zero-Shot CoT is more accurate",
            "D) They are exactly the same technique",
        ],
        "answer": "B",
        "explanation": "CoT (few-shot) provides worked examples showing reasoning steps. Zero-Shot CoT has no examples — just the trigger phrase 'Let's think step by step.'",
    },

    # ── SELF-CONSISTENCY ────────────────────────────────────
    {
        "q": "How does Self-Consistency prompting work?",
        "options": [
            "A) The model checks its own grammar",
            "B) Generate multiple CoT reasoning paths, then majority-vote on the final answer",
            "C) The model validates against a database",
            "D) Use one reasoning path but verify it twice",
        ],
        "answer": "B",
        "explanation": "Self-Consistency generates N diverse reasoning paths (using temperature > 0) and selects the most frequent final answer via majority voting.",
    },
    {
        "q": "What temperature setting is needed for Self-Consistency?",
        "options": [
            "A) Temperature = 0 (deterministic)",
            "B) Temperature > 0 (to get diverse reasoning paths)",
            "C) Temperature = -1",
            "D) Temperature doesn't matter",
        ],
        "answer": "B",
        "explanation": "Self-Consistency requires diverse reasoning paths, which are generated by sampling with temperature > 0. Temperature = 0 would give the same path every time.",
    },

    # ── TREE-OF-THOUGHT ─────────────────────────────────────
    {
        "q": "How does Tree-of-Thought (ToT) differ from Chain-of-Thought?",
        "options": [
            "A) ToT uses a single linear reasoning path",
            "B) ToT explores multiple branching reasoning paths and can backtrack from dead ends",
            "C) ToT only works with small models",
            "D) ToT doesn't involve reasoning",
        ],
        "answer": "B",
        "explanation": "ToT creates a tree of possible reasoning branches, evaluates each, and can backtrack — unlike CoT which follows a single linear chain.",
    },
    {
        "q": "Tree-of-Thought is best suited for which type of tasks?",
        "options": [
            "A) Simple Q&A",
            "B) Text summarization",
            "C) Planning, strategy, and puzzle-solving",
            "D) Sentiment analysis",
        ],
        "answer": "C",
        "explanation": "ToT excels at tasks requiring exploration and backtracking — strategic planning, puzzles (like Game of 24), and creative problem-solving.",
    },

    # ── ReAct ───────────────────────────────────────────────
    {
        "q": "What does ReAct stand for?",
        "options": [
            "A) Reactive Actions",
            "B) Reasoning + Acting",
            "C) Real-time Active Computing",
            "D) Recursive Action Training",
        ],
        "answer": "B",
        "explanation": "ReAct combines Reasoning (thinking about what to do) with Acting (executing tools) in an interleaved Thought-Action-Observation loop.",
    },
    {
        "q": "What is the defining pattern of ReAct prompting?",
        "options": [
            "A) Input → Output",
            "B) Thought → Action → Observation → (repeat) → Final Answer",
            "C) Question → Answer → Verification",
            "D) Query → Retrieve → Generate",
        ],
        "answer": "B",
        "explanation": "The Thought-Action-Observation loop is ReAct's signature pattern, enabling the model to reason about what tool to use and incorporate tool results.",
    },
    {
        "q": "Which LangChain feature is based on the ReAct prompting pattern?",
        "options": [
            "A) Document Loaders",
            "B) Text Splitters",
            "C) Agents (AgentExecutor, create_react_agent)",
            "D) Embeddings",
        ],
        "answer": "C",
        "explanation": "LangChain Agents use the ReAct pattern to reason about which tools to use, execute them, and incorporate results into their responses.",
    },

    # ── ROLE/PERSONA ────────────────────────────────────────
    {
        "q": "Where is the role/persona typically specified in a prompt?",
        "options": [
            "A) In the user message",
            "B) In the system instruction (system message)",
            "C) In the output parser",
            "D) In the model configuration file",
        ],
        "answer": "B",
        "explanation": "The role/persona is set in the system message (system instruction), which sets the model's behavior, tone, and expertise for all subsequent turns.",
    },

    # ── STRUCTURED OUTPUT ───────────────────────────────────
    {
        "q": "Which LangChain parser validates output against a Pydantic model?",
        "options": [
            "A) StrOutputParser",
            "B) JsonOutputParser",
            "C) PydanticOutputParser",
            "D) XMLOutputParser",
        ],
        "answer": "C",
        "explanation": "PydanticOutputParser validates LLM output against a Pydantic schema, ensuring type safety and structure in the parsed output.",
    },
    {
        "q": "How does Google Gemini support native JSON output?",
        "options": [
            "A) Through a special model variant",
            "B) By setting response_mime_type='application/json'",
            "C) Through fine-tuning only",
            "D) JSON is the default output format",
        ],
        "answer": "B",
        "explanation": "Gemini supports native JSON mode by setting response_mime_type='application/json' in the generation config.",
    },

    # ── RAG / GROUNDED ──────────────────────────────────────
    {
        "q": "What is the PRIMARY purpose of grounded/contextual prompting in RAG?",
        "options": [
            "A) To make the model generate longer responses",
            "B) To anchor answers in retrieved evidence and reduce hallucination",
            "C) To speed up inference",
            "D) To reduce the model size",
        ],
        "answer": "B",
        "explanation": "Grounded prompting provides retrieved context so the model answers from evidence rather than its parametric memory, reducing hallucination.",
    },
    {
        "q": "Which of these is a MUST-HAVE in a production RAG prompt?",
        "options": [
            "A) Few-shot examples",
            "B) A fallback instruction ('If answer not in context, say so')",
            "C) Temperature = 1.0",
            "D) Multiple system messages",
        ],
        "answer": "B",
        "explanation": "A fallback instruction is critical to prevent the model from hallucinating an answer when the retrieved context doesn't contain relevant information.",
    },
    {
        "q": "Does RAG completely eliminate hallucination?",
        "options": [
            "A) Yes, RAG guarantees factual accuracy",
            "B) No — RAG reduces but does not eliminate hallucination",
            "C) Only if temperature = 0",
            "D) Only with few-shot examples",
        ],
        "answer": "B",
        "explanation": "RAG significantly reduces hallucination by grounding answers in context, but the model can still misinterpret context or ignore grounding instructions.",
    },

    # ── PROMPT CHAINING ─────────────────────────────────────
    {
        "q": "What is prompt chaining?",
        "options": [
            "A) Using the same prompt repeatedly",
            "B) Breaking a complex task into sequential subtasks where one output feeds into the next",
            "C) Connecting multiple models in parallel",
            "D) Chaining vector databases together",
        ],
        "answer": "B",
        "explanation": "Prompt chaining decomposes complex tasks into a pipeline of simpler prompts, where each step's output becomes the next step's input.",
    },
    {
        "q": "What is the LangChain syntax for prompt chaining?",
        "options": [
            "A) chain = prompt + llm + parser",
            "B) chain = prompt | llm | parser (LCEL pipe operator)",
            "C) chain = prompt.run(llm, parser)",
            "D) chain = connect(prompt, llm, parser)",
        ],
        "answer": "B",
        "explanation": "LCEL (LangChain Expression Language) uses the pipe operator (|) to chain components: prompt | llm | parser.",
    },

    # ── PARAMETERS ──────────────────────────────────────────
    {
        "q": "What does temperature = 0 mean for LLM generation?",
        "options": [
            "A) The model produces no output",
            "B) Deterministic output — always picks the most likely token (greedy decoding)",
            "C) The model generates random gibberish",
            "D) The model runs in evaluation mode",
        ],
        "answer": "B",
        "explanation": "Temperature = 0 means greedy decoding — the model always selects the highest-probability token, producing deterministic, consistent output.",
    },
    {
        "q": "For a factual RAG pipeline, what temperature should you use?",
        "options": [
            "A) Temperature = 1.0 (default)",
            "B) Temperature = 0 (deterministic, factual)",
            "C) Temperature = 2.0 (maximum creativity)",
            "D) Temperature = 0.5 (balanced)",
        ],
        "answer": "B",
        "explanation": "Factual tasks like RAG Q&A benefit from temperature = 0 because you want consistent, deterministic answers based on the context.",
    },
    {
        "q": "What does Top-P (nucleus sampling) control?",
        "options": [
            "A) The number of layers in the model",
            "B) The cumulative probability threshold for token selection",
            "C) The number of parameters",
            "D) The learning rate",
        ],
        "answer": "B",
        "explanation": "Top-P limits token selection to the smallest set of tokens whose cumulative probability meets or exceeds P (e.g., top_p=0.9 considers tokens until 90% cumulative probability).",
    },
    {
        "q": "Should you vary temperature AND top-p simultaneously?",
        "options": [
            "A) Yes, always use both together",
            "B) No — typically vary one and keep the other at default",
            "C) They control the same thing, so it doesn't matter",
            "D) Both must be set to 0",
        ],
        "answer": "B",
        "explanation": "Best practice is to adjust either temperature OR top-p, not both simultaneously, as they both control output diversity in overlapping ways.",
    },

    # ── BEST PRACTICES ──────────────────────────────────────
    {
        "q": "What is the PRIMARY defense against prompt injection?",
        "options": [
            "A) Using a larger model",
            "B) Using delimiters and input validation to separate instructions from user input",
            "C) Setting temperature = 0",
            "D) Using few-shot examples",
        ],
        "answer": "B",
        "explanation": "Delimiters (triple backticks, XML tags) clearly separate system instructions from user input, making it harder for injected text to override instructions.",
    },
    {
        "q": "Why should you include 'negative instructions' in prompts?",
        "options": [
            "A) To confuse the model",
            "B) To explicitly state what the model should NOT do (e.g., 'Do NOT hallucinate')",
            "C) To reduce the prompt length",
            "D) To lower the token count",
        ],
        "answer": "B",
        "explanation": "Negative instructions like 'Do NOT make up information' and 'Do NOT include personal opinions' set explicit boundaries for the model's behavior.",
    },
    {
        "q": "When should you use fine-tuning instead of prompt engineering?",
        "options": [
            "A) Always — fine-tuning is always better",
            "B) Only when prompt engineering doesn't achieve the required quality",
            "C) Never — prompt engineering is always sufficient",
            "D) Only for simple tasks",
        ],
        "answer": "B",
        "explanation": "Prompt engineering is the first optimization step. Fine-tune only if prompting can't achieve the quality bar, or when domain-specific behavior is too complex for prompts.",
    },

    # ── ADVANCED CONCEPTS ───────────────────────────────────
    {
        "q": "What is 'Generated Knowledge Prompting'?",
        "options": [
            "A) Using a knowledge graph",
            "B) First ask the model to generate background knowledge, then use it to answer",
            "C) Generating synthetic training data",
            "D) Creating embeddings from prompts",
        ],
        "answer": "B",
        "explanation": "Generated Knowledge is a two-step technique: (1) generate relevant facts, (2) use those facts to answer the actual question. Activates the model's latent knowledge.",
    },
    {
        "q": "What is Directional Stimulus Prompting?",
        "options": [
            "A) Using electric signals to control the model",
            "B) Providing a hint or nudge to guide the model toward specific aspects of the answer",
            "C) Training the model on directional data",
            "D) Prompting the model in multiple languages",
        ],
        "answer": "B",
        "explanation": "Directional Stimulus provides a 'hint' that steers the model toward focusing on particular aspects (e.g., 'Hint: Focus on economic impact').",
    },
    {
        "q": "What is Retrieval-Augmented Prompting (Dynamic Few-Shot)?",
        "options": [
            "A) Using static examples in every prompt",
            "B) Dynamically retrieving relevant examples from a vector store based on the input query",
            "C) Retrieving the model weights before inference",
            "D) Using a database to store model outputs",
        ],
        "answer": "B",
        "explanation": "Dynamic Few-Shot retrieves the most similar examples from a vector store for each query, making few-shot examples more relevant than static ones.",
    },

    # ── LCEL & LANGCHAIN ────────────────────────────────────
    {
        "q": "What does LCEL stand for in LangChain?",
        "options": [
            "A) Large Chain Expression Library",
            "B) LangChain Expression Language",
            "C) Language Chain Evaluation Layer",
            "D) Linked Chain Execution Logic",
        ],
        "answer": "B",
        "explanation": "LCEL (LangChain Expression Language) is a declarative way to compose chains using the pipe operator: prompt | llm | parser.",
    },
    {
        "q": "In LangChain, which class creates few-shot prompts?",
        "options": [
            "A) ChatPromptTemplate",
            "B) FewShotChatMessagePromptTemplate",
            "C) PromptTemplate",
            "D) SystemMessagePromptTemplate",
        ],
        "answer": "B",
        "explanation": "FewShotChatMessagePromptTemplate is the dedicated LangChain class for creating few-shot prompts with example input-output pairs.",
    },
    {
        "q": "What does LangChain's SemanticSimilarityExampleSelector do?",
        "options": [
            "A) Selects the model based on similarity",
            "B) Selects the most relevant few-shot examples based on semantic similarity to the input",
            "C) Selects the most similar documents",
            "D) Selects the most similar embeddings",
        ],
        "answer": "B",
        "explanation": "SemanticSimilarityExampleSelector dynamically picks the most relevant examples from a pool based on embedding similarity to the user's query.",
    },

    # ── SCENARIO-BASED ──────────────────────────────────────
    {
        "q": "You need an LLM to answer customer questions from a product manual. Which technique is BEST?",
        "options": [
            "A) Zero-shot prompting",
            "B) Grounded/Contextual prompting (RAG) with anti-hallucination guardrails",
            "C) Chain-of-Thought prompting",
            "D) Tree-of-Thought prompting",
        ],
        "answer": "B",
        "explanation": "RAG with grounding is ideal: retrieve relevant manual sections and instruct the model to answer ONLY from the provided context.",
    },
    {
        "q": "A model keeps giving wrong answers to math word problems. What should you try FIRST?",
        "options": [
            "A) Fine-tune the model",
            "B) Switch to a larger model",
            "C) Add Chain-of-Thought prompting (show step-by-step reasoning)",
            "D) Add more few-shot examples without reasoning",
        ],
        "answer": "C",
        "explanation": "CoT prompting is the most impactful change for math reasoning — it forces the model to decompose the problem into steps, dramatically improving accuracy.",
    },
    {
        "q": "You need the LLM to search the web and use a calculator. Which technique?",
        "options": [
            "A) Few-shot prompting",
            "B) Chain-of-Thought",
            "C) ReAct (Reasoning + Acting) with tool access",
            "D) Zero-shot prompting",
        ],
        "answer": "C",
        "explanation": "ReAct enables the model to interleave reasoning with tool use (search, calculator, APIs) through the Thought-Action-Observation loop.",
    },
    {
        "q": "You want a model to output valid JSON for a downstream API. What's the BEST approach?",
        "options": [
            "A) Hope the model figures it out",
            "B) Structured Output Prompting with an explicit JSON schema + JsonOutputParser",
            "C) Use few-shot examples only",
            "D) Set temperature = 2.0",
        ],
        "answer": "B",
        "explanation": "Provide the exact JSON schema in the prompt and use a parser (JsonOutputParser or PydanticOutputParser) to validate and extract structured output.",
    },
    {
        "q": "A complex task needs entity extraction → classification → report generation. Best approach?",
        "options": [
            "A) One massive prompt with all instructions",
            "B) Prompt Chaining — break into 3 sequential prompts",
            "C) Zero-shot prompting",
            "D) Few-shot with 20 examples",
        ],
        "answer": "B",
        "explanation": "Prompt Chaining breaks complex workflows into manageable steps, where each step is simpler and more reliable than one monolithic prompt.",
    },
]


def run_quiz(questions: list, num_questions: int = 15) -> None:
    """Run an interactive quiz with the given questions."""
    selected = random.sample(questions, min(num_questions, len(questions)))
    score = 0
    total = len(selected)

    print("\n" + "=" * 60)
    print("🧠 PROMPTING TECHNIQUES — PRACTICE QUIZ")
    print(f"   {total} questions | GenAI L2 Exam Prep")
    print("=" * 60)

    for i, q in enumerate(selected, 1):
        print(f"\n{'─' * 50}")
        print(f"Q{i}/{total}: {q['q']}\n")
        for opt in q["options"]:
            print(f"   {opt}")

        while True:
            answer = input(f"\nYour answer (A/B/C/D): ").strip().upper()
            if answer in ("A", "B", "C", "D"):
                break
            print("   ⚠️  Please enter A, B, C, or D")

        if answer == q["answer"]:
            print(f"   ✅ Correct!")
            score += 1
        else:
            print(f"   ❌ Wrong! Correct answer: {q['answer']}")

        print(f"   📝 {q['explanation']}")

    print(f"\n{'=' * 60}")
    print(f"📊 FINAL SCORE: {score}/{total} ({score/total*100:.0f}%)")
    print("=" * 60)

    if score / total >= 0.8:
        print("🎉 Excellent! You're well prepared for the L2 exam!")
    elif score / total >= 0.6:
        print("👍 Good progress! Review the topics you missed.")
    else:
        print("📚 Keep studying! Re-read 01_Prompting_Study_Guide.md and try again.")


if __name__ == "__main__":
    print(f"\nTotal questions available: {len(QUESTIONS)}")
    print("Default quiz: 15 random questions\n")

    try:
        n = input("How many questions? (press Enter for 15): ").strip()
        num = int(n) if n else 15
    except ValueError:
        num = 15

    run_quiz(QUESTIONS, num)
