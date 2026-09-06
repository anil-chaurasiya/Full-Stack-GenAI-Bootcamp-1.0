"""
Module 12: Advanced RAG MCQs — Evaluation, Optimization & Debugging
====================================================================
60 scenario-based MCQs for GenAI L2 exam prep.
Covers topics from Modules 09–10 (Evaluation, Optimization)
plus advanced debugging scenarios NOT in 11_Practice_MCQs.

Run this script to take the quiz interactively, or read through
for self-assessment. Uses VS Code Interactive Window (# %%).
"""

import sys

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# %% [1] Quiz Engine
def run_quiz(questions: list[dict]) -> None:
    """Run an interactive MCQ quiz."""
    score = 0
    total = len(questions)

    for i, q in enumerate(questions, 1):
        print(f"\n{'='*70}")
        print(f"Q{i}/{total}: {q['question']}")
        print("-" * 70)
        for opt in q["options"]:
            print(f"  {opt}")
        print()

        try:
            answer = input("Your answer (a/b/c/d or 'q' to quit): ").strip().lower()
        except EOFError:
            # Non-interactive mode — print answer key instead
            print(f"  ✅ Answer: {q['answer']}")
            print(f"  💡 {q['explanation']}")
            continue

        if answer == 'q':
            print(f"\n📊 Score so far: {score}/{i-1}")
            return

        if answer == q["answer"]:
            print("  ✅ Correct!")
            score += 1
        else:
            print(f"  ❌ Wrong! Correct answer: {q['answer']}")
        print(f"  💡 {q['explanation']}")

    print(f"\n{'='*70}")
    print(f"📊 FINAL SCORE: {score}/{total} ({score/total*100:.0f}%)")
    print("='*70")
    if score / total >= 0.9:
        print("🟢 Exam ready!")
    elif score / total >= 0.75:
        print("🟡 Good foundation — review weak areas")
    elif score / total >= 0.6:
        print("🟠 Need more study — revisit the modules")
    else:
        print("🔴 Go through all modules again")


# %% [2] Section A: RAGAS Evaluation Metrics (Q1-Q15)
section_a = [
    {
        "question": "Which RAGAS metric measures whether the LLM's answer is grounded in the retrieved context?",
        "options": ["a) Context Recall", "b) Answer Relevance", "c) Faithfulness", "d) Context Precision"],
        "answer": "c",
        "explanation": "Faithfulness checks if claims in the answer are supported by the retrieved context. Low faithfulness = hallucination."
    },
    {
        "question": "Your RAG system has high Context Recall but low Context Precision. What does this indicate?",
        "options": [
            "a) The system can't find any relevant documents",
            "b) It finds all relevant docs but also retrieves many irrelevant ones",
            "c) The LLM is hallucinating",
            "d) The embedding model is broken"
        ],
        "answer": "b",
        "explanation": "High recall = finding everything relevant. Low precision = also retrieving lots of noise. Fix: reduce k, use MMR, or add metadata filtering."
    },
    {
        "question": "A RAG system returns answers that sound plausible but contain facts NOT present in the retrieved documents. Which metric would be low?",
        "options": ["a) Context Precision", "b) Context Recall", "c) Faithfulness", "d) Answer Relevance"],
        "answer": "c",
        "explanation": "Faithfulness measures whether answer claims are supported by context. The model is hallucinating beyond its retrieved context."
    },
    {
        "question": "Which RAGAS metric answers: 'Does the response actually address what the user asked?'",
        "options": ["a) Faithfulness", "b) Answer Relevance", "c) Context Precision", "d) Context Recall"],
        "answer": "b",
        "explanation": "Answer Relevance measures how well the generated answer matches the user's question intent."
    },
    {
        "question": "You have a RAG pipeline where retrieval returns the right documents, but the LLM ignores them and gives its own opinion. What should you fix?",
        "options": [
            "a) Use a larger vector database",
            "b) Increase chunk_overlap",
            "c) Add grounding instructions to the prompt and set temperature=0",
            "d) Switch to a different embedding model"
        ],
        "answer": "c",
        "explanation": "Grounding instructions ('Only use provided context') and temperature=0 force the LLM to stick to the retrieved context."
    },
    {
        "question": "Context Precision is LOW. Which fix is MOST appropriate?",
        "options": [
            "a) Use a bigger LLM",
            "b) Improve chunking strategy and use hybrid search",
            "c) Increase temperature",
            "d) Add more documents to the knowledge base"
        ],
        "answer": "b",
        "explanation": "Low Context Precision = retrieving irrelevant docs. Better chunking and hybrid search improve the relevance of retrieved results."
    },
    {
        "question": "Context Recall is LOW. Which fix is MOST appropriate?",
        "options": [
            "a) Decrease k value",
            "b) Use a simpler prompt",
            "c) Increase k, use multi-query retriever, or ensemble retriever",
            "d) Switch from FAISS to ChromaDB"
        ],
        "answer": "c",
        "explanation": "Low Context Recall = missing relevant docs. Increasing k and using multi-query/ensemble retrievers casts a wider net."
    },
    {
        "question": "Which metric pair best diagnoses RETRIEVAL quality (not generation quality)?",
        "options": [
            "a) Faithfulness + Answer Relevance",
            "b) Context Precision + Context Recall",
            "c) Faithfulness + Context Precision",
            "d) Answer Relevance + Context Recall"
        ],
        "answer": "b",
        "explanation": "Context Precision and Context Recall measure retrieval quality. Faithfulness and Answer Relevance measure generation quality."
    },
    {
        "question": "A production RAG system should aim for a Faithfulness score of at least:",
        "options": ["a) 0.3", "b) 0.5", "c) 0.8", "d) 1.0 always"],
        "answer": "c",
        "explanation": "Faithfulness > 0.8 is the production target. Perfect 1.0 is ideal but not always achievable."
    },
    {
        "question": "Which of these is NOT a RAGAS core metric?",
        "options": ["a) Faithfulness", "b) Latency", "c) Context Precision", "d) Answer Relevance"],
        "answer": "b",
        "explanation": "Latency is important for production but is not a RAGAS metric. RAGAS focuses on Faithfulness, Answer Relevance, Context Precision, and Context Recall."
    },
    {
        "question": "Your RAG system correctly retrieves relevant docs AND the LLM uses them faithfully, but the answer doesn't match the question. Which metric is low?",
        "options": ["a) Faithfulness", "b) Context Recall", "c) Answer Relevance", "d) Context Precision"],
        "answer": "c",
        "explanation": "Answer Relevance measures if the answer addresses the question. The LLM is using context faithfully but generating off-topic responses."
    },
    {
        "question": "You increased k from 3 to 20 and answers got WORSE. Why?",
        "options": [
            "a) The vector database crashed",
            "b) More irrelevant docs diluted the context (lower Context Precision)",
            "c) The embedding model changed",
            "d) The LLM became slower"
        ],
        "answer": "b",
        "explanation": "Higher k retrieves more docs, including irrelevant ones. This dilutes the useful context and may overflow the context window."
    },
    {
        "question": "For a medical RAG system where accuracy is critical, which temperature should you use?",
        "options": ["a) 0.7", "b) 0.5", "c) 0.0", "d) 1.0"],
        "answer": "c",
        "explanation": "Temperature=0 gives deterministic, factual outputs. For critical domains like healthcare, minimizing creativity prevents hallucination."
    },
    {
        "question": "Which failure mode does the prompt instruction 'If the context doesn't contain the answer, say I don't know' address?",
        "options": ["a) Slow response time", "b) Embedding mismatch", "c) Hallucination", "d) Context window overflow"],
        "answer": "c",
        "explanation": "This fallback instruction prevents the LLM from making up answers when the retrieved context doesn't contain relevant information."
    },
    {
        "question": "RAGAS evaluation requires which inputs?",
        "options": [
            "a) Only the question",
            "b) Question, retrieved context, and generated answer",
            "c) Only the answer",
            "d) Question and LLM model name"
        ],
        "answer": "b",
        "explanation": "RAGAS evaluates the relationship between the question, what was retrieved (context), and what was generated (answer). Some metrics also need ground truth."
    },
]


# %% [3] Section B: Optimization & Advanced Debugging (Q16-Q30)
section_b = [
    {
        "question": "Which optimization has the HIGHEST impact on retrieval quality?",
        "options": [
            "a) Switching from ChromaDB to FAISS",
            "b) Using hybrid search (dense + BM25 ensemble)",
            "c) Increasing chunk_overlap from 10% to 50%",
            "d) Using a GPU for embedding"
        ],
        "answer": "b",
        "explanation": "Hybrid search combines semantic understanding (dense) with keyword matching (BM25), catching both synonyms and exact terms."
    },
    {
        "question": "A user searches for 'LoRA' but gets no relevant results even though the knowledge base contains LoRA documentation. Dense retrieval is used. What's the fix?",
        "options": [
            "a) Use a larger LLM",
            "b) Increase chunk_size",
            "c) Add BM25 (sparse) retrieval via ensemble/hybrid search",
            "d) Change the prompt"
        ],
        "answer": "c",
        "explanation": "Dense retrieval might not surface exact technical terms. BM25 excels at exact keyword matching like 'LoRA'."
    },
    {
        "question": "You want to reduce LLM API costs in your RAG system. Which strategy is LEAST effective?",
        "options": [
            "a) Cache frequent query responses",
            "b) Use a smaller free LLM (e.g., Llama 8B instead of GPT-4)",
            "c) Increase chunk_size from 500 to 5000",
            "d) Use free local embeddings instead of OpenAI embeddings"
        ],
        "answer": "c",
        "explanation": "Increasing chunk_size to 5000 sends MORE tokens to the LLM per query, potentially increasing cost. The other options reduce cost."
    },
    {
        "question": "Your RAG pipeline takes 15 seconds per query. What should you optimize FIRST?",
        "options": [
            "a) Switch to a fancier prompt template",
            "b) Profile bottleneck: is it embedding, retrieval, or LLM generation?",
            "c) Use a larger embedding model",
            "d) Add more documents to the knowledge base"
        ],
        "answer": "b",
        "explanation": "Always profile first. The bottleneck could be slow embedding (use smaller model), slow retrieval (reduce k), or slow LLM (use streaming)."
    },
    {
        "question": "Which caching strategy reduces both cost AND latency in a RAG system?",
        "options": [
            "a) Cache LLM responses for repeated queries",
            "b) Cache vector database connections",
            "c) Cache the prompt template",
            "d) Cache the Python imports"
        ],
        "answer": "a",
        "explanation": "Caching LLM responses for identical/similar queries avoids repeated API calls, saving both time and money."
    },
    {
        "question": "A cross-encoder reranker is added after retrieval. What does it do?",
        "options": [
            "a) Creates new embeddings for the query",
            "b) Re-scores retrieved documents using a more powerful model for better ranking",
            "c) Splits documents into smaller chunks",
            "d) Converts sparse results to dense vectors"
        ],
        "answer": "b",
        "explanation": "Reranking uses a cross-encoder to re-score query-document pairs, producing more accurate relevance rankings than bi-encoder similarity."
    },
    {
        "question": "Your embedding model produces 384-dimensional vectors, but you create a Pinecone index with dimension=768. What happens?",
        "options": [
            "a) Pinecone auto-adjusts",
            "b) Dimension mismatch error when upserting vectors",
            "c) Slower search but correct results",
            "d) The vectors are zero-padded"
        ],
        "answer": "b",
        "explanation": "The index dimension MUST exactly match the embedding model's output dimension. You'd need to recreate the index with dimension=384."
    },
    {
        "question": "You changed from 'all-MiniLM-L6-v2' to 'all-mpnet-base-v2' for better quality. Queries now return no results. Why?",
        "options": [
            "a) The new model is too slow",
            "b) The API key expired",
            "c) Dimension mismatch: old vectors are 384d, new queries are 768d — need to re-index",
            "d) The vector database is corrupted"
        ],
        "answer": "c",
        "explanation": "MiniLM produces 384d vectors, mpnet produces 768d. Existing vectors must be re-indexed with the new model."
    },
    {
        "question": "When would you use 'map_reduce' chain type instead of 'stuff'?",
        "options": [
            "a) When you have only 2 documents",
            "b) When retrieved documents exceed the LLM's context window",
            "c) When you need faster responses",
            "d) When using BM25 retrieval"
        ],
        "answer": "b",
        "explanation": "map_reduce processes each document separately then combines summaries, avoiding context window overflow. Trade-off: multiple LLM calls = slower."
    },
    {
        "question": "Which approach BEST reduces hallucination in a RAG system?",
        "options": [
            "a) Use a larger vector database",
            "b) Increase chunk_overlap to 90%",
            "c) Combine grounding prompt, temperature=0, and faithful source citing",
            "d) Use FAISS instead of ChromaDB"
        ],
        "answer": "c",
        "explanation": "The generation side controls hallucination: grounding prompt constrains to context, temp=0 ensures determinism, citing forces traceability."
    },
    {
        "question": "You have a RAG system for a legal firm. Documents contain metadata like 'department', 'year', and 'case_type'. A user asks about 'employment law cases from 2024'. What improves retrieval most?",
        "options": [
            "a) Increase k to 50",
            "b) Use metadata filtering with semantic search",
            "c) Switch to a larger embedding model",
            "d) Use CharacterTextSplitter"
        ],
        "answer": "b",
        "explanation": "Metadata filtering narrows results to year=2024 AND case_type='employment law' BEFORE semantic search, drastically improving precision."
    },
    {
        "question": "Which vector DB would you choose for a system that needs complex metadata filtering with nested boolean queries?",
        "options": ["a) FAISS", "b) ChromaDB", "c) Qdrant", "d) SQLite"],
        "answer": "c",
        "explanation": "Qdrant has the best filtering capabilities, supporting complex payload filters with range queries, nested filters, and boolean combinations."
    },
    {
        "question": "Streaming LLM output in a RAG system improves:",
        "options": [
            "a) Actual total generation time",
            "b) Perceived responsiveness (time to first token)",
            "c) Retrieval quality",
            "d) Embedding accuracy"
        ],
        "answer": "b",
        "explanation": "Streaming doesn't reduce total time but shows tokens as they're generated, making the system feel more responsive."
    },
    {
        "question": "Your RAG system works well in testing but performs poorly in production. Which is the MOST likely cause?",
        "options": [
            "a) The LLM changed its name",
            "b) Production queries are more diverse and complex than test queries",
            "c) Python version changed",
            "d) The monitor resolution changed"
        ],
        "answer": "b",
        "explanation": "Production queries are more varied, ambiguous, and edge-case-heavy. Test with realistic, diverse queries before deploying."
    },
    {
        "question": "Which strategy gives the best retrieval quality for a production RAG system?",
        "options": [
            "a) Similarity search with k=1",
            "b) BM25 only",
            "c) Ensemble (dense + BM25) with reranking",
            "d) Random sampling"
        ],
        "answer": "c",
        "explanation": "Ensemble catches both semantic and keyword matches. Reranking re-scores results using a cross-encoder for maximum precision."
    },
]


# %% [4] Section C: Broader GenAI L2 — Transformers & LLMs (Q31-Q45)
section_c = [
    {
        "question": "The Transformer architecture was introduced in which paper?",
        "options": [
            "a) 'BERT: Pre-training of Deep Bidirectional Transformers'",
            "b) 'Attention Is All You Need'",
            "c) 'GPT: Improving Language Understanding'",
            "d) 'ImageNet Classification with Deep CNNs'"
        ],
        "answer": "b",
        "explanation": "The Transformer was introduced in 'Attention Is All You Need' (Vaswani et al., 2017), replacing RNNs with self-attention."
    },
    {
        "question": "What is the core mechanism that makes Transformers so powerful?",
        "options": [
            "a) Convolutional layers",
            "b) Recurrent connections",
            "c) Self-attention mechanism",
            "d) Pooling layers"
        ],
        "answer": "c",
        "explanation": "Self-attention allows each token to attend to every other token in the sequence, capturing long-range dependencies efficiently."
    },
    {
        "question": "What is multi-head attention?",
        "options": [
            "a) Attention computed on multiple documents",
            "b) Running multiple independent attention operations in parallel, each learning different patterns",
            "c) Attention with multiple layers stacked",
            "d) Paying attention to multiple users simultaneously"
        ],
        "answer": "b",
        "explanation": "Multi-head attention runs several attention functions in parallel, each 'head' learning different aspects (syntax, semantics, etc.)."
    },
    {
        "question": "In a Transformer, what is the purpose of positional encoding?",
        "options": [
            "a) To encode the meaning of words",
            "b) To inject information about token position in the sequence (since self-attention is permutation-invariant)",
            "c) To reduce the model size",
            "d) To speed up training"
        ],
        "answer": "b",
        "explanation": "Self-attention treats input as a set (no order). Positional encoding adds position info so the model knows word order."
    },
    {
        "question": "What is the difference between the Transformer encoder and decoder?",
        "options": [
            "a) Encoder generates text, decoder understands text",
            "b) Encoder processes input bidirectionally, decoder generates output autoregressively with masked attention",
            "c) They are identical",
            "d) Encoder uses RNNs, decoder uses attention"
        ],
        "answer": "b",
        "explanation": "Encoder sees all tokens (bidirectional). Decoder generates tokens one by one, using masked attention to prevent seeing future tokens."
    },
    {
        "question": "GPT models are based on which part of the Transformer?",
        "options": ["a) Encoder only", "b) Decoder only", "c) Both encoder and decoder", "d) Neither"],
        "answer": "b",
        "explanation": "GPT uses decoder-only architecture for autoregressive text generation (predicting next token)."
    },
    {
        "question": "BERT models are based on which part of the Transformer?",
        "options": ["a) Encoder only", "b) Decoder only", "c) Both encoder and decoder", "d) Neither"],
        "answer": "a",
        "explanation": "BERT uses encoder-only architecture for bidirectional understanding tasks (classification, NER, etc.)."
    },
    {
        "question": "What does 'LLM' stand for and what defines one?",
        "options": [
            "a) Large Learning Machine — any ML model over 1GB",
            "b) Large Language Model — a neural network with billions of parameters trained on large text corpora",
            "c) Logical Language Mechanism — a rule-based system",
            "d) Linear Language Model — a statistical model"
        ],
        "answer": "b",
        "explanation": "LLMs are neural networks with billions of parameters (e.g., GPT-4, Llama 3) trained on massive text data for language understanding and generation."
    },
    {
        "question": "What is the 'context window' of an LLM?",
        "options": [
            "a) The number of layers in the model",
            "b) The maximum number of tokens the model can process in a single input+output",
            "c) The training time of the model",
            "d) The physical memory of the GPU"
        ],
        "answer": "b",
        "explanation": "Context window = max tokens (input + output). GPT-4 has 128K, Llama 3 has 8K-128K depending on variant."
    },
    {
        "question": "What is the difference between an SLM (Small Language Model) and an LLM?",
        "options": [
            "a) SLMs are always faster and better",
            "b) SLMs have fewer parameters (1-7B), are faster/cheaper but less capable; LLMs (70B+) are more capable but expensive",
            "c) SLMs only work on mobile",
            "d) There is no difference"
        ],
        "answer": "b",
        "explanation": "SLMs (like Phi-3, Gemma 2B) trade capability for efficiency. LLMs (like GPT-4, Llama 70B) offer superior reasoning but need more resources."
    },
    {
        "question": "What is tokenization in the context of LLMs?",
        "options": [
            "a) Converting money to digital tokens",
            "b) Splitting text into subword units (tokens) that the model processes",
            "c) Encrypting the input text",
            "d) Compressing text files"
        ],
        "answer": "b",
        "explanation": "Tokenization splits text into tokens (subwords/words). 'unhappiness' might become ['un', 'happiness']. Models process token IDs, not raw text."
    },
    {
        "question": "Which of these is a multimodal LLM?",
        "options": ["a) BERT", "b) GPT-4o (Vision)", "c) Word2Vec", "d) TF-IDF"],
        "answer": "b",
        "explanation": "GPT-4o can process text, images, and audio — making it multimodal. BERT and Word2Vec are text-only."
    },
    {
        "question": "What does 'temperature' control in LLM generation?",
        "options": [
            "a) The speed of generation",
            "b) The randomness/creativity of the output (higher = more creative, lower = more deterministic)",
            "c) The number of tokens generated",
            "d) The model's memory usage"
        ],
        "answer": "b",
        "explanation": "Temperature scales the logits before softmax. Low temp → sharp distribution (deterministic). High temp → flat distribution (diverse/creative)."
    },
    {
        "question": "What is 'top_p' (nucleus sampling)?",
        "options": [
            "a) The number of top words to consider",
            "b) Sampling from the smallest set of tokens whose cumulative probability exceeds p",
            "c) The model's accuracy score",
            "d) The GPU utilization percentage"
        ],
        "answer": "b",
        "explanation": "top_p=0.9 means sample from the smallest token set whose probabilities sum to ≥0.9, dynamically adapting the vocabulary size."
    },
    {
        "question": "Which LLM access method is FREE and runs locally without an API key?",
        "options": [
            "a) OpenAI GPT-4 API",
            "b) Ollama with open-source models (Llama, Mistral)",
            "c) Anthropic Claude API",
            "d) Google Gemini Pro API"
        ],
        "answer": "b",
        "explanation": "Ollama runs open-source models locally for free. All API providers (OpenAI, Anthropic, Google) require API keys and may charge."
    },
]


# %% [5] Section D: Fine-Tuning, LoRA, RLHF (Q46-Q60)
section_d = [
    {
        "question": "What is fine-tuning in the context of LLMs?",
        "options": [
            "a) Training a model from scratch on a new dataset",
            "b) Further training a pre-trained model on task-specific data to adapt its behavior",
            "c) Adjusting the learning rate during pre-training",
            "d) Compressing the model for mobile deployment"
        ],
        "answer": "b",
        "explanation": "Fine-tuning takes a pre-trained model and continues training on domain-specific or task-specific data to adapt its outputs."
    },
    {
        "question": "What is the key difference between instruction tuning and non-instruction fine-tuning?",
        "options": [
            "a) Instruction tuning is faster",
            "b) Instruction tuning uses instruction-response pairs; non-instruction uses raw text (next-token prediction)",
            "c) Non-instruction tuning produces better results",
            "d) They are the same"
        ],
        "answer": "b",
        "explanation": "Instruction tuning trains on (instruction, response) pairs for following commands. Non-instruction trains on raw text for domain adaptation."
    },
    {
        "question": "What does LoRA stand for and what does it do?",
        "options": [
            "a) Large Optimization for Retrieval Augmentation — speeds up RAG",
            "b) Low-Rank Adaptation — adds small trainable matrices to frozen model weights for parameter-efficient fine-tuning",
            "c) Long-Range Attention — extends context window",
            "d) Layered Output Reformulation Algorithm — improves output quality"
        ],
        "answer": "b",
        "explanation": "LoRA inserts small rank-decomposed matrices into the model, training only ~0.1-1% of parameters while keeping original weights frozen."
    },
    {
        "question": "How does QLoRA differ from LoRA?",
        "options": [
            "a) QLoRA uses larger matrices",
            "b) QLoRA quantizes the base model to 4-bit, then applies LoRA adapters, drastically reducing memory",
            "c) QLoRA is a different model architecture",
            "d) QLoRA doesn't use adapters"
        ],
        "answer": "b",
        "explanation": "QLoRA combines 4-bit quantization (NF4) of the base model with LoRA adapters, enabling fine-tuning of 70B models on a single GPU."
    },
    {
        "question": "What is RLHF (Reinforcement Learning from Human Feedback)?",
        "options": [
            "a) Training with reinforcement learning on game environments",
            "b) Aligning LLM outputs with human preferences using a reward model trained on human comparisons",
            "c) Replacing the attention mechanism with reinforcement learning",
            "d) Using humans to label training data"
        ],
        "answer": "b",
        "explanation": "RLHF trains a reward model from human preference data (A vs B), then uses RL (PPO) to fine-tune the LLM to maximize the reward."
    },
    {
        "question": "What is DPO (Direct Preference Optimization) and how does it relate to RLHF?",
        "options": [
            "a) DPO is identical to RLHF",
            "b) DPO simplifies RLHF by eliminating the reward model — it directly optimizes from preference pairs",
            "c) DPO is for image generation only",
            "d) DPO replaces the LLM entirely"
        ],
        "answer": "b",
        "explanation": "DPO skips the reward model and PPO step, directly training on (preferred, rejected) pairs. Simpler, more stable, and increasingly popular."
    },
    {
        "question": "When should you fine-tune instead of using RAG?",
        "options": [
            "a) When you need access to private documents",
            "b) When you need to change the model's writing style, tone, or output format",
            "c) When data changes daily",
            "d) When you need source citations"
        ],
        "answer": "b",
        "explanation": "Fine-tuning modifies model behavior (style, format). RAG provides external knowledge. For behavioral changes, fine-tuning is the answer."
    },
    {
        "question": "What is the advantage of parameter-efficient fine-tuning (PEFT) methods like LoRA?",
        "options": [
            "a) Better accuracy than full fine-tuning always",
            "b) Train only a small fraction of parameters, reducing memory and compute dramatically",
            "c) No GPU needed",
            "d) Models become smaller after training"
        ],
        "answer": "b",
        "explanation": "PEFT trains ~0.1-1% of parameters (e.g., 10M instead of 7B), needing far less GPU memory and time while achieving comparable results."
    },
    {
        "question": "In Hugging Face Transformers, what is the role of the 'Trainer' class?",
        "options": [
            "a) It loads pre-trained models",
            "b) It handles the training loop, optimization, evaluation, and logging for fine-tuning",
            "c) It converts text to tokens",
            "d) It deploys models to production"
        ],
        "answer": "b",
        "explanation": "Trainer abstracts the training loop, handling batching, gradient computation, optimizer steps, evaluation, checkpointing, and logging."
    },
    {
        "question": "What is Unsloth used for in the context of LLM fine-tuning?",
        "options": [
            "a) A vector database",
            "b) A library that speeds up LoRA/QLoRA fine-tuning by 2-5x with optimized kernels",
            "c) A prompt engineering tool",
            "d) A model evaluation framework"
        ],
        "answer": "b",
        "explanation": "Unsloth provides optimized CUDA kernels for LoRA fine-tuning, achieving 2-5x speed improvements with 60-80% less memory usage."
    },
    {
        "question": "What is a 'chat template' in the context of instruction-tuned LLMs?",
        "options": [
            "a) A CSS template for chatbot UIs",
            "b) A specific formatting structure (e.g., <s>[INST] ... [/INST]) that the model was trained on for multi-turn conversations",
            "c) A database schema for storing chats",
            "d) A prompt for generating HTML"
        ],
        "answer": "b",
        "explanation": "Chat templates format messages into the specific structure the model expects (e.g., Llama uses [INST]...[/INST], ChatML uses <|im_start|>)."
    },
    {
        "question": "What does 'quantization' mean in the context of LLMs?",
        "options": [
            "a) Making the model output shorter",
            "b) Reducing the precision of model weights (e.g., float32 → int4) to reduce memory and increase speed",
            "c) Adding more parameters",
            "d) Converting text to numbers"
        ],
        "answer": "b",
        "explanation": "Quantization reduces weight precision (32-bit → 8-bit or 4-bit), shrinking model size dramatically with minimal quality loss."
    },
    {
        "question": "A 7B parameter model in float32 requires approximately how much GPU memory?",
        "options": ["a) 1 GB", "b) 7 GB", "c) 28 GB", "d) 56 GB"],
        "answer": "c",
        "explanation": "Each float32 parameter = 4 bytes. 7B × 4 bytes = 28 GB. With 4-bit quantization, it drops to ~3.5 GB."
    },
    {
        "question": "Which of the following is the correct RLHF pipeline order?",
        "options": [
            "a) SFT → Reward Model → PPO",
            "b) PPO → SFT → Reward Model",
            "c) Reward Model → SFT → PPO",
            "d) SFT → PPO → Reward Model"
        ],
        "answer": "a",
        "explanation": "RLHF: (1) SFT (supervised fine-tuning on instructions), (2) Train a reward model on human preferences, (3) Optimize with PPO against the reward model."
    },
    {
        "question": "What is the 'LoRA rank' (r) parameter and how does it affect fine-tuning?",
        "options": [
            "a) It sets the learning rate",
            "b) It controls the size of the low-rank matrices — higher r = more parameters = more capacity but more memory",
            "c) It determines the number of training epochs",
            "d) It sets the batch size"
        ],
        "answer": "b",
        "explanation": "LoRA rank r controls the dimension of the decomposed matrices. Typical values: r=8-64. Higher r = more expressive but needs more memory."
    },
]


# %% [6] Run the full quiz
if __name__ == "__main__":
    all_questions = section_a + section_b + section_c + section_d

    print("=" * 70)
    print("📝 GenAI L2 Exam — Advanced MCQ Practice (60 Questions)")
    print("=" * 70)
    print("""
Sections:
  A: RAGAS Evaluation Metrics           (Q1-Q15)
  B: Optimization & Advanced Debugging  (Q16-Q30)
  C: Transformers & LLMs                (Q31-Q45)
  D: Fine-Tuning, LoRA, RLHF           (Q46-Q60)

Enter your answer as a/b/c/d, or 'q' to quit.
""")

    run_quiz(all_questions)
