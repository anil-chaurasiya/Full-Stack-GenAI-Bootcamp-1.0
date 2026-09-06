"""
Module 13: GenAI L2 Foundations MCQs — Encodings, Embeddings & APIs
====================================================================
50 scenario-based MCQs covering foundational GenAI L2 topics:
  - Text encoding methods (OHE, BOW, TF-IDF, Word2Vec)
  - Transformer internals (attention, encoder/decoder)
  - LLM API access (Groq, OpenAI, Google, HuggingFace, Ollama)
  - Model comparison (LLM vs SLM vs MLLM)
  - Practical debugging and API scenarios

Run this script to take the quiz interactively.
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
    print("=" * 70)
    if score / total >= 0.9:
        print("🟢 Exam ready!")
    elif score / total >= 0.75:
        print("🟡 Good foundation — review weak areas")
    elif score / total >= 0.6:
        print("🟠 Need more study — revisit the modules")
    else:
        print("🔴 Go through all modules again")


# %% [2] Section A: Text Encoding & Representation (Q1-Q15)
section_a = [
    {
        "question": "One-Hot Encoding (OHE) represents each word as a vector where:",
        "options": [
            "a) All values are non-zero with learned weights",
            "b) Exactly one element is 1 and all others are 0, with vector length = vocabulary size",
            "c) Values represent word frequency",
            "d) Values capture semantic meaning"
        ],
        "answer": "b",
        "explanation": "OHE creates a sparse vector of vocabulary size with a single 1 at the word's index. It's simple but captures no relationships between words."
    },
    {
        "question": "What is the major limitation of One-Hot Encoding for NLP?",
        "options": [
            "a) It's too slow to compute",
            "b) It doesn't capture semantic similarity — 'king' and 'queen' are equally distant as 'king' and 'pizza'",
            "c) It requires a GPU",
            "d) It only works with English text"
        ],
        "answer": "b",
        "explanation": "All OHE vectors are orthogonal (cosine similarity = 0 for any pair), so no semantic relationships are captured."
    },
    {
        "question": "Bag of Words (BOW) represents a document by:",
        "options": [
            "a) Preserving word order and grammar",
            "b) Counting the frequency of each word in the vocabulary, ignoring order",
            "c) Creating embeddings for each word",
            "d) Using a neural network to encode meaning"
        ],
        "answer": "b",
        "explanation": "BOW counts word occurrences. 'The cat sat on the mat' → {the:2, cat:1, sat:1, on:1, mat:1}. Word order is lost."
    },
    {
        "question": "What does TF-IDF stand for?",
        "options": [
            "a) Text Frequency — Inverse Data Format",
            "b) Term Frequency — Inverse Document Frequency",
            "c) Token Formatting — Indexed Document Features",
            "d) Total Features — Internal Document Finder"
        ],
        "answer": "b",
        "explanation": "TF = how often a term appears in a document. IDF = log(total docs / docs containing term). TF-IDF highlights important, distinguishing words."
    },
    {
        "question": "In TF-IDF, a word that appears in EVERY document will have:",
        "options": [
            "a) A very high TF-IDF score",
            "b) A TF-IDF score of zero (IDF = log(1) = 0)",
            "c) An infinite TF-IDF score",
            "d) It depends on the word length"
        ],
        "answer": "b",
        "explanation": "IDF = log(N/N) = log(1) = 0 when a word appears in all documents. Common words like 'the' get near-zero TF-IDF scores."
    },
    {
        "question": "What is Word2Vec?",
        "options": [
            "a) A rule-based dictionary lookup",
            "b) A neural network that learns dense vector representations of words from large text corpora",
            "c) A text compression algorithm",
            "d) A database for storing words"
        ],
        "answer": "b",
        "explanation": "Word2Vec trains a shallow neural network to learn word embeddings where semantically similar words have similar vectors."
    },
    {
        "question": "Word2Vec has two architectures. What are they?",
        "options": [
            "a) Encoder and Decoder",
            "b) CBOW (Continuous Bag of Words) and Skip-gram",
            "c) CNN and RNN",
            "d) Forward and Backward"
        ],
        "answer": "b",
        "explanation": "CBOW predicts a word from its context words. Skip-gram predicts context words from a target word. Skip-gram is better for rare words."
    },
    {
        "question": "The famous Word2Vec analogy 'king - man + woman ≈ queen' demonstrates:",
        "options": [
            "a) Word2Vec memorizes facts",
            "b) Dense embeddings capture semantic relationships as vector arithmetic",
            "c) Word2Vec can only handle royalty terms",
            "d) This only works with English"
        ],
        "answer": "b",
        "explanation": "Vector arithmetic on Word2Vec embeddings captures analogies: the gender direction (man→woman) applies to king→queen."
    },
    {
        "question": "Which representation captures word ORDER?",
        "options": [
            "a) Bag of Words",
            "b) TF-IDF",
            "c) One-Hot Encoding",
            "d) None of the above — these are all order-agnostic"
        ],
        "answer": "d",
        "explanation": "BOW, TF-IDF, and OHE all discard word order. You need RNNs, Transformers, or n-gram models to capture order."
    },
    {
        "question": "Which is a SPARSE representation and which is DENSE?",
        "options": [
            "a) OHE/BOW/TF-IDF are sparse; Word2Vec/Transformer embeddings are dense",
            "b) All representations are dense",
            "c) All representations are sparse",
            "d) Sparse and dense are interchangeable terms"
        ],
        "answer": "a",
        "explanation": "Sparse: most values are 0 (OHE, BOW, TF-IDF). Dense: all values are non-zero, learned by neural networks (Word2Vec, BERT, etc.)."
    },
    {
        "question": "What is the typical dimensionality of Word2Vec vectors?",
        "options": ["a) 2-5", "b) 50-300", "c) 10,000+", "d) 1 million"],
        "answer": "b",
        "explanation": "Word2Vec commonly uses 100-300 dimensions. Much smaller than sparse representations (vocabulary-sized) but captures richer meaning."
    },
    {
        "question": "TF-IDF is commonly used in which retrieval method in RAG?",
        "options": [
            "a) Dense retrieval with FAISS",
            "b) BM25 sparse retrieval (conceptually similar to TF-IDF)",
            "c) ChromaDB vector search",
            "d) Neural reranking"
        ],
        "answer": "b",
        "explanation": "BM25 is an evolved version of TF-IDF that accounts for document length normalization and term saturation. Both are sparse methods."
    },
    {
        "question": "What problem does subword tokenization (BPE, WordPiece) solve that Word2Vec doesn't?",
        "options": [
            "a) Speed of training",
            "b) Out-of-vocabulary (OOV) words — unseen words can be split into known subwords",
            "c) Model size",
            "d) Color encoding"
        ],
        "answer": "b",
        "explanation": "Word2Vec can't handle words not in its vocabulary. BPE/WordPiece split unknown words into subwords (e.g., 'unhappiness' → 'un' + 'happiness')."
    },
    {
        "question": "Which encoding method would you use to create a simple document similarity search without neural networks?",
        "options": [
            "a) One-Hot Encoding",
            "b) TF-IDF with cosine similarity",
            "c) Random vectors",
            "d) ASCII encoding"
        ],
        "answer": "b",
        "explanation": "TF-IDF creates meaningful document vectors where cosine similarity effectively measures content similarity. It's the classic pre-neural approach."
    },
    {
        "question": "Compared to OHE, dense embeddings (Word2Vec, BERT) are better because:",
        "options": [
            "a) They use more memory",
            "b) They capture semantic relationships, use fixed dimensions regardless of vocab size, and enable similarity computation",
            "c) They are simpler to implement",
            "d) They don't require any training"
        ],
        "answer": "b",
        "explanation": "Dense embeddings encode meaning in a compact, fixed-size vector. Similar words have similar vectors, enabling semantic search and analogies."
    },
]


# %% [3] Section B: Transformer Architecture Deep Dive (Q16-Q30)
section_b = [
    {
        "question": "Self-attention computes three matrices from the input. What are they?",
        "options": [
            "a) Mean, Variance, Standard Deviation",
            "b) Query (Q), Key (K), Value (V)",
            "c) Input, Output, Hidden",
            "d) Encoder, Decoder, Attention"
        ],
        "answer": "b",
        "explanation": "Self-attention projects input into Q, K, V matrices. Attention(Q,K,V) = softmax(QK^T / √d_k) × V."
    },
    {
        "question": "In the attention formula softmax(QK^T / √d_k) × V, what is the purpose of dividing by √d_k?",
        "options": [
            "a) To speed up computation",
            "b) To prevent the dot products from growing too large, which would push softmax into regions with tiny gradients",
            "c) To reduce model size",
            "d) To add randomness"
        ],
        "answer": "b",
        "explanation": "The scaling factor √d_k prevents large dot products that would make softmax saturate (all attention on one token), causing vanishing gradients."
    },
    {
        "question": "What is 'masked' attention in the Transformer decoder?",
        "options": [
            "a) Attention that ignores padding tokens",
            "b) Attention that prevents the model from seeing future tokens during generation (causal mask)",
            "c) Attention with randomly dropped connections",
            "d) Attention that only looks at the first token"
        ],
        "answer": "b",
        "explanation": "Masked attention applies a causal mask (upper triangle = -∞) to prevent attending to future positions during autoregressive generation."
    },
    {
        "question": "What is 'cross-attention' in the Transformer?",
        "options": [
            "a) Attention between encoder and decoder — decoder queries attend to encoder keys/values",
            "b) Attention between two different models",
            "c) Self-attention applied twice",
            "d) Attention that crosses language boundaries"
        ],
        "answer": "a",
        "explanation": "Cross-attention: decoder provides Q, encoder provides K and V. This lets the decoder attend to relevant parts of the encoder's output."
    },
    {
        "question": "Layer Normalization in Transformers is used to:",
        "options": [
            "a) Remove layers from the model",
            "b) Normalize activations across features for each sample, stabilizing training",
            "c) Sort the layers by importance",
            "d) Compress the model"
        ],
        "answer": "b",
        "explanation": "LayerNorm normalizes across the feature dimension (not batch), making training more stable and faster to converge."
    },
    {
        "question": "What is the 'Feed-Forward Network' (FFN) in each Transformer block?",
        "options": [
            "a) The attention mechanism itself",
            "b) Two linear layers with a non-linear activation (e.g., ReLU/GELU) in between, applied independently to each position",
            "c) A recurrent layer",
            "d) The output projection"
        ],
        "answer": "b",
        "explanation": "FFN = Linear → ReLU/GELU → Linear. It processes each position independently, adding non-linear transformation capacity to the model."
    },
    {
        "question": "Residual connections (skip connections) in Transformers help by:",
        "options": [
            "a) Making the model smaller",
            "b) Allowing gradients to flow directly through the network, preventing vanishing gradients in deep models",
            "c) Removing unnecessary layers",
            "d) Adding regularization"
        ],
        "answer": "b",
        "explanation": "Residual connections add the input directly to the output (x + sublayer(x)), creating gradient highways that enable training very deep models."
    },
    {
        "question": "How many attention heads does a typical Transformer model have?",
        "options": [
            "a) Always exactly 1",
            "b) Usually 8-128, depending on model size (e.g., BERT-base has 12, GPT-3 has 96)",
            "c) Always 256",
            "d) The same as the vocabulary size"
        ],
        "answer": "b",
        "explanation": "Number of heads varies with model size. Each head has dimension d_model/num_heads. More heads = more parallel attention patterns."
    },
    {
        "question": "T5 (Text-to-Text Transfer Transformer) uses which architecture?",
        "options": ["a) Encoder only", "b) Decoder only", "c) Full encoder-decoder", "d) Neither"],
        "answer": "c",
        "explanation": "T5 uses the full encoder-decoder architecture, treating all NLP tasks as text-to-text (input text → output text)."
    },
    {
        "question": "Which model family is used for text GENERATION (completion)?",
        "options": [
            "a) Encoder-only (BERT)",
            "b) Decoder-only (GPT, Llama)",
            "c) Encoder-decoder (T5)",
            "d) Convolutional networks"
        ],
        "answer": "b",
        "explanation": "Decoder-only models (GPT, Llama, Mistral) excel at text generation via autoregressive next-token prediction."
    },
    {
        "question": "Which model family is best for text UNDERSTANDING (classification, NER)?",
        "options": [
            "a) Encoder-only (BERT, RoBERTa)",
            "b) Decoder-only (GPT)",
            "c) GAN-based models",
            "d) VAE models"
        ],
        "answer": "a",
        "explanation": "Encoder-only models process the full input bidirectionally, making them ideal for understanding tasks like classification and NER."
    },
    {
        "question": "The 'attention is all you need' innovation eliminated the need for:",
        "options": [
            "a) Neural networks entirely",
            "b) Recurrent connections (RNNs/LSTMs) and convolutions for sequence processing",
            "c) GPUs for training",
            "d) Training data"
        ],
        "answer": "b",
        "explanation": "Transformers replaced sequential RNN processing with parallelizable self-attention, dramatically speeding up training."
    },
    {
        "question": "What is the computational complexity of self-attention with respect to sequence length n?",
        "options": ["a) O(n)", "b) O(n log n)", "c) O(n²)", "d) O(n³)"],
        "answer": "c",
        "explanation": "Self-attention computes pairwise interactions between all n tokens: QK^T is an n×n matrix. This is why long contexts are expensive."
    },
    {
        "question": "KV-cache optimization in LLM inference helps by:",
        "options": [
            "a) Reducing model parameters",
            "b) Caching the Key and Value matrices from previous tokens so they don't need to be recomputed at each generation step",
            "c) Compressing the input text",
            "d) Using fewer attention heads"
        ],
        "answer": "b",
        "explanation": "During autoregressive generation, KV-cache stores computed K,V for previous tokens, avoiding redundant computation. Trades memory for speed."
    },
    {
        "question": "Flash Attention is an optimization that:",
        "options": [
            "a) Uses fewer attention heads",
            "b) Computes attention in a memory-efficient, tiled manner that reduces GPU memory reads/writes (IO-aware)",
            "c) Replaces attention with convolution",
            "d) Only works on CPUs"
        ],
        "answer": "b",
        "explanation": "Flash Attention tiles the attention computation to stay in fast GPU SRAM, reducing memory IO and enabling longer sequences without quadratic memory growth."
    },
]


# %% [4] Section C: LLM APIs & Practical Usage (Q31-Q50)
section_c = [
    {
        "question": "Which of these LLM API providers offers a free tier with fast inference?",
        "options": ["a) OpenAI", "b) Anthropic", "c) Groq", "d) Azure OpenAI"],
        "answer": "c",
        "explanation": "Groq offers free-tier API access with very fast inference using their custom LPU hardware. OpenAI and Anthropic are paid."
    },
    {
        "question": "In LangChain, how do you initialize a Groq-hosted LLM?",
        "options": [
            "a) ChatOpenAI(model='groq-model')",
            "b) ChatGroq(model='llama-3.1-8b-instant')",
            "c) Groq.create(model='...')",
            "d) LLM(provider='groq')"
        ],
        "answer": "b",
        "explanation": "LangChain provides ChatGroq class from langchain_groq. The API key is read from GROQ_API_KEY environment variable."
    },
    {
        "question": "What is the standard way to store API keys in a Python project?",
        "options": [
            "a) Hardcode them in the source code",
            "b) Store in a .env file and load with python-dotenv, ensuring .env is in .gitignore",
            "c) Put them in the README",
            "d) Store them in a public GitHub repository"
        ],
        "answer": "b",
        "explanation": "API keys go in .env files loaded by dotenv. Never hardcode keys or commit them to version control."
    },
    {
        "question": "What is the purpose of os.getenv('GROQ_API_KEY') vs hardcoding the key?",
        "options": [
            "a) It's faster",
            "b) Security — keys are not in source code and can vary per environment (dev/staging/prod)",
            "c) It's required by Python syntax",
            "d) It saves memory"
        ],
        "answer": "b",
        "explanation": "Environment variables keep secrets out of code, enable different keys per environment, and prevent accidental commits to git."
    },
    {
        "question": "What happens if you call an LLM API without wrapping it in try/except?",
        "options": [
            "a) Nothing — APIs never fail",
            "b) Unhandled exceptions crash the program on rate limits, network errors, or invalid keys",
            "c) The API auto-retries",
            "d) Python automatically catches all errors"
        ],
        "answer": "b",
        "explanation": "LLM API calls can fail (rate limits, network issues, auth errors). Always wrap in try/except for graceful error handling."
    },
    {
        "question": "Ollama is used to:",
        "options": [
            "a) Access OpenAI's API",
            "b) Run open-source LLMs locally on your machine without API keys",
            "c) Create vector databases",
            "d) Fine-tune models in the cloud"
        ],
        "answer": "b",
        "explanation": "Ollama downloads and runs open-source models (Llama, Mistral, etc.) locally. Free, private, no API key needed."
    },
    {
        "question": "What is OpenRouter?",
        "options": [
            "a) A networking library",
            "b) A unified API that routes to many LLM providers (OpenAI, Anthropic, Google, open-source) through a single endpoint",
            "c) A vector database",
            "d) A web framework"
        ],
        "answer": "b",
        "explanation": "OpenRouter provides a single API to access 100+ models from different providers, with some free models available."
    },
    {
        "question": "In LangChain, what is the difference between ChatOpenAI and OpenAI classes?",
        "options": [
            "a) They are identical",
            "b) ChatOpenAI is for chat models (message-based); OpenAI is for completion models (text-based, deprecated)",
            "c) OpenAI is newer",
            "d) ChatOpenAI only works with GPT-4"
        ],
        "answer": "b",
        "explanation": "ChatOpenAI handles chat message format (system/human/ai roles). The older OpenAI class was for text completions, now largely deprecated."
    },
    {
        "question": "What does 'max_tokens' control in an LLM API call?",
        "options": [
            "a) The maximum input length",
            "b) The maximum number of tokens in the generated response",
            "c) The model's parameter count",
            "d) The API rate limit"
        ],
        "answer": "b",
        "explanation": "max_tokens limits the output length. If set too low, responses get cut off. If too high, you pay for unnecessary capacity."
    },
    {
        "question": "What is the benefit of LLM streaming?",
        "options": [
            "a) Reduces total generation time",
            "b) Tokens are sent to the client as they're generated, improving perceived responsiveness",
            "c) Reduces API cost",
            "d) Improves model accuracy"
        ],
        "answer": "b",
        "explanation": "Streaming shows tokens incrementally instead of waiting for the complete response. Total time is the same, but UX is much better."
    },
    {
        "question": "Which HuggingFace class is used for embedding models in LangChain?",
        "options": [
            "a) HuggingFaceHub",
            "b) HuggingFaceEmbeddings (from langchain_huggingface)",
            "c) HuggingFaceTokenizer",
            "d) HuggingFacePipeline"
        ],
        "answer": "b",
        "explanation": "HuggingFaceEmbeddings from langchain_huggingface runs sentence-transformers models locally for free."
    },
    {
        "question": "Which model from HuggingFace is commonly used as the DEFAULT embedding model for RAG development?",
        "options": [
            "a) bert-base-uncased",
            "b) all-MiniLM-L6-v2",
            "c) gpt2",
            "d) t5-small"
        ],
        "answer": "b",
        "explanation": "all-MiniLM-L6-v2 produces 384d vectors, runs locally for free, is fast, and provides good quality. It's the standard default for development."
    },
    {
        "question": "What is the difference between a 'system' message and a 'human' message in chat models?",
        "options": [
            "a) They are the same",
            "b) System sets the LLM's behavior/persona; human contains the user's input/question",
            "c) System is for error messages",
            "d) Human messages are optional"
        ],
        "answer": "b",
        "explanation": "System messages set context, rules, and persona ('You are a helpful assistant. Only use provided context.'). Human messages contain the user's query."
    },
    {
        "question": "A company wants to run LLMs without sending data to external APIs (privacy concern). What should they use?",
        "options": [
            "a) OpenAI API",
            "b) Self-hosted open-source models via Ollama or vLLM",
            "c) Google Gemini API",
            "d) Anthropic Claude API"
        ],
        "answer": "b",
        "explanation": "Self-hosted models (Ollama, vLLM, TGI) keep data on-premises. All cloud APIs send data to external servers."
    },
    {
        "question": "What is 'rate limiting' in LLM APIs?",
        "options": [
            "a) How fast the model generates tokens",
            "b) A limit on how many requests you can make per minute/day, enforced by the API provider",
            "c) The maximum model size",
            "d) A limit on prompt length"
        ],
        "answer": "b",
        "explanation": "Rate limits prevent abuse. Groq free tier has RPM (requests per minute) limits. Exceeding them returns 429 errors. Use exponential backoff."
    },
    {
        "question": "What is the advantage of using LangChain wrappers for LLM APIs over raw API calls?",
        "options": [
            "a) They are always faster",
            "b) Unified interface — switch between providers (OpenAI, Groq, Google) with minimal code changes",
            "c) They bypass rate limits",
            "d) They don't require API keys"
        ],
        "answer": "b",
        "explanation": "LangChain abstracts provider differences. Switching from ChatGroq to ChatOpenAI requires changing one line, not rewriting the entire integration."
    },
    {
        "question": "Which is the correct way to handle an LLM API failure in Python?",
        "options": [
            "a) Ignore the error",
            "b) Wrap in try/except, log the error, and return a graceful fallback message",
            "c) Restart the entire application",
            "d) Increase max_tokens"
        ],
        "answer": "b",
        "explanation": "Graceful error handling catches exceptions, logs them for debugging, and returns a user-friendly message instead of crashing."
    },
    {
        "question": "What is a 'callback' in LangChain?",
        "options": [
            "a) A phone call to the user",
            "b) A hook that lets you log, monitor, or modify chain execution at various stages (on_llm_start, on_chain_end, etc.)",
            "c) A retry mechanism",
            "d) A database query"
        ],
        "answer": "b",
        "explanation": "Callbacks hook into chain execution for logging, monitoring, cost tracking, and debugging. Useful in production for observability."
    },
    {
        "question": "You're building a production GenAI app and need to choose between paying for GPT-4o and using free Groq with Llama 3.1 8B. Which factors matter most?",
        "options": [
            "a) Only the model name matters",
            "b) Task complexity, accuracy requirements, cost budget, rate limits, and latency needs",
            "c) Always choose the most expensive option",
            "d) The programming language used"
        ],
        "answer": "b",
        "explanation": "Model selection is a trade-off: GPT-4o has better reasoning but costs more. Llama 8B on Groq is free and fast but less capable for complex tasks."
    },
    {
        "question": "What is the LCEL (LangChain Expression Language) pipe operator used for?",
        "options": [
            "a) Mathematical division",
            "b) Chaining components: output of one becomes input of the next (prompt | llm | parser)",
            "c) File system operations",
            "d) Unix shell commands"
        ],
        "answer": "b",
        "explanation": "LCEL's | operator chains runnables: prompt | llm | parser means format→generate→parse. Clean, composable, and supports streaming/async."
    },
]


# %% [5] Run the full quiz
if __name__ == "__main__":
    all_questions = section_a + section_b + section_c

    print("=" * 70)
    print("📝 GenAI L2 Foundations — MCQ Practice (50 Questions)")
    print("=" * 70)
    print("""
Sections:
  A: Text Encoding & Representation     (Q1-Q15)
  B: Transformer Architecture Deep Dive  (Q16-Q30)
  C: LLM APIs & Practical Usage          (Q31-Q50)

Enter your answer as a/b/c/d, or 'q' to quit.
""")

    run_quiz(all_questions)
