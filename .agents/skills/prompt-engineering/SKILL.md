---
name: prompt-engineering
description: Design and optimize prompts for LLMs including zero-shot, few-shot, chain-of-thought, and structured output prompting techniques using LangChain prompt templates.
---

# Prompt Engineering Skill

Use this skill when the user wants to create, improve, or debug prompts for LLM interactions.

## Prompting Techniques

### 1. Zero-Shot Prompting
No examples provided — rely on the model's pre-trained knowledge.

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert data scientist."),
    ("human", "{question}")
])
```

### 2. Few-Shot Prompting
Provide examples to guide the model's output format and reasoning.

```python
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

examples = [
    {"input": "What is 2+2?", "output": "The answer is 4."},
    {"input": "What is the capital of France?", "output": "The capital of France is Paris."},
]

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

final_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer concisely."),
    few_shot_prompt,
    ("human", "{input}"),
])
```

### 3. Chain-of-Thought (CoT)
Guide the model to reason step by step.

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a logical reasoning expert. Think step by step before giving your final answer."),
    ("human", """
Question: {question}

Let's think through this step by step:
1. First, identify the key information
2. Then, apply the relevant logic
3. Finally, state your conclusion

Step-by-step reasoning:""")
])
```

### 4. Structured Output
Force the model to respond in a specific format.

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a data extraction assistant. 
Always respond in the following JSON format:
{{
    "answer": "your answer here",
    "confidence": "high/medium/low",
    "reasoning": "brief explanation"
}}"""),
    ("human", "{question}")
])
```

### 5. Role-Based Prompting
```python
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a senior Python developer with 15 years of experience.
You specialize in:
- Clean, maintainable code
- Performance optimization
- Best practices and design patterns

When reviewing code, always provide:
1. Issues found
2. Suggested fixes
3. Best practice recommendations"""),
    ("human", "Review this code:\n```python\n{code}\n```")
])
```

## Prompt Management with JSON

Store prompts externally for reuse (as done in Class-37):

```json
// prompts.json
{
    "summarize": {
        "system": "You are a summarization expert.",
        "template": "Summarize the following text in {num_sentences} sentences:\n\n{text}"
    },
    "qa": {
        "system": "Answer questions based on the given context only.",
        "template": "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    }
}
```

```python
import json

with open("prompts.json", "r") as f:
    prompts = json.load(f)

prompt = ChatPromptTemplate.from_messages([
    ("system", prompts["qa"]["system"]),
    ("human", prompts["qa"]["template"])
])
```

## LangChain Output Parsers

### String Output
```python
from langchain_core.output_parsers import StrOutputParser

chain = prompt | llm | StrOutputParser()
result = chain.invoke({"question": "What is AI?"})
```

### JSON Output
```python
from langchain_core.output_parsers import JsonOutputParser

chain = prompt | llm | JsonOutputParser()
result = chain.invoke({"question": "What is AI?"})
```

## Best Practices
- Be **specific and explicit** — vague prompts get vague answers
- Use **delimiters** (triple backticks, XML tags) to separate sections
- Specify the **output format** you want
- Include **negative instructions** ("Do NOT include...", "Do NOT make up...")
- Test prompts with multiple inputs to check robustness
- Use **temperature=0** for deterministic/factual tasks, higher for creative tasks
- Keep system prompts focused — one role per prompt
