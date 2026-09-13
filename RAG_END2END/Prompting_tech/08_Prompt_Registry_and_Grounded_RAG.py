"""Versioned prompt registries and grounded RAG prompts — Class 37 lesson.

The classroom notebook loads ``prompt.json`` / ``prompts.json`` and rebuilds
LangChain templates. This self-contained version stores the same idea in a
Python registry, validates it, then formats prompts without calling an LLM.

Run: .venv_genai/bin/python RAG_END2END/Prompting_tech/08_Prompt_Registry_and_Grounded_RAG.py
Needs: pip install langchain-core
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.prompts import ChatPromptTemplate


# %% [1] Prompt registry: in production this usually lives in JSON/YAML/a database
PROMPT_REGISTRY: dict[str, dict[str, Any]] = {
    "rag_prompt": {
        "version": "1.0.0",
        "template_format": "jinja2",
        "messages": [
            {
                "role": "system",
                "template": (
                    "You are a RAG assistant. Answer only from the provided context. "
                    "If the answer is not available, say: 'I do not have enough information.'"
                ),
            },
            {"role": "human", "template": "Context:\n{{ context }}\n\nQuestion:\n{{ question }}"},
        ],
    },
    "summarization_prompt": {
        "version": "1.0.0",
        "template_format": "jinja2",
        "messages": [
            {"role": "system", "template": "You are a professional text summarizer."},
            {"role": "human", "template": "Summarize this in {{ num_points }} bullet points:\n\n{{ text }}"},
        ],
    },
    "classification_prompt": {
        "version": "1.0.0",
        "template_format": "jinja2",
        "messages": [
            {"role": "system", "template": "You are a support classifier. Return only one category."},
            {
                "role": "human",
                "template": "Classify into one category:\n{% for category in categories %}- {{ category }}\n{% endfor %}\nQuery: {{ query }}",
            },
        ],
    },
}


@dataclass(frozen=True)
class PromptInfo:
    name: str
    version: str
    input_variables: list[str]


def load_prompt(name: str) -> tuple[ChatPromptTemplate, PromptInfo]:
    """Validate a registry entry and turn it into a reusable LangChain template."""

    if name not in PROMPT_REGISTRY:
        choices = ", ".join(PROMPT_REGISTRY)
        raise ValueError(f"Prompt '{name}' was not found. Available: {choices}")

    config = PROMPT_REGISTRY[name]
    messages = config.get("messages")
    template_format = config.get("template_format")
    if not isinstance(messages, list) or not messages or template_format not in {"f-string", "jinja2"}:
        raise ValueError(f"Prompt '{name}' has an invalid registry configuration.")

    prompt = ChatPromptTemplate.from_messages(
        [(message["role"], message["template"]) for message in messages],
        template_format=template_format,
    )
    return prompt, PromptInfo(name, str(config.get("version", "unversioned")), sorted(prompt.input_variables))


def print_prompt_value(title: str, prompt_value) -> None:
    print(f"\n{'=' * 88}\n{title}\n{'=' * 88}")
    for message in prompt_value.to_messages():
        print(f"\n{message.type.upper()}:\n{message.content}")


# %% [2] Grounded RAG: the prompt controls how retrieved evidence is used
def grounded_rag_demo() -> None:
    prompt, info = load_prompt("rag_prompt")
    print("Loaded:", info)

    context = (
        "Employee leave policy: Employees receive 24 paid leaves every year. "
        "A maximum of 10 unused leaves can be carried forward."
    )
    formatted = prompt.invoke({"context": context, "question": "How many paid leaves do employees receive?"})
    print_prompt_value("1. Grounded RAG prompt", formatted)

    print("\nImportant: a strong prompt reduces hallucination but cannot repair bad retrieval. "
          "Always inspect the retrieved context before calling the model.")


# %% [3] Reuse registry templates with different inputs
def reusable_templates_demo() -> None:
    summary_prompt, summary_info = load_prompt("summarization_prompt")
    formatted_summary = summary_prompt.invoke(
        {"num_points": 3, "text": "RAG retrieves source chunks and gives them to an LLM before it answers."}
    )
    print("\nLoaded:", summary_info)
    print_prompt_value("2. Summarization template", formatted_summary)

    classifier_prompt, classifier_info = load_prompt("classification_prompt")
    formatted_classifier = classifier_prompt.invoke(
        {"categories": ["billing", "technical", "account"], "query": "My payment failed but the amount was deducted."}
    )
    print("\nLoaded:", classifier_info)
    print_prompt_value("3. Classification template with a Jinja loop", formatted_classifier)


# %% [4] Prompt-management habits to adopt before production
def prompt_management_checklist() -> None:
    print("""
Prompt management checklist:
1. Give every prompt a stable name and semantic version.
2. Store the template outside business logic (JSON/YAML, a database, or LangSmith Hub).
3. Log the prompt name/version with each model result.
4. Test a candidate prompt against the same evaluation questions before replacing it.
5. Keep secrets and user data OUT of the prompt registry.

Optional LangSmith flow from the notebook:
    client.push_prompt("my-prompt-name", object=prompt)
    prompt = client.pull_prompt("my-prompt-name")
This requires a configured LangSmith account; it is not run by this lesson.
""")


def main() -> None:
    print("PROMPT REGISTRY + GROUNDED RAG")
    grounded_rag_demo()
    reusable_templates_demo()
    prompt_management_checklist()


if __name__ == "__main__":
    main()
