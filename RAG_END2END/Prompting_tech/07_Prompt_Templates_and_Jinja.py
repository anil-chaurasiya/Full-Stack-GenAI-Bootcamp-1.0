"""PromptTemplate, ChatPromptTemplate, and Jinja templates — Class 37 lesson.

Adapted from ``Class-37-08-Aug-2026-prompting-advance-retriever/prompting.ipynb``.
This lesson only FORMATS prompts. It deliberately makes no LLM/API call, so you
can debug the exact text and roles that would be sent to a model first.

Run: .venv_genai/bin/python RAG_END2END/Prompting_tech/07_Prompt_Templates_and_Jinja.py
Needs: pip install langchain-core
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate


def print_messages(title: str, prompt_value) -> None:
    """Show role-labelled messages; this is the best first prompt-debug step."""

    print(f"\n{'=' * 88}\n{title}\n{'=' * 88}")
    for message in prompt_value.to_messages():
        print(f"\n{message.type.upper()}:\n{message.content}")


# %% [1] PromptTemplate: one formatted text string
def plain_template_demo() -> None:
    template = PromptTemplate.from_template(
        """You are an AI instructor.

Explain {topic} to {audience}.

Requirements:
- Use simple English.
- Include one practical example.
- Keep the answer under {word_limit} words."""
    )
    formatted = template.invoke(
        {"topic": "Vector Database", "audience": "beginner developers", "word_limit": 120}
    )
    print("\n1. PromptTemplate input variables:", template.input_variables)
    print("\nFormatted single text prompt:\n", formatted.to_string())


# %% [2] ChatPromptTemplate: preserve system and user roles
def chat_template_demo() -> None:
    template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an experienced AI instructor. Use simple English."),
            ("human", "Explain {topic} to {audience}. Include one example."),
        ]
    )
    formatted = template.invoke({"topic": "Vector Database", "audience": "beginner developers"})
    print_messages("2. ChatPromptTemplate: roles are part of the prompt", formatted)


# %% [3] Jinja conditions: adapt one prompt without building many nearly-identical prompts
def jinja_condition_demo() -> None:
    template = PromptTemplate.from_template(
        """You are an AI instructor.

Explain {{ topic }} to {{ audience }}.

{% if include_example %}Include one practical example.{% endif %}

{% if level == "beginner" %}
Use very simple English and avoid complex terminology.
{% elif level == "advanced" %}
Include technical details and architecture.
{% else %}
Use moderate technical depth.
{% endif %}""",
        template_format="jinja2",
    )
    beginner = template.invoke(
        {"topic": "RAG", "audience": "Python developers", "include_example": True, "level": "beginner"}
    )
    advanced = template.invoke(
        {"topic": "RAG", "audience": "ML engineers", "include_example": False, "level": "advanced"}
    )
    print("\n3. Jinja beginner version:\n", beginner.to_string())
    print("\n3. Jinja advanced version:\n", advanced.to_string())
    print("\nJinja is valuable when a safe, controlled variable changes prompt instructions.")


# %% [4] Jinja loops: render a variable-sized list of policy rules
def jinja_loop_demo() -> None:
    template = ChatPromptTemplate.from_messages(
        [
            ("system", """You are an enterprise AI assistant.

Follow these rules:
{% for rule in rules %}- {{ rule }}
{% endfor %}"""),
            ("human", """Question: {{ question }}

{% if output_format == "json" %}Return valid JSON only.
{% else %}Return Markdown.
{% endif %}"""),
        ],
        template_format="jinja2",
    )
    formatted = template.invoke(
        {
            "rules": ["Do not fabricate information", "Keep the answer concise", "Use only supplied context"],
            "question": "What is RAG?",
            "output_format": "json",
        }
    )
    print_messages("4. Jinja loop + condition in a chat prompt", formatted)


def main() -> None:
    print("PROMPT TEMPLATE FOUNDATION: inspect before calling an LLM")
    plain_template_demo()
    chat_template_demo()
    jinja_condition_demo()
    jinja_loop_demo()
    print("\nDebug checklist: verify input variables, rendered conditions, message roles, "
          "and final text. Then send this prompt to your chosen model.")


if __name__ == "__main__":
    main()
