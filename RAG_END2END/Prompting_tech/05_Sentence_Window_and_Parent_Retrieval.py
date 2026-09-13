"""Context-preserving retrieval: sentence windows and parent documents.

Based on Class 37's sentence-window and ParentDocumentRetriever practicals.
Both methods solve the same RAG tension: small chunks are precise for search,
but larger context is clearer for an LLM or a human reader.

Run: .venv_genai/bin/python RAG_END2END/Prompting_tech/05_Sentence_Window_and_Parent_Retrieval.py
Needs: pip install numpy scikit-learn
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# %% [1] Source text and basic data structures
TEXT = (
    "Llama 2-Chat first undergoes supervised fine-tuning with demonstrations written by annotators. "
    "The team then collects human preference data by comparing candidate responses. "
    "A reward model is trained to score which candidate response people prefer. "
    "Reinforcement learning with human feedback uses that reward model to improve alignment. "
    "Safety evaluation and red teaming identify unsafe behaviours for further improvement."
)


@dataclass(frozen=True)
class SearchUnit:
    """A small searchable text plus the larger context to return after search."""

    searchable_text: str
    returned_text: str
    metadata: dict[str, int | str]


def split_sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]


class SearchIndex:
    def __init__(self, units: list[SearchUnit]):
        self.units = units
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.vectors = self.vectorizer.fit_transform([unit.searchable_text for unit in units]).toarray()

    def search(self, query: str, k: int = 3) -> list[tuple[float, SearchUnit]]:
        query_vector = self.vectorizer.transform([query]).toarray()[0]
        if np.linalg.norm(query_vector) == 0:
            return []
        scores = self.vectors @ query_vector / (np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(query_vector))
        indices = np.argsort(scores)[::-1][:k]
        return [(float(scores[i]), self.units[i]) for i in indices if scores[i] > 0]


def display(title: str, results: list[tuple[float, SearchUnit]]) -> None:
    print(f"\n{'=' * 88}\n{title}\n{'=' * 88}")
    for rank, (score, unit) in enumerate(results, start=1):
        print(f"{rank}. score={score:.3f} | {unit.metadata}")
        print("   matched:", unit.searchable_text)
        print("   returned:", unit.returned_text)


# %% [2] Sentence window retrieval: retrieve one sentence, return nearby sentences
def build_sentence_windows(text: str, window_size: int = 1) -> list[SearchUnit]:
    """Embed each sentence separately while storing its surrounding window.

    ``window_size=1`` means previous + matching + next sentence. The class
    notebook stores this context in metadata; here it is explicit as
    ``returned_text`` so the distinction is easy to inspect.
    """

    sentences = split_sentences(text)
    units: list[SearchUnit] = []
    for index, sentence in enumerate(sentences):
        start = max(0, index - window_size)
        end = min(len(sentences), index + window_size + 1)
        units.append(
            SearchUnit(
                searchable_text=sentence,
                returned_text=" ".join(sentences[start:end]),
                metadata={"sentence_index": index, "window_start": start, "window_end": end - 1},
            )
        )
    return units


def demonstrate_sentence_windows() -> None:
    query = "How did human preference data improve alignment?"
    units = build_sentence_windows(TEXT, window_size=1)
    index = SearchIndex(units)
    results = index.search(query)
    display("1. Sentence-window retrieval", results)
    print("\nNotice: only the short 'matched' sentence was indexed, but the LLM receives "
          "the surrounding explanation in 'returned'.")


# %% [3] Parent-document retrieval: retrieve a small child, return its large parent
def split_into_children(text: str, parent_id: str, words_per_child: int = 12) -> list[SearchUnit]:
    """Make small child chunks and record their parent identifier.

    A production ParentDocumentRetriever stores children in a vector DB and
    parents in a separate docstore. This tiny version uses a Python dictionary
    for the parent docstore to expose that relationship directly.
    """

    words = text.split()
    return [
        SearchUnit(
            searchable_text=" ".join(words[start:start + words_per_child]),
            returned_text="",  # Filled from the parent docstore after retrieval.
            metadata={"child_number": number, "parent_id": parent_id},
        )
        for number, start in enumerate(range(0, len(words), words_per_child), start=1)
    ]


def demonstrate_parent_documents() -> None:
    parents = {
        "parent-fine-tuning": TEXT,
        "parent-safety": (
            "Safety work includes red teaming, safety-specific annotation, and evaluations. "
            "The findings are used for iterative fine-tuning and safer model behaviour."
        ),
    }
    children = [child for parent_id, parent_text in parents.items() for child in split_into_children(parent_text, parent_id)]
    child_index = SearchIndex(children)
    query = "What role does the reward model play?"
    child_results = child_index.search(query, k=3)

    print(f"\n{'=' * 88}\n2. Child search, then parent return\n{'=' * 88}")
    seen_parent_ids: set[str] = set()
    for score, child in child_results:
        parent_id = str(child.metadata["parent_id"])
        print(f"\nMatched child (score={score:.3f}, {child.metadata}):\n{child.searchable_text}")
        if parent_id not in seen_parent_ids:
            seen_parent_ids.add(parent_id)
            print(f"\nReturned parent {parent_id}:\n{parents[parent_id]}")

    print("\nTrade-off: children improve retrieval precision; parents give enough context "
          "for a grounded answer. Deduplicate parent ids when several children match.")


def main() -> None:
    print("CONTEXT-PRESERVING RETRIEVAL")
    demonstrate_sentence_windows()
    demonstrate_parent_documents()


if __name__ == "__main__":
    main()
