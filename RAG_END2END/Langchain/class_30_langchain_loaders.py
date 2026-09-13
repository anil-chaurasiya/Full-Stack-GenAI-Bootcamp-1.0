"""Class 30: load common data sources with LangChain.

This module demonstrates the first step of a RAG pipeline: converting files
into LangChain ``Document`` objects. Every document contains two things:

* ``page_content``: the text that can later be split and embedded.
* ``metadata``: source information that helps with citations and debugging.

The examples create small local files so the script can run without requiring
private course files. To try a real PDF, pass its path with ``--pdf``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from langchain_community.document_loaders import (
    CSVLoader,
    DirectoryLoader,
    TextLoader,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEMO_DATA_DIR = SCRIPT_DIR / "class_30_demo_data"


def write_demo_files(data_dir: Path = DEMO_DATA_DIR) -> dict[str, Path]:
    """Create small text, CSV, and JSON examples and return their paths."""
    data_dir.mkdir(parents=True, exist_ok=True)

    text_path = data_dir / "sample.txt"
    text_path.write_text(
        "Class 30: Data Parsing for RAG\n\n"
        "A loader reads a source file and returns LangChain Documents.\n"
        "The next RAG stages split, embed, and retrieve this content.\n",
        encoding="utf-8",
    )

    csv_path = data_dir / "concepts.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(
            [
                ["Concept", "Description", "Category"],
                ["RAG", "Retrieval-Augmented Generation", "Architecture"],
                ["Chunking", "Splitting documents for retrieval", "Processing"],
                ["Embedding", "Representing text as vectors", "Representation"],
            ]
        )

    json_path = data_dir / "topics.json"
    json_path.write_text(
        json.dumps(
            {
                "topics": [
                    {
                        "name": "RAG Pipeline",
                        "description": "Load, split, embed, retrieve, and generate.",
                    },
                    {
                        "name": "Vector Database",
                        "description": "Store and search embedding vectors.",
                    },
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {"text": text_path, "csv": csv_path, "json": json_path}


def show_documents(title: str, documents: list, preview_length: int = 220) -> None:
    """Print a compact, readable summary of loaded Documents."""
    print(f"\n{title}: {len(documents)} document(s)")
    for index, document in enumerate(documents, start=1):
        preview = " ".join(document.page_content.split())[:preview_length]
        print(f"  {index}. {preview}")
        print(f"     metadata: {document.metadata}")


def load_text_file(path: Path) -> list:
    """Load one plain-text file with ``TextLoader``."""
    return TextLoader(str(path), encoding="utf-8").load()


def load_csv_file(path: Path) -> list:
    """Load each CSV row as a separate Document with ``CSVLoader``."""
    return CSVLoader(str(path)).load()


def load_json_descriptions(path: Path) -> list:
    """Load JSON descriptions using JSONLoader when the optional ``jq`` package exists."""
    try:
        from langchain_community.document_loaders import JSONLoader
        return JSONLoader(
            file_path=str(path),
            jq_schema=".topics[].description",
            text_content=False,
        ).load()
    except ImportError as error:
        print(f"\nJSONLoader skipped: install the optional 'jq' package ({error}).")
        return []


def load_directory(data_dir: Path) -> list:
    """Load all text files in a directory with ``DirectoryLoader``."""
    loader = DirectoryLoader(
        str(data_dir),
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    return loader.load()


def load_pdf(path: Path) -> list:
    """Load one PDF page per Document with the optional PyPDF integration."""
    try:
        from langchain_community.document_loaders import PyPDFLoader
    except ImportError as error:
        print(f"\nPDF loading skipped: install PDF dependencies ({error}).")
        return []

    return PyPDFLoader(str(path)).load()


def main() -> None:
    """Run the loader demonstrations and optionally inspect a real PDF."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdf",
        type=Path,
        help="Optional PDF path to load with LangChain's PyPDFLoader.",
    )
    args = parser.parse_args()

    paths = write_demo_files()
    show_documents("TextLoader", load_text_file(paths["text"]))
    show_documents("CSVLoader", load_csv_file(paths["csv"]))
    show_documents("JSONLoader", load_json_descriptions(paths["json"]))
    show_documents("DirectoryLoader", load_directory(DEMO_DATA_DIR))

    if args.pdf:
        if not args.pdf.exists():
            parser.error(f"PDF does not exist: {args.pdf}")
        show_documents("PyPDFLoader", load_pdf(args.pdf))

    print("\nKey idea: loaders normalize different sources into Documents.")
    print("The next RAG step is usually text splitting, followed by embeddings.")


if __name__ == "__main__":
    main()