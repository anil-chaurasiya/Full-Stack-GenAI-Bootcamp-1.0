"""Class 30: parse complex PDFs into LangChain Documents.

Real PDFs can contain selectable text, scanned pages, embedded images, and
tables. No single parser is reliable for every PDF, so this example combines:

* PyMuPDF (``fitz``) for normal text, page rendering, and embedded images.
* Tesseract through ``pytesseract`` for OCR when an image contains text.
* pdfplumber for table extraction.
* LangChain ``Document`` objects as the common output format for RAG.

OCR is optional. If Tesseract is unavailable, the script keeps parsing the
other content and records an explanatory status in the output document.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document


def run_ocr(image_path: Path) -> str:
    """Extract text from an image, returning a diagnostic instead of failing."""
    try:
        import pytesseract
        from PIL import Image

        return pytesseract.image_to_string(Image.open(image_path)).strip()
    except Exception as error:  # OCR depends on an external executable.
        return f"[OCR_SKIPPED_OR_FAILED: {error}]"


def extract_text_and_images(pdf_path: Path, output_dir: Path) -> tuple[list[dict], list[dict]]:
    """Extract selectable text, rendered page images, and embedded images."""
    import fitz

    image_dir = output_dir / "extracted_images"
    page_image_dir = output_dir / "page_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    page_image_dir.mkdir(parents=True, exist_ok=True)

    page_records: list[dict] = []
    image_records: list[dict] = []

    with fitz.open(pdf_path) as document:
        for page_index, page in enumerate(document):
            page_number = page_index + 1
            page_image_path = page_image_dir / f"page_{page_number:03d}.png"
            page.get_pixmap(matrix=fitz.Matrix(2, 2)).save(str(page_image_path))

            page_records.append(
                {
                    "page_number": page_number,
                    "text": page.get_text("text").strip(),
                    "image_count": len(page.get_images(full=True)),
                    "ocr_text": run_ocr(page_image_path),
                    "page_image_path": str(page_image_path),
                }
            )

            for image_index, image in enumerate(page.get_images(full=True), start=1):
                base_image = document.extract_image(image[0])
                image_path = image_dir / (
                    f"page_{page_number:03d}_image_{image_index}.{base_image['ext']}"
                )
                image_path.write_bytes(base_image["image"])
                image_records.append(
                    {
                        "page_number": page_number,
                        "image_index": image_index,
                        "image_path": str(image_path),
                        "image_ext": base_image["ext"],
                        "image_ocr_text": run_ocr(image_path),
                    }
                )

    return page_records, image_records


def extract_tables(pdf_path: Path) -> list[dict]:
    """Extract tables with pdfplumber and serialize each table as Markdown/CSV."""
    import pdfplumber

    table_records: list[dict] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            for table_index, table in enumerate(page.extract_tables(), start=1):
                if not table:
                    continue
                cleaned = [
                    [cell.strip() if isinstance(cell, str) else cell for cell in row]
                    for row in table
                ]
                try:
                    frame = pd.DataFrame(cleaned[1:], columns=cleaned[0])
                except Exception:
                    frame = pd.DataFrame(cleaned)
                table_records.append(
                    {
                        "page_number": page_number,
                        "table_index": table_index,
                        "markdown": frame.to_markdown(index=False),
                        "csv": frame.to_csv(index=False),
                    }
                )
    return table_records


def make_documents(
    pdf_path: Path,
    page_records: list[dict],
    image_records: list[dict],
    table_records: list[dict],
) -> list[Document]:
    """Convert extracted content into typed LangChain Documents with metadata."""
    documents: list[Document] = []

    for page in page_records:
        documents.append(
            Document(
                page_content=(
                    f"PAGE {page['page_number']}\n\n"
                    f"SELECTABLE TEXT:\n{page['text']}\n\n"
                    f"OCR TEXT:\n{page['ocr_text']}"
                ),
                metadata={
                    "source": str(pdf_path),
                    "page_number": page["page_number"],
                    "content_type": "page_text_plus_ocr",
                    "image_count": page["image_count"],
                    "page_image_path": page["page_image_path"],
                },
            )
        )

    for table in table_records:
        documents.append(
            Document(
                page_content=(
                    f"TABLE FOUND ON PAGE {table['page_number']}\n"
                    f"TABLE INDEX: {table['table_index']}\n\n"
                    f"{table['markdown']}"
                ),
                metadata={
                    "source": str(pdf_path),
                    "page_number": table["page_number"],
                    "content_type": "table",
                    "table_index": table["table_index"],
                },
            )
        )

    for image in image_records:
        documents.append(
            Document(
                page_content=(
                    f"IMAGE FOUND ON PAGE {image['page_number']}\n"
                    f"IMAGE INDEX: {image['image_index']}\n\n"
                    f"IMAGE OCR TEXT:\n{image['image_ocr_text']}"
                ),
                metadata={
                    "source": str(pdf_path),
                    "page_number": image["page_number"],
                    "content_type": "image",
                    "image_index": image["image_index"],
                    "image_path": image["image_path"],
                },
            )
        )

    return documents


def main() -> None:
    """Parse one PDF and save machine-readable extraction results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="PDF to parse")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "parsed_pdf_output",
        help="Directory for page images and JSON output",
    )
    args = parser.parse_args()
    if not args.pdf.exists():
        parser.error(f"PDF does not exist: {args.pdf}")

    pages, images = extract_text_and_images(args.pdf, args.output_dir)
    tables = extract_tables(args.pdf)
    documents = make_documents(args.pdf, pages, images, tables)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "parsed_records.json").write_text(
        json.dumps(
            {"pages": pages, "images": images, "tables": tables},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"Pages: {len(pages)} | Images: {len(images)} | Tables: {len(tables)}")
    print(f"LangChain Documents: {len(documents)}")
    if documents:
        print(f"First document metadata: {documents[0].metadata}")


if __name__ == "__main__":
    main()