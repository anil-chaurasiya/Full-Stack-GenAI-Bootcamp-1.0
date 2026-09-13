#!/usr/bin/env python
# coding: utf-8

# In[1]:


print("all ok")


# PDF
#  ↓
# Upload to LlamaCloud
#  ↓
# LlamaParse processes the PDF
#  ├── OCR
#  ├── Text extraction
#  ├── Layout analysis
#  ├── Table detection
#  ├── Image detection
#  └── Reading-order detection
#  ↓
# Parsed results
#  ├── Markdown
#  ├── Plain text
#  ├── Page-wise records
#  ├── Tables
#  ├── Images
#  ├── Excel workbook
#  ├── JSON

# In[3]:


import os
import io
import re
import json
from pathlib import Path
from typing import Any


# In[2]:


import pandas as pd
import requests
from dotenv import load_dotenv


# In[1]:


from llama_cloud import LlamaCloud


# In[2]:


# ============================================================
# 1. File Paths
# ============================================================

PDF_PATH = Path(
    r"D:\complete_content_new\Full-Stack-GenAI-Bootcamp-1.0"
    r"\Class-30-Data-Parsing-for-RAG\data"
    r"\complex_rag_parsing_sample_with_sunny_image.pdf"
)

OUTPUT_DIR = Path(
    r"D:\complete_content_new\Full-Stack-GenAI-Bootcamp-1.0"
    r"\Class-30-Data-Parsing-for-RAG\data"
    r"\llamaparse_output"
)

IMAGE_DIR = OUTPUT_DIR / "extracted_images"
SCREENSHOT_DIR = IMAGE_DIR / "screenshots"
EMBEDDED_IMAGE_DIR = IMAGE_DIR / "embedded"
LAYOUT_IMAGE_DIR = IMAGE_DIR / "layout"
OTHER_IMAGE_DIR = IMAGE_DIR / "other"

TABLE_DIR = OUTPUT_DIR / "extracted_tables"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDED_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
LAYOUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
OTHER_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

if not PDF_PATH.exists():
    raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

print("PDF found:", PDF_PATH)
print("Output directory:", OUTPUT_DIR)


# In[3]:


load_dotenv()

LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

if not LLAMA_CLOUD_API_KEY:
    raise ValueError(
        "LLAMA_CLOUD_API_KEY was not found.\n"
        "Create a .env file and add:\n"
        "LLAMA_CLOUD_API_KEY=llx-your-api-key"
    )

client = LlamaCloud(
    api_key=LLAMA_CLOUD_API_KEY
)

print("LlamaCloud client initialized successfully.")


# In[4]:


# ============================================================
# 3. Helper Functions
# ============================================================

def safe_filename(filename: str) -> str:
    """
    Removes unsafe characters from a filename.
    """
    filename = Path(filename).name
    return re.sub(r'[<>:"/\\|?*]', "_", filename)


def object_to_dict(obj: Any) -> Any:
    """
    Safely converts an SDK/Pydantic object into a dictionary.
    """
    if obj is None:
        return None

    if isinstance(obj, dict):
        return {
            key: object_to_dict(value)
            for key, value in obj.items()
        }

    if isinstance(obj, (list, tuple)):
        return [object_to_dict(value) for value in obj]

    if hasattr(obj, "to_dict"):
        return object_to_dict(obj.to_dict())

    if hasattr(obj, "model_dump"):
        return object_to_dict(obj.model_dump(mode="json"))

    if hasattr(obj, "__dict__"):
        return {
            key: object_to_dict(value)
            for key, value in vars(obj).items()
            if not key.startswith("_")
        }

    return obj


def download_file(
    url: str,
    output_path: Path,
    timeout: int = 120
) -> bool:
    """
    Downloads a file from a presigned URL.
    """
    try:
        response = requests.get(
            url,
            timeout=timeout,
            stream=True
        )

        response.raise_for_status()

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with output_path.open("wb") as file:
            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):
                if chunk:
                    file.write(chunk)

        return True

    except Exception as error:
        print(
            f"Download failed for {output_path.name}: "
            f"{type(error).__name__}: {error}"
        )
        return False


def get_image_output_directory(category: str | None) -> Path:
    """
    Selects the correct folder according to image category.
    """
    category = (category or "other").lower()

    if category == "screenshot":
        return SCREENSHOT_DIR

    if category == "embedded":
        return EMBEDDED_IMAGE_DIR

    if category == "layout":
        return LAYOUT_IMAGE_DIR

    return OTHER_IMAGE_DIR


# In[5]:


print("\nUploading PDF to LlamaCloud...")

uploaded_file = client.files.create(
    file=PDF_PATH,
    purpose="parse"
)

print("Upload completed.")
print("Uploaded file ID:", uploaded_file.id)


# In[6]:


# ============================================================
# 5. Parse PDF
# ============================================================

print("\nParsing PDF with LlamaParse...")

result = client.parsing.parse(
    file_id=uploaded_file.id,

    # Suitable for complex documents containing
    # tables, images, scans and mixed layouts.
    tier="agentic",

    version="latest",

    output_options={
        # Preserve tables in Markdown output.
        "markdown": {
            "tables": {
                "output_tables_as_markdown": True
            }
        },

        # Generate an Excel workbook containing tables.
        "tables_as_spreadsheet": {
            "enable": True
        },

        # Save complete page screenshots,
        # original embedded images and detected layout regions.
        "images_to_save": [
            "screenshot",
            "embedded",
            "layout"
        ],
    },

    processing_options={
        # OCR language configuration.
        "ocr_parameters": {
            "languages": ["en"]
        }
    },

    # Controls which results are returned.
    expand=[
        "text",
        "text_full",
        "markdown",
        "markdown_full",
        "items",
        "metadata",
        "job_metadata",
        "images_content_metadata",
        "xlsx_content_metadata",
    ],

    verbose=True,
    timeout=600,
)

print("\nParsing completed.")
print("Job ID:", result.job.id)
print("Job status:", result.job.status)


# In[7]:


raw_result = object_to_dict(result)

raw_result_path = OUTPUT_DIR / "llamaparse_raw_result.json"

with raw_result_path.open(
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        raw_result,
        file,
        indent=2,
        ensure_ascii=False,
        default=str
    )

print("Raw JSON saved:", raw_result_path)


# In[8]:


job_info = {
    "job_id": result.job.id,
    "status": result.job.status,
    "file_id": uploaded_file.id,
    "source": str(PDF_PATH),
}

with (OUTPUT_DIR / "job_info.json").open(
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        job_info,
        file,
        indent=2,
        ensure_ascii=False
    )


# In[9]:


full_markdown = result.markdown_full or ""

if not full_markdown and result.markdown:
    full_markdown = "\n\n---\n\n".join(
        getattr(page, "markdown", "")
        for page in result.markdown.pages
        if getattr(page, "success", True)
    )

full_text = result.text_full or ""

if not full_text and result.text:
    full_text = "\n\n".join(
        page.text
        for page in result.text.pages
    )


markdown_path = OUTPUT_DIR / "rag_ready_document.md"
text_path = OUTPUT_DIR / "extracted_text.txt"

markdown_path.write_text(
    full_markdown,
    encoding="utf-8"
)

text_path.write_text(
    full_text,
    encoding="utf-8"
)

print("Markdown saved:", markdown_path)
print("Plain text saved:", text_path)


# In[10]:


# ============================================================
# 8. Create Page-wise Records
# ============================================================

text_by_page = {}

if result.text:
    for page in result.text.pages:
        text_by_page[page.page_number] = page.text


metadata_by_page = {}

if result.metadata:
    for page in result.metadata.pages:
        metadata_by_page[page.page_number] = object_to_dict(page)


page_records = []

if result.markdown:
    for page in result.markdown.pages:
        page_number = page.page_number
        success = getattr(page, "success", True)

        if success:
            markdown_content = getattr(
                page,
                "markdown",
                ""
            )

            error_message = None
        else:
            markdown_content = ""
            error_message = getattr(
                page,
                "error",
                "Unknown parsing error"
            )

        page_records.append({
            "page_number": page_number,
            "success": success,
            "text": text_by_page.get(page_number, ""),
            "markdown": markdown_content,
            "header": getattr(page, "header", None),
            "footer": getattr(page, "footer", None),
            "error": error_message,
            "metadata": metadata_by_page.get(
                page_number,
                {}
            ),
        })


page_records_path = OUTPUT_DIR / "page_records.json"

with page_records_path.open(
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        page_records,
        file,
        indent=2,
        ensure_ascii=False,
        default=str
    )

print("Page records saved:", page_records_path)
print("Total pages parsed:", len(page_records))


# In[11]:


table_records = []
table_counter = 0

if result.items:
    for page in result.items.pages:
        page_number = page.page_number

        if not getattr(page, "success", True):
            continue

        for item_index, item in enumerate(
            page.items,
            start=1
        ):
            if getattr(item, "type", None) != "table":
                continue

            table_counter += 1

            rows = getattr(item, "rows", []) or []
            csv_text = getattr(item, "csv", "") or ""
            markdown_text = getattr(item, "md", "") or ""
            html_text = getattr(item, "html", "") or ""

            table_record = {
                "table_number": table_counter,
                "page_number": page_number,
                "item_index": item_index,
                "rows": rows,
                "csv": csv_text,
                "markdown": markdown_text,
                "html": html_text,
            }

            table_records.append(table_record)

            base_name = (
                f"page_{page_number:03d}"
                f"_table_{table_counter:03d}"
            )

            csv_path = TABLE_DIR / f"{base_name}.csv"
            markdown_path = TABLE_DIR / f"{base_name}.md"
            html_path = TABLE_DIR / f"{base_name}.html"

            # Save the CSV representation.
            if csv_text:
                csv_path.write_text(
                    csv_text,
                    encoding="utf-8"
                )

            # Save Markdown representation.
            if markdown_text:
                markdown_path.write_text(
                    markdown_text,
                    encoding="utf-8"
                )

            # Save HTML representation.
            if html_text:
                html_path.write_text(
                    html_text,
                    encoding="utf-8"
                )

            # Convert into Pandas DataFrame for preview.
            try:
                if csv_text:
                    dataframe = pd.read_csv(
                        io.StringIO(csv_text)
                    )
                elif rows:
                    dataframe = pd.DataFrame(
                        rows[1:],
                        columns=rows[0]
                    )
                else:
                    dataframe = pd.DataFrame()

                print(
                    f"\nTable {table_counter} "
                    f"found on page {page_number}"
                )

                print(dataframe.head())

            except Exception as error:
                print(
                    f"Could not create DataFrame for "
                    f"table {table_counter}: {error}"
                )


table_records_path = OUTPUT_DIR / "table_records.json"

with table_records_path.open(
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        table_records,
        file,
        indent=2,
        ensure_ascii=False,
        default=str
    )

print("\nTotal tables found:", len(table_records))
print("Table records saved:", table_records_path)


# In[12]:


image_records = []

images_metadata = result.images_content_metadata

if images_metadata:
    print(
        "\nTotal images available:",
        images_metadata.total_count
    )

    for image in images_metadata.images:
        filename = safe_filename(image.filename)
        category = getattr(
            image,
            "category",
            None
        )

        destination_directory = (
            get_image_output_directory(category)
        )

        destination_path = (
            destination_directory / filename
        )

        download_success = False

        if image.presigned_url:
            download_success = download_file(
                image.presigned_url,
                destination_path
            )

        image_record = {
            "index": image.index,
            "filename": filename,
            "category": category,
            "content_type": getattr(
                image,
                "content_type",
                None
            ),
            "bbox": object_to_dict(
                getattr(image, "bbox", None)
            ),
            "local_path": (
                str(destination_path)
                if download_success
                else None
            ),
            "download_success": download_success,
        }

        image_records.append(image_record)

        print(
            f"{category or 'other'}: "
            f"{filename} → "
            f"{'saved' if download_success else 'failed'}"
        )


image_records_path = OUTPUT_DIR / "image_records.json"

with image_records_path.open(
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        image_records,
        file,
        indent=2,
        ensure_ascii=False,
        default=str
    )

print("Image records saved:", image_records_path)


# In[13]:


def find_metadata_download_url(
    metadata: dict | None,
    keyword: str
) -> str | None:
    """
    Finds a presigned URL from result content metadata.
    """
    if not metadata:
        return None

    for key, value in metadata.items():
        if keyword.lower() not in key.lower():
            continue

        presigned_url = getattr(
            value,
            "presigned_url",
            None
        )

        if presigned_url:
            return presigned_url

        if isinstance(value, dict):
            url = value.get("presigned_url")

            if url:
                return url

    return None


xlsx_url = find_metadata_download_url(
    result.result_content_metadata,
    "xlsx"
)

if xlsx_url:
    xlsx_path = OUTPUT_DIR / "all_extracted_tables.xlsx"

    if download_file(xlsx_url, xlsx_path):
        print("Excel tables downloaded:", xlsx_path)
else:
    print("No XLSX download URL was returned.")


# In[ ]:


# figure_records = []

# if result.items:
#     for page in result.items.pages:
#         if not getattr(page, "success", True):
#             continue

#         for item_index, item in enumerate(
#             page.items,
#             start=1
#         ):
#             if getattr(item, "type", None) != "image":
#                 continue

#             figure_records.append({
#                 "page_number": page.page_number,
#                 "item_index": item_index,
#                 "markdown": getattr(
#                     item,
#                     "md",
#                     ""
#                 ),
#                 "raw_item": object_to_dict(item),
#             })


# with (OUTPUT_DIR / "figure_records.json").open(
#     "w",
#     encoding="utf-8"
# ) as file:
#     json.dump(
#         figure_records,
#         file,
#         indent=2,
#         ensure_ascii=False,
#         default=str
#     )

# print("Total figure/image items:", len(figure_records))


# In[ ]:




