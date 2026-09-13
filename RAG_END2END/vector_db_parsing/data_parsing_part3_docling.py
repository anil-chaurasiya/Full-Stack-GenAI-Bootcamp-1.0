#!/usr/bin/env python
# coding: utf-8

# In[1]:


print("all ok")


# Input PDF
#    ↓
# Docling configuration
#    ↓
# OCR + layout analysis + table detection
#    ↓
# DoclingDocument
#    ├── Text
#    ├── Headings
#    ├── Tables
#    ├── Pictures
#    ├── Page information
#    └── Reading order
#    ↓
# JSON / Markdown / Text / CSV / Images

# In[ ]:


#uv pip install docling langchain-docling tabulate


# In[ ]:


import json
from pathlib import Path


# /
# 
# \
#     
# \\
#     
# OUTOUT_DIR\ "file.json"

# In[ ]:


from docling.datamodel.base_models import InputFormat


# In[ ]:


InputFormat.PDF


# Complete PDF parsing pipeline configure
# for the Table extraction behaviour configure
# for Tesseract OCR(for fetcing the data from the images)

# In[ ]:


from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableStructureOptions,
    TesseractCliOcrOptions,
)


# Main Docling parser/converter

# In[ ]:


from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)


# In[ ]:


from docling_core.types.doc import (
    ImageRefMode,
    PictureItem,
    TableItem,
)


# In[3]:


PDF_PATH = Path(
    r"D:\complete_content_new\Full-Stack-GenAI-Bootcamp-1.0"
    r"\Class-30-Data-Parsing-for-RAG\data"
    r"\complex_rag_parsing_sample_with_sunny_image.pdf"
)

OUTPUT_DIR = Path(
    r"D:\complete_content_new\Full-Stack-GenAI-Bootcamp-1.0"
    r"\Class-30-Data-Parsing-for-RAG\data"
    r"\docling_parsed_output"
)

PAGE_IMAGE_DIR = OUTPUT_DIR / "page_images"
PICTURE_DIR = OUTPUT_DIR / "extracted_pictures"
TABLE_DIR = OUTPUT_DIR / "extracted_tables"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PAGE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
PICTURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

if not PDF_PATH.exists():
    raise FileNotFoundError(f"PDF not found: {PDF_PATH}")


# In[ ]:


pipeline_options = PdfPipelineOptions()


# In[ ]:


# Enable OCR
pipeline_options.do_ocr = True


# In[ ]:


pipeline_options.ocr_options = TesseractCliOcrOptions(
    tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    lang=["eng"],

    # False = selectable text + OCR where required
    # True = OCR complete pages
    force_full_page_ocr=False,
)


# In[ ]:


# Enable table structure extraction
pipeline_options.do_table_structure = True


# In[ ]:


pipeline_options.table_structure_options = TableStructureOptions(
    do_cell_matching=True,
)


# Name       Score
# Sunny      90
# Anil       95
# 
# Row 1, Column 1 → Name
# Row 1, Column 2 → Score
# Row 2, Column 1 → Sunny
# Row 2, Column 2 → 90
# 

# In[ ]:


# Preserve page and picture images
pipeline_options.images_scale = 2.0
pipeline_options.generate_page_images = True
pipeline_options.generate_picture_images = True


# In[ ]:


converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options
        )
    }
)


# In[ ]:


print("Parsing PDF with Docling...")

conversion_result = converter.convert(PDF_PATH)


# In[ ]:


docling_document = conversion_result.document

print("PDF parsed successfully.")
print("Total pages:", len(docling_document.pages))
print("Total tables:", len(docling_document.tables))
print("Total pictures:", len(docling_document.pictures))


# In[ ]:


json_output_path = OUTPUT_DIR / "docling_document.json"


# In[ ]:


json_output_path.write_text(
    json.dumps(
        docling_document.export_to_dict(),
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


# In[ ]:


print("JSON saved:", json_output_path)


# In[ ]:


markdown_output_path = OUTPUT_DIR / "rag_ready_document.md"


# In[ ]:


docling_document.save_as_markdown(
    markdown_output_path,
    image_mode=ImageRefMode.REFERENCED,
)

print("Markdown saved:", markdown_output_path)


# In[9]:


text_output_path = OUTPUT_DIR / "extracted_text.txt"

text_output_path.write_text(
    docling_document.export_to_markdown(strict_text=True),
    encoding="utf-8",
)

print("Plain text saved:", text_output_path)


# In[10]:


for page_number, page in docling_document.pages.items():

    if page.image is None:
        continue

    page_image_path = (
        PAGE_IMAGE_DIR
        / f"page_{page.page_no:03d}.png"
    )

    page.image.pil_image.save(
        page_image_path,
        format="PNG",
    )

print("Page images saved:", PAGE_IMAGE_DIR)


# In[11]:


for table_index, table in enumerate(
    docling_document.tables,
    start=1,
):
    table_df = table.export_to_dataframe(
        doc=docling_document
    )

    csv_path = TABLE_DIR / f"table_{table_index:03d}.csv"
    markdown_path = TABLE_DIR / f"table_{table_index:03d}.md"
    image_path = TABLE_DIR / f"table_{table_index:03d}.png"

    table_df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8",
    )

    markdown_path.write_text(
        table_df.to_markdown(index=False),
        encoding="utf-8",
    )

    table_image = table.get_image(docling_document)

    if table_image is not None:
        table_image.save(image_path, format="PNG")

print("Tables saved:", TABLE_DIR)


# In[12]:


picture_counter = 0

for element, _level in docling_document.iterate_items():

    if not isinstance(element, PictureItem):
        continue

    picture_counter += 1

    picture_path = (
        PICTURE_DIR
        / f"picture_{picture_counter:03d}.png"
    )

    picture_image = element.get_image(docling_document)

    if picture_image is not None:
        picture_image.save(
            picture_path,
            format="PNG",
        )

print("Pictures saved:", PICTURE_DIR)


# In[ ]:




