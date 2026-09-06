"""
Convert retriever_advance.ipynb from OpenAI to Google Gemini (Free)
-------------------------------------------------------------------
Creates: retriever_advance_gemini.ipynb
- Replaces OpenAI imports with Google Gemini imports
- Uses GOOGLE_API_KEY (free) instead of OPENAI_API_KEY (paid)
- Uses GoogleGenerativeAIEmbeddings instead of OpenAIEmbeddings
- Uses ChatGoogleGenerativeAI instead of OpenAI / ChatOpenAI
"""
import json
from pathlib import Path

DIR_PATH = Path(__file__).parent
NB_PATH = DIR_PATH / "retriever_advance.ipynb"
OUTPUT_PATH = DIR_PATH / "retriever_advance_gemini.ipynb"

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") != "code":
        continue

    new_source = []
    for line in cell.get("source", []):
        # 1. Fix imports & dotenv
        line = line.replace("load_dotenv()", "load_dotenv(override=True)")
        line = line.replace(
            "from langchain_openai import OpenAI, OpenAIEmbeddings",
            "from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings",
        )
        line = line.replace(
            "from langchain_openai import ChatOpenAI",
            "from langchain_google_genai import ChatGoogleGenerativeAI",
        )

        # 2. Fix API Key configuration
        line = line.replace("OPENAI_API_KEY", "GOOGLE_API_KEY")
        line = line.replace("Enter your OpenAI API key:", "Enter your Google API key:")
        line = line.replace(
            "OpenAI API key configured successfully.",
            "Google API key configured successfully.",
        )
        line = line.replace("# 2. OPENAI API KEY", "# 2. GOOGLE API KEY")

        # 3. Fix Embeddings
        line = line.replace(
            "# 11. CREATE EMBEDDING MODEL",
            "# 11. CREATE EMBEDDING MODEL (Google Gemini - FREE)",
        )
        line = line.replace('model="text-embedding-3-small"', 'model="models/text-embedding-004"')
        line = line.replace("OpenAIEmbeddings(", "GoogleGenerativeAIEmbeddings(")

        # 4. Fix LLMs (HyDE & Multi-Query)
        line = line.replace(
            "# 17. CREATE HYDE EMBEDDINGS",
            "# 17. CREATE HYDE EMBEDDINGS (using Gemini)",
        )
        line = line.replace('model="gpt-3.5-turbo-instruct"', 'model="gemini-2.0-flash"')
        line = line.replace('model="gpt-4.1-mini"', 'model="gemini-2.0-flash"')
        line = line.replace("llm=OpenAI(", "llm=ChatGoogleGenerativeAI(")
        line = line.replace("llm = ChatOpenAI(", "llm = ChatGoogleGenerativeAI(")

        new_source.append(line)

    cell["source"] = new_source
    cell["outputs"] = []
    cell["execution_count"] = None

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Successfully generated {OUTPUT_PATH.name}")
