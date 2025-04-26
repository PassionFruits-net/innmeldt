# parsing.py
from pymupdf4llm import to_markdown

def parse_pdf_to_markdown(pdf_path: str) -> list[str]:
    pages = to_markdown(pdf_path, page_chunks=True)
    return [p["text"] for p in pages]