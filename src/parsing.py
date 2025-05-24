from langchain_core.documents import Document
from pymupdf4llm import to_markdown


def load_pdf(path: str) -> list[str]:
    pages = to_markdown(path, page_chunks=True)

    return [Document(page_content=p["text"], metadata={"source": path}) for p in pages]


def load_text(path: str) -> Document:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    return [Document(page_content=content, metadata={"source": path})]


LOADERS = {
    "txt": load_text,
    "md": load_text,
    "pdf": load_pdf
}
