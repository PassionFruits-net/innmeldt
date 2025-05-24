from langchain_core.documents import Document
from pymupdf4llm import to_markdown


# METADATA = {
#     "title": ""
# }


# def metadata_decorator(func):
#     def add_metadata(path: str):
#         documents = func(path)

#         for doc in documents:
#             for key, value in METADATA.items():
#                 doc.metadata[key] = doc.metadata.get(key, value)

#         return documents

#     return add


def load_pdf(path: str) -> list[str]:
    pages = to_markdown(path, page_chunks=True)

    return [Document(page_content=p["text"], metadata={
        "source": path,
        "start_page": 1,
        "end_page": p["metadata"]["page_count"],
        "chunk_index": p["metadata"]["page"]
        
        }) for p in pages]


def load_text(path: str) -> Document:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    return [Document(page_content=content, metadata={"source": path})]


LOADERS = {
    "txt": load_text,
    "md": load_text,
    "pdf": load_pdf
}
