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

import PyPDF2
# or
from pypdf import PdfReader

def load_pdf(path: str) -> list[Document]:
    documents = []
    
    with open(path, 'rb') as file:
        pdf_reader = PdfReader(file)
        
        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text.strip() and len(text.strip()) > 50:  # Filter meaningful content
                documents.append(Document(
                    page_content=text,
                    metadata={
                        "source": path,
                        "start_page": page_num + 1,
                        "end_page": page_num + 1,
                        "chunk_index": page_num
                    }
                ))
    
    return documents


def _load_pdf(path: str) -> list[Document]:
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
