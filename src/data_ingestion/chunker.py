# chunker.py
import os
from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core import Document as LlamaDocument
from .parsing import LOADERS
import hashlib


class Chunker:
    def __init__(self, paths: List[str]):
        self.paths = paths
        self.files = None

    def chunk_data(self):
        raw_docs = self.__load_files()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", " "]
        )

        chunked_docs = splitter.split_documents(raw_docs)
        llama_docs = [LlamaDocument(text=self.__make_text(doc), metadata=doc.metadata) for doc in chunked_docs]
        return llama_docs


    def get_index_name(self):
        files = self.__load_files()
        sorted_chunks = sorted([f.page_content for f in files])
        combined_text = "CHUNK".join(sorted_chunks)

        hash_object = hashlib.sha256(combined_text.encode('utf-8'))
        return hash_object.hexdigest()

    def __load_files(self) -> List[Document]:
        if self.files:
            return self.files
        docs = []

        for path in self.paths:
            ext = path.split(".")[-1]
            loader = LOADERS.get(ext)

            if loader is None:
                raise ValueError(f"No loader defined for extension {ext}")

            loaded_docs = loader(path)
            docs.extend(loaded_docs)

        self.files = docs
        return docs

    def __make_text(self, doc: Document) -> str:
        filename = os.path.basename(doc.metadata["source"])
        return f"Filename: '{filename}'\n{doc.page_content}"


# import re
# from typing import List, Dict
# from langchain.schema import Document

# class MarkdownChunker:
#     def chunk(self, markdown_pages: List[str], source_file: str) -> List[Document]:
#         sections = self._parse_merge(markdown_pages)
#         docs: List[Document] = []
#         for idx, sect in enumerate(sections):
#             metadata = {
#                 "source": source_file,
#                 "title": sect["title"],
#                 "level": sect["level"],
#                 "start_page": sect["start"],
#                 "end_page": sect["end"],
#                 "chunk_index": idx
#             }
#             docs.append(Document(text=sect["text"], metadata=metadata))
#         return docs

#     def _parse_merge(self, pages: List[str]) -> List[Dict]:
#         chunks: List[Dict] = []
#         for i, pg in enumerate(pages):
#             parts = re.split(r'(#+ .+)', pg)
#             current = {"title": "", "text": "", "level": 0, "start": i, "end": i}
#             for part in parts:
#                 if part.startswith("#"):
#                     if current["text"].strip():
#                         chunks.append(current)
#                     level = part.count("#")
#                     title = part.strip("# ")
#                     current = {"title": title, "text": "", "level": level, "start": i, "end": i}
#                 else:
#                     current["text"] += part
#             if current["text"].strip():
#                 chunks.append(current)
#         return chunks