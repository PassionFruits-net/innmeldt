# chunker.py
import re
from typing import List, Dict
from llama_index.core.schema import Document

class MarkdownChunker:
    def chunk(self, markdown_pages: List[str], source_file: str) -> List[Document]:
        sections = self._parse_merge(markdown_pages)
        docs: List[Document] = []
        for idx, sect in enumerate(sections):
            metadata = {
                "source": source_file,
                "title": sect["title"],
                "level": sect["level"],
                "start_page": sect["start"],
                "end_page": sect["end"],
                "chunk_index": idx
            }
            docs.append(Document(text=sect["text"], metadata=metadata))
        return docs

    def _parse_merge(self, pages: List[str]) -> List[Dict]:
        chunks: List[Dict] = []
        for i, pg in enumerate(pages):
            parts = re.split(r'(#+ .+)', pg)
            current = {"title": "", "text": "", "level": 0, "start": i, "end": i}
            for part in parts:
                if part.startswith("#"):
                    if current["text"].strip():
                        chunks.append(current)
                    level = part.count("#")
                    title = part.strip("# ")
                    current = {"title": title, "text": "", "level": level, "start": i, "end": i}
                else:
                    current["text"] += part
            if current["text"].strip():
                chunks.append(current)
        return chunks