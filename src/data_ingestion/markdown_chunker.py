import re
from typing import List, Dict, Optional, Union
from llama_index.core.schema import Document


class MarkdownChunker:
    def __init__(self):
        pass

    def chunk(
        self,
        markdown_pages: List[str],
        source_file: str,
    ) -> List[Document]:
        """
        Main entry point: accepts page-wise markdown, returns semantically chunked Documents.
        """
        structured_chunks = self._parse_and_merge_by_headings(markdown_pages, source_file)
        return self._convert_chunks_to_documents(structured_chunks, source_file)

    def _parse_and_merge_by_headings(
        self,
        markdown_pages: List[str],
        source_file: str
    ) -> List[Dict]:
        """
        Parses per-page markdown and merges sections across pages based on markdown headings.
        Returns a list of semantic section chunks.
        """
        all_sections = []
        current_section = None

        for page_index, page_md in enumerate(markdown_pages):
            lines = page_md.splitlines()
            for line in lines:
                header_match = re.match(r"^(#{1,6})\s+(.*)", line.strip())

                if header_match:
                    # Start of a new section
                    if current_section:
                        all_sections.append(current_section)

                    hashes, title = header_match.groups()
                    current_section = {
                        "title": title.strip(),
                        "level": len(hashes),
                        "text_lines": [],
                        "start_page": page_index + 1,
                        "end_page": page_index + 1
                    }
                else:
                    if current_section:
                        current_section["text_lines"].append(line)
                        current_section["end_page"] = page_index + 1
                    else:
                        # No heading yet — create an "Unknown" section
                        current_section = {
                            "title": f"Untitled Section (Page {page_index + 1})",
                            "level": 0,
                            "text_lines": [line],
                            "start_page": page_index + 1,
                            "end_page": page_index + 1
                        }

        # Add final section
        if current_section:
            all_sections.append(current_section)

        return all_sections

    def _convert_chunks_to_documents(
        self,
        structured_chunks: List[Dict],
        source_file: str
    ) -> List[Document]:
        """
        Converts parsed chunks into LlamaIndex Documents with rich metadata.
        """
        documents = []

        for idx, chunk in enumerate(structured_chunks):
            text = "\n".join(chunk["text_lines"]).strip()
            if not text:
                continue

            doc = Document(
                text=text,
                metadata={
                    "source": source_file,
                    "section_title": chunk["title"],
                    "section_level": chunk["level"],
                    "start_page": chunk["start_page"],
                    "end_page": chunk["end_page"],
                    "chunk_index": idx
                }
            )
            documents.append(doc)

        return documents
