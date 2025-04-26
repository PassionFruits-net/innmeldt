# main.py
import sys
from config import settings
from parsing import parse_pdf_to_markdown
from chunker import MarkdownChunker
from indexer import upload_chunks
from chat import chat
    
def main():
    pdf_path = sys.argv[1]
    pages = parse_pdf_to_markdown(pdf_path)
    docs = MarkdownChunker().chunk(pages, str(pdf_path))
    upload_chunks(settings, docs, index_name=settings.index_name)
    while True:
        q = input("Ask something: ")
        if q.lower() in ("exit", "quit"):
            break
        print(chat(q))

if __name__ == "__main__":
    main()