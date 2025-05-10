# streamlit_app.py

import tempfile
import streamlit as st
from config import settings
from parsing import parse_pdf_to_markdown
from chunker import MarkdownChunker
from indexer import AzureVectorStore
# from indexer import upload_chunks
# from indexer import ensure_index, generate_index_name
import requests


def chat(query):
    response = requests.get("http://localhost:8000/run", {"thread_id": "testing", "content": query, "index_name": settings.index_name})
    return response.text


def main():
    st.title("Chat me up!")

    if "index_name" not in st.session_state:
        st.session_state["index_name"] = ""

    uploaded = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
    if uploaded and st.button("Index"):
        names, docs = [], []

        for f in uploaded:
            names.append(f.name)

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(f.getbuffer())
                path = tmp.name

            pages = parse_pdf_to_markdown(path)
            docs.extend(MarkdownChunker().chunk(pages, source_file=f.name))

        vec = AzureVectorStore(settings, docs)
        vec.upload()

        index_name = vec.index_name
        settings.index_name = index_name

        # print(settings.index_name)
        # index_name = generate_index_name(names)
        # ensure_index(settings, index_name)
        # upload_chunks(settings, docs, index_name)
        # settings.index_name = index_name
        # print(settings.index_name)

        st.session_state["index_name"] = index_name
        st.success(f"Indexed into {index_name}")

    query = st.text_input("Ask a question")
    if query and st.button("Ask"):
        answer = chat(query)
        st.write(answer)

if __name__ == "__main__":
    main()
