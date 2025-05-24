import tempfile
import streamlit as st
from config import settings
from chunker import Chunker
from indexer import AzureVectorStore
import requests
import json
import os


def chat(query):
    response = requests.get(
        "http://localhost:8000/run",
        params={"thread_id": "testing", "content": query, "index_name": st.session_state["index_name"]}
    )
    return json.loads(response.text)


def save_uploads_to_temp_paths(uploaded_files):
    """
    Save uploaded Streamlit files to temp files and return list of file paths.
    The caller must handle the lifecycle of the temp files.
    """
    temp_dir = tempfile.TemporaryDirectory()
    temp_paths = []

    for uploaded_file in uploaded_files:
        temp_path = os.path.join(temp_dir.name, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        temp_paths.append(temp_path)

    # Return both the temp dir object and paths, so temp_dir isn't deleted immediately
    return temp_dir, temp_paths



def main():
    st.title("Chat me up!")

    if "index_name" not in st.session_state:
        st.session_state["index_name"] = ""

    if "show_indexing" not in st.session_state:
        st.session_state["show_indexing"] = False


    if st.button("Change documents"):
        st.session_state["show_indexing"] = not st.session_state["show_indexing"]


    if st.session_state["show_indexing"]:
        uploaded = st.file_uploader("Upload File(s)", type=["pdf", "md"], accept_multiple_files=True)

        if uploaded and st.button("Index"):
            names, docs = [], []

            temp_dir, temp_paths = save_uploads_to_temp_paths(uploaded)
            docs = Chunker(temp_paths).chunk_data()

            print(docs)

            # vec = AzureVectorStore(settings, docs, st.session_state["index_name"])
            # vec.upload()

            # index_name = vec.index_name
            # settings.index_name = index_name

            # st.session_state["index_name"] = index_name
            # st.success(f"Indexed into {index_name}")

    query = st.text_input("Ask a question")
    if st.button("Ask") and query:
        answer = chat(query)
        st.markdown(f"## Answer:\n{answer['content']}")
        
        context_formatted = [f"{i}. {line}" for i, line in enumerate(answer["context"], 1)]
        st.markdown("## Context:\n" + "\n".join(context_formatted))


if __name__ == "__main__":
    main()
