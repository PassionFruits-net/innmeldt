import tempfile
import streamlit as st
from config import settings
from data_ingestion.chunker import Chunker
from data_ingestion.indexer import AzureIndexer
import requests
import json
from auth import handle_authentication
import os
from langchain_core.messages import HumanMessage
from RAG.bot_graph import graph


def retrieve_model(thread_id: str, index_name: str, content: str):
    state = graph.invoke({"messages": HumanMessage(content)}, {"configurable": {"thread_id": thread_id, "index_name": index_name}})

    result_content = state["messages"][-1].content
    result_context = state["context"]

    return {
        "content": result_content,
        "context": result_context
    }



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
        st.session_state["index_name"] = os.getenv("INDEX_NAME")

    if "show_indexing" not in st.session_state:
        st.session_state["show_indexing"] = False


    if False: # st.button("Change documents"):
        st.session_state["show_indexing"] = not st.session_state["show_indexing"]

    if st.session_state["show_indexing"]:
        uploaded = st.file_uploader("Upload File(s)", type=["pdf", "md"], accept_multiple_files=True)

        if uploaded and st.button("Index"):
            names, docs = [], []

            temp_dir, temp_paths = save_uploads_to_temp_paths(uploaded)

            indexer = AzureIndexer(settings, temp_paths, chunker=Chunker)
            indexer.upload()

            index_name = indexer.index_name
            settings.index_name = index_name

            st.session_state["index_name"] = index_name
            st.success(f"Indexed into {index_name}")

    query = st.text_input("Ask a question")
    if st.button("Ask") and query:
        answer = retrieve_model("thread", st.session_state["index_name"], query)

        st.markdown(f"## Answer:\n{answer['content']}")
            
        if os.getenv("DEBUGGING"):
            context_formatted = [f"{i}. {line}" for i, line in enumerate(answer["context"], 1)]
            st.markdown("## Context:\n" + "\n".join(context_formatted))


if __name__ == "__main__":
    if handle_authentication():
        main()
