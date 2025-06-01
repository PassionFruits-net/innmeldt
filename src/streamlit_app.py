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
from pdf_viewer import render_pdf_viewer
import time

st.set_page_config(page_title="Chatbot", layout="wide")


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
    if "index_name" not in st.session_state:
        st.session_state["index_name"] = os.getenv("INDEX_NAME")

    # Initialize context if not exists
    if "context" not in st.session_state:
        st.session_state["context"] = []

    # Add custom CSS to maximize screen usage
    st.markdown("""
    <style>
    .main .block-container {
        max-width: 95%;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Create two columns - left for chat (smaller), right for PDF viewer (larger)
    # Using a 3:5 ratio and adding gap between columns
    col1, spacer, col2 = st.columns([3, 0.5, 5])

    # LEFT COLUMN - Chat Interface
    with col1:
        st.header("Chatbot")
        
        # Add some padding/styling to the chat interface
        with st.container():
            st.markdown("""
            <style>
            .chat-container {
                padding-right: 20px;
            }
            </style>
            """, unsafe_allow_html=True)
            
            query = st.text_input("Ask a question")
            if st.button("Ask") and query:
                start = time.time()
                answer = retrieve_model("thread", st.session_state["index_name"], query)
                elapsed = time.time() - start

                st.markdown(f"## Answer:\n{answer['content']}")
                st.markdown(f"Response time: {elapsed: .2f}s")

                # Update context for PDF viewer
                st.session_state["context"] = answer["context"]

                if os.getenv("DEBUGGING") == "True":
                    context_formatted = [f"{i}. '{line}'" for i, line in enumerate(answer["context"], 1)]
                    st.markdown("## Context:\n" + "\n".join(context_formatted))

    # SPACER COLUMN - Creates visual separation
    with spacer:
        st.markdown("")  # Empty spacer

    # RIGHT COLUMN - PDF Viewer (Larger)
    with col2:
        st.header("Relevant paragraphs")
        
        # Only show PDF viewer if we have context
        if st.session_state["context"]:
            pdf_path = "tjenesteloven.pdf"
            pdf_data = [("Kort om loven\nTjenestepensjonsloven"), ("§ 1-2. Definisjoner\nI loven betyr:")]
            render_pdf_viewer(pdf_path, st.session_state["context"])
        else:
            st.info("Ask a question to see relevant document sections with highlights.")


if __name__ == "__main__":
    if handle_authentication():
        main()