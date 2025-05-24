from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from RAG.bot_graph import graph
import asyncio

app = FastAPI()

@app.get("/run")
def retrieve_model(thread_id: str, index_name: str, content: str):
    state = graph.invoke({"messages": HumanMessage(content)}, {"configurable": {"thread_id": thread_id, "index_name": index_name}})

    result_content = state["messages"][-1].content
    # result_vimcontext = state["relevant"]

    return {
        "content": result_content,
        "context": [] # result_context
    }
