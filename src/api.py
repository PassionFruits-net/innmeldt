from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from bot_graph import llm_app
import asyncio

app = FastAPI()

@app.get("/run")
def retrieve_model(thread_id: str, index_name: str, content: str):
    state = llm_app.invoke({"messages": HumanMessage(content), "index_name": index_name}, {"configurable": {"thread_id": thread_id}})

    result_content = state["messages"][-1].content
    result_context = state["relevant"]

    return {
        "content": result_content,
        "context": result_context
    }
