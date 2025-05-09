from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from bot_graph import llm_app
import asyncio

app = FastAPI()


@app.get("/run")
def retrieve_model(thread_id: str, index_name: str, content: str):
    print(content, index_name)
    state = llm_app.invoke({"messages": HumanMessage(content), "index_name": index_name}, {"configurable": {"thread_id": thread_id}})
    return state["messages"][-1].content
