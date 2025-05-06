from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from bot_graph import llm_app
import asyncio

app = FastAPI()


@app.get("/run")
async def retrieve_model(thread_id: str, content: str):
    print(content)
    state = llm_app.invoke({"messages": HumanMessage(content)}, {"configurable": {"thread_id": thread_id}})
    return state["messages"][-1].content
