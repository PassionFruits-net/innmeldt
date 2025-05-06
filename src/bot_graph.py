from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import os
load_dotenv()


context_template = PromptTemplate.from_template(
    "context:\n{context}\nquestion:\n{content}"
)

llm = ChatOpenAI(model="gpt-4o-mini")


class BotState(MessagesState):
    context: str # RAG context


async def retrieval(state: BotState):
    messages = state["messages"]
    last_message = messages[-1]

    # Example retrieval logic
    retrieved = "Nothing to retrieve"
    
    return {"context": retrieved}


async def model(state: BotState):
    messages = state["messages"]
    retrieved = state["context"]

    # messages[-1].content = context_template.format(context=retrieved, content=messages[-1].content)

    response = await llm.ainvoke(messages)
    return {"messages": response.content}


workflow = StateGraph(MessagesState)
workflow.add_node(retrieval)
workflow.add_node(model)

workflow.add_edge(START, "retrieval")
workflow.add_edge("retrieval", "model")
workflow.add_edge("model", END)

llm_app = workflow.compile(checkpointer=MemorySaver())
