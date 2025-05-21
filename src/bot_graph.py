from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import SystemMessage
from langchain.prompts import PromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from retriever import semantic_search
from api_client import openai_llm
from pydantic import BaseModel
from typing import List

context_template = PromptTemplate.from_template(
    "context:\n{context}\nquestion:\n{content}"
)

system_prompt = [SystemMessage("You are a helpful assistant. Answer ONLY based on the provided context.")]


class BotState(MessagesState):
    context: str
    index_name: str
    relevant: List[str] = []
    optimized: str


optimize_prompt = PromptTemplate.from_file("prompts/optimize_prompt.txt")


def prompt_optimalization(state: BotState):
    msgs = state["messages"]
    optimized = openai_llm.invoke(msgs[:-1] + [optimize_prompt.format(content=msgs[-1].content)])

    return {"messages": optimized}


def retrieval(state: BotState):
    """Retrieve context based on the last message's content."""
    
    messages = state["messages"]
    index_name = state["index_name"]
    last_message = messages[-1].content
    
    retrieved = semantic_search(last_message, index_name)
    
    return {"context": retrieved}


def model(state: BotState):
    """Generate a response based on the context and the question."""
    
    messages = state["messages"]
    retrieved = "\n".join(state["context"])
     
    messages[-1].content = context_template.format(context=retrieved, content=messages[-1].content)
    
    response = openai_llm.invoke(system_prompt + messages)
    
    return {"messages": response}


workflow = StateGraph(BotState)

workflow.add_node(retrieval)
workflow.add_node(model)
workflow.add_node(prompt_optimalization)

workflow.add_edge(START, "prompt_optimalization")
workflow.add_edge("prompt_optimalization", "retrieval")
# workflow.add_edge("retrieval", "highlight_relevance")
# # workflow.add_conditional_edges("highlight_relevance", is_question_relevant)
# 
# workflow.add_edge("question_irrelevant", END)
# workflow.add_edge("model", END)

llm_app = workflow.compile(checkpointer=MemorySaver())
