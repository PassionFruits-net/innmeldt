from langgraph.graph import StateGraph, MessagesState, START, END
from langchain.prompts import PromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from typing import List, TypedDict
from pydantic import BaseModel
from .retriever import semantic_search
from .api_client import openai_llm
from collections import Counter


answer_template = PromptTemplate.from_file("src/prompts/rag_answer_prompt.txt")
optimize_prompt = PromptTemplate.from_file("src/prompts/optimize_prompt.txt")


class BotState(MessagesState):
    optimized: str
    context: List[str]


class ConfigScema(TypedDict):
    index_name: str
    retrieval_n: int


def check_if_retrieval_needed(state: BotState) -> str:
    """Determine whether retrieval is necessary based on the input state."""
    # TODO: analyze if retrieval needed with structured output
    if True:
        return "optimize_prompt"
    else:
        return "generate_directly"


def generate_directly_node(state: BotState):
    """Generate a direct response using the LLM without retrieval."""
    messages = state["messages"]
    response = openai_llm.invoke(messages)
    return {"messages": response}


def optimize_prompt_node(state: BotState):
    """Refine or optimize the user prompt before retrieval."""
    messages = state["messages"]
    optimized = openai_llm.invoke(messages[:-1] + [optimize_prompt.format(content=messages[-1].content)])
    return {"optimized": optimized}


def retrieval_node(state: BotState, config: RunnableConfig):
    """Perform semantic retrieval based on the optimized prompt."""
    # TODO: implement hybrid search
    retrieval_n = config["configurable"].get("retrieval_n", 20)
    index_name = config["configurable"]["index_name"]

    last_message = state["optimized"].content
    retrieved = semantic_search(last_message, index_name, k=retrieval_n)
    context = [s[0] for s in Counter(retrieved).most_common(3)][::-1] # reversing to get most common section further down, better for llm

    return {"context": context}


def structuring_node(state: BotState):
    pass

def generate_with_rag_node(state: BotState):
    """Generate a response using retrieved and context (RAG)."""
    messages = state["messages"]
    combined_context = "\n".join(state["context"])

    last_message = [answer_template.format(context=combined_context, content=messages[-1].content)]
    response = openai_llm.invoke(messages[:-1] + last_message)

    return {"messages": response}


graph_builder = StateGraph(BotState, ConfigScema)

# graph_builder.add_node("check_retrieval", check_if_retrieval_needed)
graph_builder.add_node("generate_directly", generate_directly_node)
graph_builder.add_node("optimize_prompt", optimize_prompt_node)
graph_builder.add_node("retrieval", retrieval_node)
graph_builder.add_node("structuring", structuring_node)
graph_builder.add_node("generate_with_rag", generate_with_rag_node)

graph_builder.set_entry_point("optimize_prompt")
graph_builder.add_edge("optimize_prompt", "retrieval")
graph_builder.add_edge("retrieval", "structuring")
graph_builder.add_edge("structuring", "generate_with_rag")
graph_builder.add_edge("generate_with_rag", END)

graph = graph_builder.compile(checkpointer=MemorySaver())
