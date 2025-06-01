from langgraph.graph import StateGraph, MessagesState, START, END
from langchain.prompts import PromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from typing import List, TypedDict, Annotated
from pydantic import BaseModel, Field
import asyncio
from .retriever import semantic_search
from .api_client import openai_llm


answer_template = PromptTemplate.from_file("src/prompts/rag_answer_prompt.txt")
optimize_prompt = PromptTemplate.from_file("src/prompts/optimize_prompt.txt")
rerank_prompt = PromptTemplate.from_file("src/prompts/rerank_prompt.txt")

DEFAULT_RETRIEVAL = 15
DEFAULT_RERANK = 5


class BotState(MessagesState):
    optimized: str
    retrieved: List[str]
    context: List[str]


class ConfigScema(TypedDict):
    index_name: str
    retrieval_n: int
    rerank_n : int


def optimize_prompt_node(state: BotState):
    """Refine or optimize the user prompt before retrieval."""
    messages = state["messages"]
    optimized = openai_llm.invoke(messages[:-1] + [optimize_prompt.format(content=messages[-1].content)])
    print("Prompt optimalization complete")
    return {"optimized": optimized}


def retrieval_node(state: BotState, config: RunnableConfig):
    """Perform semantic retrieval based on the optimized prompt."""
    # TODO: implement hybrid search
    retrieval_n = config["configurable"].get("retrieval_n", DEFAULT_RETRIEVAL)
    index_name = config["configurable"]["index_name"]

    last_message = state["optimized"].content
    retrieved = semantic_search(last_message, index_name, k=retrieval_n)

    print("Retrieval complete")
    return {"retrieved": retrieved}


def reranking_node(state: BotState, config: RunnableConfig):
    """Reranking retrieved chunks with llm, only keeping the most important chunks."""
    retrieval_n = config["configurable"].get("retrieval_n", DEFAULT_RETRIEVAL)
    rerank_n = config["configurable"].get("rerank_n", DEFAULT_RERANK)

    query = state["optimized"]
    context_chunks = state["retrieved"]


    class RankParagraphs(BaseModel):
        scores: Annotated[
            List[float], 
            Field(
                min_length=retrieval_n, 
                max_length=retrieval_n,
                description="The relevance score of each corresponding chunk from 0-1"
            )
        ]    
    rerank_llm = openai_llm.with_structured_output(RankParagraphs)

    chunks = [f"{i}. {c["content"]}\n" for i, c in enumerate(context_chunks, 1)]
    prompt = rerank_prompt.format(chunks=chunks, query=query)
    scores = rerank_llm.invoke(prompt).scores

    # giving each chunk index that is used to get score, reverse makes it descending
    context_chunks = [(i, chunk) for i, chunk in enumerate(context_chunks)]
    context_chunks.sort(key=lambda x: scores[x[0]], reverse=True)

    # reversing makes most relevant chunk nearest output, better for llm
    return {"context": [chunk[1]["content"] for chunk in context_chunks[:rerank_n][::-1]]}


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

graph_builder.add_node("optimize_prompt", optimize_prompt_node)
graph_builder.add_node("retrieval", retrieval_node)
graph_builder.add_node("reranking", reranking_node)
graph_builder.add_node("structuring", structuring_node)
graph_builder.add_node("generate_with_rag", generate_with_rag_node)

graph_builder.set_entry_point("optimize_prompt")
graph_builder.add_edge("optimize_prompt", "retrieval")
graph_builder.add_edge("retrieval", "reranking")
graph_builder.add_edge("reranking", "generate_with_rag")
graph_builder.add_edge("generate_with_rag", END)

graph = graph_builder.compile(checkpointer=MemorySaver())
