from langgraph.graph import StateGraph, MessagesState, START, END
from langchain.prompts import PromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from typing import List, TypedDict
from pydantic import BaseModel, Field
import asyncio
from .retriever import semantic_search
from .api_client import openai_llm


answer_template = PromptTemplate.from_file("src/prompts/rag_answer_prompt.txt")
optimize_prompt = PromptTemplate.from_file("src/prompts/optimize_prompt.txt")
rerank_prompt = PromptTemplate.from_file("src/prompts/rerank_prompt.txt")


class BotState(MessagesState):
    optimized: str
    retrieved: List[str]
    context: List[str]


class ConfigScema(TypedDict):
    index_name: str
    retrieval_n: int
    rerank_n : int


class RankChunk(BaseModel):
    score:      float = Field(..., description="Give a final score from 0-1 describing how relevant the chunk is to the user query.")


rerank_llm: RankChunk = openai_llm.with_structured_output(RankChunk)


def optimize_prompt_node(state: BotState):
    """Refine or optimize the user prompt before retrieval."""
    messages = state["messages"]
    optimized = openai_llm.invoke(messages[:-1] + [optimize_prompt.format(content=messages[-1].content)])
    print("Prompt optimalization complete")
    return {"optimized": optimized}


def retrieval_node(state: BotState, config: RunnableConfig):
    """Perform semantic retrieval based on the optimized prompt."""
    # TODO: implement hybrid search
    retrieval_n = config["configurable"].get("retrieval_n", 15)
    index_name = config["configurable"]["index_name"]

    last_message = state["optimized"].content
    retrieved = semantic_search(last_message, index_name, k=retrieval_n)

    print("Retrieval complete")
    return {"retrieved": retrieved}


def reranking_node(state: BotState, config: RunnableConfig):
    rerank_n = config["configurable"].get("rerank_n", 5)
    query = state["optimized"]
    context_chunks = state["retrieved"]
    return {"context": [c["content"] for c in context_chunks[:5]]}

    async def score_all_chunks():
        async def score_chunk(chunk, title, i):
            prompt = rerank_prompt.format(title=title, chunk=chunk, query=query)
            print("Running: %d" % i)
            result = await rerank_llm.ainvoke(prompt)
            print("Complete: %d" % i)
            return {"chunk": chunk, "score": result.score}

        tasks = [score_chunk(chunk["content"], chunk["title"], i) for i, chunk in enumerate(context_chunks)]
        return await asyncio.gather(*tasks)

    scored_chunks = asyncio.run(score_all_chunks())

    top_chunks = sorted(scored_chunks, key=lambda x: x["score"], reverse=True)[:rerank_n]

    print("Rerank complete")
    return {"context": [item["chunk"] for item in top_chunks]}


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
