from langgraph.graph import StateGraph, MessagesState, START, END
from langchain.prompts import PromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from typing import List, TypedDict, Annotated
from pydantic import BaseModel, Field
from pydantic import ValidationError
from .retriever import semantic_search
from .api_client import openai_llm


answer_template = PromptTemplate.from_file("src/prompts/answer_prompt.txt")
optimize_prompt = PromptTemplate.from_file("src/prompts/optimize_prompt.txt")
rerank_prompt = PromptTemplate.from_file("src/prompts/rerank_prompt.txt")
reasoning_prompt = PromptTemplate.from_file("src/prompts/reasoning_prompt.txt")

DEFAULT_RETRIEVAL = 15
DEFAULT_RERANK = 5


class BotState(MessagesState):
    optimized: str
    retrieved: List[str]
    context: List[str]
    reasoning: str


class ConfigScema(TypedDict):
    index_name: str
    retrieval_n: int
    rerank_n : int


class ReasoningStep(BaseModel):
    reasoning: str = Field(..., description="Think through if the context answers the question and what the relevant chunks are.")
    relevant_chunks: List[int] = Field(..., description="The index of each relevant chunk.")


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

    class RankChunks(BaseModel):
        scores: Annotated[
            List[float], 
            Field(
                min_length=retrieval_n, 
                max_length=retrieval_n,
                description="The relevance score of each corresponding chunk from 0-1"
            )
        ]    

    rerank_llm = openai_llm.with_structured_output(RankChunks)

    chunks = "\n".join([f"{i}. {c['content']}" for i, c in enumerate(context_chunks, 1)])
    prompt = rerank_prompt.format(chunks=chunks, query=query)

    max_retries = 5
    for attempt in range(max_retries):
        try:
            scores = rerank_llm.invoke(prompt).scores
            break
        except ValidationError as e:
            if attempt == max_retries - 1:
                # After max retries, fallback to returning first rerank_n chunks directly
                return {"context": [chunk["content"] for chunk in context_chunks[:rerank_n]]}

    # Pair each chunk with its index, sort by score descending
    indexed_chunks = [(i, chunk) for i, chunk in enumerate(context_chunks)]
    indexed_chunks.sort(key=lambda x: scores[x[0]], reverse=True)

    # Return the top rerank_n chunks reversed for LLM consumption
    print("Reranking complete")
    return {"context": [chunk[1]["content"] for chunk in indexed_chunks[:rerank_n][::-1]]}


def structuring_node(state: BotState):
    messages = state["messages"]
    question = messages[-1]

    context = state["context"]
    chunks = "\n".join([f"{i}. {chunk}" for i, chunk in enumerate(context, 1)])
    prompt = [reasoning_prompt.format(chunks=chunks, question=question)]

    structuring_llm = openai_llm.with_structured_output(ReasoningStep)
    response = structuring_llm.invoke(messages[:-1] + prompt)

    new_context = [context[i-1] for i in response.relevant_chunks]

    print("Structuring complete")
    return {"reasoning": response.reasoning, "context": new_context}


def generate_response_node(state: BotState):
    """Generate a response using reasoning step."""
    messages = state["messages"]
    thoughts = state["reasoning"]
    optimized = state["optimized"]
    context = state["context"]

    chunks = "\n".join([f"{i}. {chunk}" for i, chunk in enumerate(context, 1)])

    last_message = [answer_template.format(context=chunks, question=optimized, reasoning=thoughts)]
    response = openai_llm.invoke(messages[:-1] + last_message)

    messages[-1].content = response.content

    print("Answer complete")
    return {"messages": messages[-1]}


graph_builder = StateGraph(BotState, ConfigScema)

graph_builder.add_node("optimize_prompt", optimize_prompt_node)
graph_builder.add_node("retrieval", retrieval_node)
graph_builder.add_node("reranking", reranking_node)
graph_builder.add_node("structuring", structuring_node)
graph_builder.add_node("generate_response", generate_response_node)

graph_builder.set_entry_point("optimize_prompt")
graph_builder.add_edge("optimize_prompt", "retrieval")
graph_builder.add_edge("retrieval", "reranking")
graph_builder.add_edge("reranking", "structuring")
graph_builder.add_edge("structuring", "generate_response")
graph_builder.add_edge("generate_response", END)

graph = graph_builder.compile()
