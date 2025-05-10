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
    relevant: List[str]


relevance_prompt = PromptTemplate.from_template(
    "Given the following context and question, which lines of the context are relevant to answering the question?\n\n"
    "Context:\n{context}\n\nQuestion:\n{question}\n\n"
    "You will add DIRECT sentences from the context that is relevant to the question. If there is multiple sentences after each other then that is one line. If the context is not relevant, then don't add anything to the list."
)


class RelevanceResponse(BaseModel):
    relevant_lines: List[str]


def retrieval(state: BotState):
    """Retrieve context based on the last message's content."""
    
    messages = state["messages"]
    index_name = state["index_name"]
    last_message = messages[-1].content
    
    retrieved = semantic_search(last_message, index_name)
    
    return {"context": retrieved}


def highlight_relevance(state: BotState):
    """highlight direct quotes that are relevant."""
    
    context = state["context"]
    last_message = state["messages"][-1].content
    context_str = "\n".join(context)
    
    formatted_prompt = relevance_prompt.format(context=context_str, question=last_message)
    
    structured_llm = openai_llm.with_structured_output(RelevanceResponse)
    response: RelevanceResponse = structured_llm.invoke(formatted_prompt)

    for line in response.relevant_lines:
        if line not in context_str:
            print(line)
        else:
            print("CORRECT.")

    return {"relevant": response.relevant_lines}


def is_question_relevant(state: BotState):
    """Check if there is relevant context."""
    
    if len(state["relevant"]):
        return "model"
    
    return "question_irrelevant"


def question_irrelevant(state: BotState):
    return {"messages": "There is no context to answer that question."}


def model(state: BotState):
    """Generate a response based on the context and the question."""
    
    messages = state["messages"]
    retrieved = "\n".join(state["relevant"])
     
    messages[-1].content = context_template.format(context=retrieved, content=messages[-1].content)
    
    response = openai_llm.invoke(system_prompt + messages)
    
    return {"messages": response}


workflow = StateGraph(BotState)
workflow.add_node(retrieval)
workflow.add_node(highlight_relevance)
workflow.add_node(is_question_relevant)
workflow.add_node(question_irrelevant)
workflow.add_node(model)


workflow.add_edge(START, "retrieval")
workflow.add_edge("retrieval", "highlight_relevance")
workflow.add_conditional_edges("highlight_relevance", is_question_relevant)

workflow.add_edge("question_irrelevant", END)
workflow.add_edge("model", END)

llm_app = workflow.compile(checkpointer=MemorySaver())
