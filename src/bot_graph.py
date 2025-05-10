from langgraph.graph import StateGraph, MessagesState, START, END
from langchain.prompts import PromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from retriever import semantic_search
from chat import chat

context_template = PromptTemplate.from_template(
    "context:\n{context}\nquestion:\n{content}"
)


class BotState(MessagesState):
    context: str # RAG context
    index_name: str # azure search index


def retrieval(state: BotState):
    messages = state["messages"]
    index_name = state["index_name"]
    last_message = messages[-1].content
    retrieved = semantic_search(last_message, index_name)

    return {"context": "\n\n".join(retrieved)}


def model(state: BotState):
    messages = state["messages"]
    retrieved = state["context"]

    messages[-1].content = context_template.format(context=retrieved, content=messages[-1].content)
    response = chat(messages)

    return {"messages": response}


workflow = StateGraph(BotState)
workflow.add_node(retrieval)
workflow.add_node(model)

workflow.add_edge(START, "retrieval")
workflow.add_edge("retrieval", "model")
workflow.add_edge("model", END)

llm_app = workflow.compile(checkpointer=MemorySaver())
