# chat.py

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from api_client import openai_client

system_prompt = SystemMessage("You are a helpful assistant. Answer ONLY based on the provided context.")



def chat(history):
    response = openai_client.invoke([system_prompt, *history])
    return response
