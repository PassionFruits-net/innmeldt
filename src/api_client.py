from config import settings
from langchain_openai import AzureChatOpenAI
from openai import AzureOpenAI

openai_llm = AzureChatOpenAI(
    azure_deployment=settings.chat_deployment,
    api_version=settings.openai_api_version
)

embedding_client = AzureOpenAI(
    api_key=settings.openai_api_key,
    azure_endpoint=settings.openai_endpoint,
    api_version=settings.openai_api_version
)
