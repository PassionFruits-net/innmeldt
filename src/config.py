# # config.py
# from pydantic_settings import BaseSettings
# from pydantic import SecretStr

# class Settings(BaseSettings):
# # Azure Cognitive Search
# search_endpoint: str
# search_key: SecretStr
# index_name: str

# # Azure OpenAI
# openai_endpoint: str
# openai_api_key: SecretStr
# openai_api_version: str
# embedding_deployment: str
# chat_deployment: str

# class Config:
# env_file = ".env"
# env_file_encoding = "utf-8"

# settings = Settings()

# # src/config.py

import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import SecretStr
from langchain_openai import AzureChatOpenAI



# 1) explicitly point at your .env, one level up from src/
env_path = Path(__file__).parent.parent / ".env"
if not env_path.exists():
 raise RuntimeError(f".env file not found at {env_path}")
load_dotenv(dotenv_path=env_path)

def _missing(varname: str):
 raise RuntimeError(f"Required env var {varname} is not set or is empty")

class Settings:
 # Azure Cognitive Search
 search_endpoint: str = os.getenv("SEARCH_ENDPOINT") or _missing("SEARCH_ENDPOINT")
 search_key: SecretStr = os.getenv("SEARCH_KEY") or _missing("SEARCH_KEY")
 index_name: str = os.getenv("INDEX_NAME") or _missing("INDEX_NAME")

 # Azure OpenAI
 openai_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT") or _missing("AZURE_OPENAI_ENDPOINT")
 openai_api_key: SecretStr = os.getenv("AZURE_OPENAI_API_KEY") or _missing("AZURE_OPENAI_API_KEY")
 openai_api_version: str = os.getenv("AZURE_OPENAI_API_VERSION") or _missing("AZURE_OPENAI_API_VERSION")
 embedding_deployment: str = os.getenv("EMBEDDING_DEPLOYMENT") or _missing("EMBEDDING_DEPLOYMENT")
 chat_deployment: str = os.getenv("CHAT_DEPLOYMENT") or _missing("CHAT_DEPLOYMENT")

def _missing(varname: str):
 raise RuntimeError(f"Required env var {varname} is not set or is empty")

settings = Settings()

openai_client = AzureChatOpenAI(
    azure_deployment=settings.chat_deployment,
    api_version=settings.openai_api_version
)
