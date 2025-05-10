from config import settings


embedding_client = AzureOpenAI(
    api_key=settings.openai_api_key,
    azure_endpoint=settings.openai_endpoint,
    api_version=settings.openai_api_version
)


def search_client(index_name):
    return SearchClient(
        endpoint=settings.search_endpoint,
        index_name=index_name,
        credential=AzureKeyCredential(settings.search_key)
    )
