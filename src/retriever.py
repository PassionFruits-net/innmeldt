# retriever.py

from typing import List
from azure.search.documents.models import VectorizedQuery
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
from api_client import embedding_client
from config import settings


def semantic_search(query: str, index_name: str, k: int = 3) -> List[str]:
    search_client = SearchClient(
        endpoint=settings.search_endpoint,
        index_name=index_name,
        credential=AzureKeyCredential(settings.search_key)
    )

    # 1. Embed the query
    embedding = embedding_client.embeddings.create(
        model=settings.embedding_deployment,
        input=[query]
    ).data[0].embedding

    # 2. Prepare vectorized query (no need to set kind manually!)
    vector_query = VectorizedQuery(
        vector=embedding,
        k_nearest_neighbors=k,
        fields="contentVector"
    )

    # 3. Perform the search
    results = list(search_client.search(
        search_text=None,
        vector_queries=[vector_query],
        select="*",
    ))


    return [
        f"File: {r['file_name']}, Score: {r['@search.score']}, Content: {r["content"]}"
        for r in results
    ]
