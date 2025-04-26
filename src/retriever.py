# retriever.py

from typing import List
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
from config import settings, Settings


def semantic_search(query: str, settings: Settings, k: int = 3) -> List[str]:
    # 1. Embed the query
    openai_client = AzureOpenAI(
        api_key=settings.openai_api_key,
        azure_endpoint=settings.openai_endpoint,
        api_version=settings.openai_api_version,
    )

    embedding = openai_client.embeddings.create(
        model=settings.embedding_deployment,
        input=[query]
    ).data[0].embedding
    print(f"[Search] Embedding for query '{query}' generated successfully.")

    # 2. Set up Azure Search client
    search_client = SearchClient(
        endpoint=settings.search_endpoint,
        index_name=settings.index_name,
        credential=AzureKeyCredential(settings.search_key)
    )

    # 3. Prepare vectorized query (no need to set kind manually!)
    vector_query = VectorizedQuery(
        vector=embedding,
        k_nearest_neighbors=k,
        fields="contentVector"
    )

    # 4. Perform the search
    results = list(search_client.search(
        search_text=None,
        vector_queries=[vector_query],
        select="*",
    ))

    #print(f"[Search] Found {len(results)} results for query '{query}'")
    #print(f"[Search] Results: {results}")
    # return [r["content"] for r in results if "content" in r]
    return [
        f"File: {r['file_name']}, Score: {r['@search.score']}, Content: {r["content"]}"
        for r in results
    ]

