# chat.py

from openai import AzureOpenAI
from retriever import semantic_search
from config import settings

def chat(query, settings):
    context = semantic_search(query, settings)
    print(f"[Chat] Retrieved {len(context)} chunks from search index")

    if not context:
        return "Sorry, I couldn’t find any relevant information in the indexed documents."

    prompt = "\n\n".join(context) + f"\n\nUser: {query}\nAI (based ONLY on the above context):"
    client = AzureOpenAI(
        api_key=settings.openai_api_key,
        azure_endpoint=settings.openai_endpoint,
        api_version=settings.openai_api_version,
    )
    response = client.chat.completions.create(
        model=settings.chat_deployment,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer ONLY based on the provided context."},
            {"role": "user", "content": prompt},
        ]
    )
    return response.choices[0].message.content
