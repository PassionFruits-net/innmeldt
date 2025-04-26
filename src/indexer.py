# indexer.py
from typing import List
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
from llama_index.core.schema import Document
import hashlib
import uuid
import json
import time
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from config import settings
from azure.search.documents.indexes import SearchIndexClient
from datetime import datetime
import re


BATCH_SIZE = 100

def deterministic_uuid(file_name: str, chunk_index: str) -> str:
    return str(uuid.UUID(hashlib.md5(f"{file_name}-{chunk_index}".encode()).hexdigest()))

def embed_documents(client, docs: List[Document], model: str):
    for doc in docs:
        embedding = retry_embedding(client, doc.text, model)
        if embedding:
            doc.metadata["vector"] = embedding
        else:
            print(f"[Embedding] Skipped chunk {doc.metadata.get('chunk_index')} from {doc.metadata.get('source')}")
    return docs

def prepare_documents(docs: List[Document]) -> List[dict]:
    azure_docs = []
    for doc in docs:
        azure_docs.append({
            "id": doc.metadata["id"],
            "file_name": doc.metadata["source"],
            "chunk_index": doc.metadata["chunk_index"],
            "title": doc.metadata["title"],
            "content": doc.text,
            "contentVector": doc.metadata["vector"]
        })
    return azure_docs

def upload_batches(settings, documents: List[dict], index_name: str):
    search_client = SearchClient(
        endpoint=settings.search_endpoint,
        index_name=index_name,
        credential=AzureKeyCredential(settings.search_key),
    )
    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i:i + BATCH_SIZE]
        result = search_client.upload_documents(documents=batch)

        # Log success/failure
        succeeded = sum(1 for r in result if r.succeeded)
        failed = len(result) - succeeded

        print(f"[Upload] Batch {i // BATCH_SIZE + 1}: {succeeded} succeeded, {failed} failed")
        for r in result:
            if not r.succeeded:
                print(f"❌ Failed to upload doc ID {r.key}: {r.error_message}")
        
        print(f"[Upload] Uploaded batch {i // BATCH_SIZE + 1}: {len(result)} documents")

def upload_chunks(settings, docs: List[Document], index_name: str):
    print(f"[Upload] Starting upload of {len(docs)} chunks...")

    # Clients
    embedding_client = AzureOpenAI(
        api_version=settings.openai_api_version,
        azure_endpoint=settings.openai_endpoint,
        api_key=settings.openai_api_key,
    )
    search_client = SearchClient(
        endpoint=settings.search_endpoint,
        index_name=index_name,
        credential=AzureKeyCredential(settings.search_key)
    )

    upload_candidates = []

    for doc in docs:
        file_name = doc.metadata.get("source", "unknown_file")
        chunk_index = str(doc.metadata.get("chunk_index", "0"))
        doc_id = deterministic_uuid(file_name, chunk_index)

        if document_exists(search_client, doc_id):
            print(f"[SKIP] Already indexed: {doc_id}")
            continue

        print(f"[EMBED] Chunk {chunk_index} from {file_name}")
        embedding = retry_embedding(embedding_client, doc.text, settings.embedding_deployment)
        doc.metadata["vector"] = embedding
        doc.metadata["id"] = doc_id
        upload_candidates.append(doc)

    if not upload_candidates:
        print("[Upload] Nothing new to upload.")
        return

    azure_docs = prepare_documents(upload_candidates)
    upload_batches(settings, azure_docs, index_name)

    print(f"[Upload] Uploaded {len(azure_docs)} new chunks.")


def document_exists(_search_client, document_id: str) -> bool:
    """Check if a document already exists in the index."""
    try:
        result = _search_client.get_document(key=document_id)
        return True if result else False
    except Exception:
        return False

def generate_index_name(file_names: List[str]) -> str:
    base = "_".join(re.sub(r'[^a-zA-Z0-9]', '_', n.lower()) for n in file_names)
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    #return f"{base}_{ts}"
    return base[:128]  # Limit to 128 characters

def index_exists(settings, index_name: str) -> bool:
    client = SearchIndexClient(
        endpoint=settings.search_endpoint,
        credential=AzureKeyCredential(settings.search_key),
    )
    try:
        client.get_index(index_name)
        return True
    except:
        return False

def initialize_index(settings, index_name: str):
    creds = AzureKeyCredential(settings.search_key)

    index_client = SearchIndexClient(
        endpoint=settings.search_endpoint, credential=creds
    )
    create_index(settings, index_client, index_name, "fields_config.json")

def ensure_index(settings, index_name: str):
    if not index_exists(settings, index_name):
        initialize_index(settings, index_name)
    
    
def load_fields_config(fields_config_path):
    with open(fields_config_path, "r") as f:
        config_data = json.load(f)

    fields = []
    for field in config_data["fields"]:
        field_type = getattr(
            SearchFieldDataType,
            field["type"].replace("Collection(", "").replace(")", ""),
            None,
        )
        if field_type is None:
            raise ValueError(
                f"Invalid field type: {field['type']} for field {field['name']}"
            )

        if field["name"] == "contentVector":
            field_instance = SearchField(
                name="contentVector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=field["vector_search_dimensions"],
                vector_search_profile_name="vector_search_profile",
            )
        elif field["type"].startswith("Collection"):
            field_instance = SimpleField(
                name=field["name"],
                type=SearchFieldDataType.Collection(field_type),
                filterable=field.get("filterable", False),
                facetable=True,
            )
        elif field.get("searchable", False):
            field_instance = SearchableField(
                name=field["name"],
                type=field_type,
                analyzer_name=field.get("analyzer", None),
                facetable=True,
            )
        else:
            field_instance = SimpleField(
                name=field["name"],
                type=field_type,
                key=field.get("key", False),
                filterable=field.get("filterable", False),
                facetable=True,
            )

        fields.append(field_instance)
    return fields


def create_index(settings, _index_client, INDEX_NAME, field_config_path):
    """Create the search index if it doesn't exist."""
    print("Creating search index...")
    if index_exists(settings, INDEX_NAME):
        print(f"Index '{INDEX_NAME}' already exists. Skipping creation.")
        return

    fields = load_fields_config(field_config_path)
    print("FIELDS:")
    print(fields)
    print()

    vector_search = VectorSearch(
        profiles=[
            VectorSearchProfile(
                name="vector_search_profile",
                algorithm_configuration_name="hnsw_algorithm_config",
            )
        ],
        algorithms=[HnswAlgorithmConfiguration(name="hnsw_algorithm_config")],
    )

    index = SearchIndex(name=INDEX_NAME, fields=fields, vector_search=vector_search)

    print("Creating new index...")
    _index_client.create_index(index)
    print("Index created successfully")


def retry_embedding(client, text, model, max_retries=5, initial_delay=1):
    for attempt in range(max_retries):
        try:
            print(f"[Embedding] Attempt {attempt + 1}...")
            resp = client.embeddings.create(
                model=model,
                input=[text],
            )
            return resp.data[0].embedding
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)
                print(f"[Embedding] Rate limit hit. Retrying in {delay} seconds... (attempt {attempt + 1})")
                time.sleep(delay)
                continue
            print(f"[Embedding] Failed after {attempt + 1} attempts: {e}")
            raise
    return None


def build_documents(chunks: list[dict], embedding_function, file_name: str):
    """
    Generate documents conforming to Azure Cognitive Search index structure.

    :param chunks: List of content chunks (e.g., from markdown or PDF)
    :param embedding_function: Callable that returns a 3072-dimensional embedding
    :param file_name: Name of the source file
    :return: List of dicts ready for indexing
    """
    documents = []

    for i, chunk in enumerate(chunks):
        vector = embedding_function(chunk['content'])  # Ensure it returns 3072 floats
        assert len(vector) == 3072, "Embedding must be 3072-dim for contentVector"

        doc = {
            "id": str(uuid.uuid4()),
            "file_name": file_name,
            "contentVector": vector,
        }
        documents.append(doc)

    return documents
