from typing import List
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
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
from openai import AzureOpenAI
from llama_index.core.schema import Document
import hashlib
import uuid
import json
import time
import re
from datetime import datetime


class AzureVectorStore:
    """Handles embedding, indexing, and uploading documents to Azure Cognitive Search."""

    def __init__(self, settings, documents: List[Document], index_name: str = None):
        """Initialize the vector store with settings, documents, and optional index name."""
        self.settings = settings
        self.documents = documents
        self.index_name = index_name or self._generate_index_name()
        self._ensure_index()

    def upload(self):
        """Uploads embedded document chunks to the Azure Search index."""
        print(f"[Upload] Starting upload to index: {self.index_name}")
        embedding_client = AzureOpenAI(
            api_version=self.settings.openai_api_version,
            azure_endpoint=self.settings.openai_endpoint,
            api_key=self.settings.openai_api_key,
        )
        search_client = SearchClient(
            endpoint=self.settings.search_endpoint,
            index_name=self.index_name,
            credential=AzureKeyCredential(self.settings.search_key),
        )

        upload_candidates = []

        for doc in self.documents:
            file_name = doc.metadata.get("source", "unknown_file")
            chunk_index = str(doc.metadata.get("chunk_index", "0"))
            doc_id = self._deterministic_uuid(file_name, chunk_index)

            if self._document_exists(search_client, doc_id):
                print(f"[SKIP] Already indexed: {doc_id}")
                continue

            print(f"[EMBED] Chunk {chunk_index} from {file_name}")
            embedding = self._retry_embedding(embedding_client, doc.text, self.settings.embedding_deployment)
            doc.metadata["vector"] = embedding
            doc.metadata["id"] = doc_id
            upload_candidates.append(doc)

        if not upload_candidates:
            print("[Upload] Nothing new to upload.")
            return

        azure_docs = self._prepare_documents(upload_candidates)
        self._upload_batches(azure_docs)

        print(f"[Upload] Uploaded {len(azure_docs)} new chunks.")

    def _deterministic_uuid(self, file_name: str, chunk_index: str) -> str:
        """Generate a deterministic UUID from file name and chunk index."""
        return str(uuid.UUID(hashlib.md5(f"{file_name}-{chunk_index}".encode()).hexdigest()))

    def _prepare_documents(self, docs: List[Document]) -> List[dict]:
        """Convert internal document format to Azure Search-compatible structure."""
        return [{
            "id": doc.metadata["id"],
            "file_name": doc.metadata["source"],
            "chunk_index": doc.metadata["chunk_index"],
            "title": doc.metadata.get("title", ""),
            "content": doc.text,
            "contentVector": doc.metadata["vector"]
        } for doc in docs]

    def _upload_batches(self, documents: List[dict]):
        """Upload documents in batches to the Azure Search index."""
        search_client = SearchClient(
            endpoint=self.settings.search_endpoint,
            index_name=self.index_name,
            credential=AzureKeyCredential(self.settings.search_key),
        )
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            result = search_client.upload_documents(documents=batch)
            succeeded = sum(1 for r in result if r.succeeded)
            print(f"[Upload] Batch {i // batch_size + 1}: {succeeded} succeeded")

    def _document_exists(self, search_client, document_id: str) -> bool:
        """Check if a document already exists in the index."""
        try:
            search_client.get_document(key=document_id)
            return True
        except Exception:
            return False

    def _generate_index_name(self) -> str:
        """Generate a sanitized and timestamped index name from document sources."""
        base_names = [doc.metadata.get("source", "") for doc in self.documents]
        print(base_names)
        base = "_".join(re.sub(r'[^a-zA-Z0-9]', '_', n.lower()) for n in base_names)
        return base[:128]  # Azure Search index name limit

    def _index_exists(self) -> bool:
        """Check if the Azure Search index already exists."""
        client = SearchIndexClient(
            endpoint=self.settings.search_endpoint,
            credential=AzureKeyCredential(self.settings.search_key),
        )
        try:
            client.get_index(self.index_name)
            return True
        except:
            return False

    def _ensure_index(self):
        """Ensure the Azure Search index exists or create it."""
        if not self._index_exists():
            self._initialize_index()

    def _initialize_index(self):
        """Create a new Azure Search index using field configuration."""
        client = SearchIndexClient(
            endpoint=self.settings.search_endpoint,
            credential=AzureKeyCredential(self.settings.search_key),
        )
        fields = self._load_fields_config("fields_config.json")
        vector_search = VectorSearch(
            profiles=[
                VectorSearchProfile(
                    name="vector_search_profile",
                    algorithm_configuration_name="hnsw_algorithm_config",
                )
            ],
            algorithms=[HnswAlgorithmConfiguration(name="hnsw_algorithm_config")],
        )
        index = SearchIndex(name=self.index_name, fields=fields, vector_search=vector_search)
        client.create_index(index)
        print(f"[Index] Created new index: {self.index_name}")

    def _load_fields_config(self, config_path: str) -> List:
        """Load and parse the field definitions for the search index."""
        with open(config_path, "r") as f:
            config_data = json.load(f)

        fields = []
        for field in config_data["fields"]:
            field_type = getattr(
                SearchFieldDataType,
                field["type"].replace("Collection(", "").replace(")", ""),
                None,
            )
            if field_type is None:
                raise ValueError(f"Invalid field type: {field['type']} for field {field['name']}")

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

    def _retry_embedding(self, client, text: str, model: str, max_retries=5, initial_delay=1):
        """creating embedding and retrying max_retries times."""
        for attempt in range(max_retries):
            try:
                print(f"[Embedding] Attempt {attempt + 1}...")
                resp = client.embeddings.create(model=model, input=[text])
                return resp.data[0].embedding
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    delay = initial_delay * (2 ** attempt)
                    print(f"[Embedding] Rate limit hit. Retrying in {delay}s...")
                    time.sleep(delay)
                    continue
                print(f"[Embedding] Failed after {attempt + 1} attempts: {e}")
                raise
        return None
