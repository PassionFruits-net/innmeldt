from typing import List, Dict, Any
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
from llama_index.core import Document
import hashlib
import uuid
import json
import time
from datetime import datetime


def create_embedding(text: str, settings) -> List[float]:
    """Create embedding for a given text using Azure OpenAI."""
    client = AzureOpenAI(
        api_version=settings.openai_api_version,
        azure_endpoint=settings.openai_endpoint,
        api_key=settings.openai_api_key,
    )
    
    max_retries = 5
    initial_delay = 1
    
    for attempt in range(max_retries):
        try:
            print(f"[Embedding] Attempt {attempt + 1}...")
            resp = client.embeddings.create(
                model=settings.embedding_deployment, 
                input=[text]
            )
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


class AzureIndexer:
    """Handles indexing and uploading documents to Azure Cognitive Search."""

    def __init__(self, settings):
        """Initialize the indexer with settings only."""
        self.settings = settings
        self.documents = []

    def add_document(self, document: Document, embedding: List[float]):
        """Add a LlamaIndex Document with its embedding to the upload queue.
        
        Args:
            document: A LlamaIndex Document object with `.text` and `.metadata`.
            embedding: Pre-computed embedding vector.
        """
        metadata = document.metadata or {}
        required_fields = ['source', 'chunk_index']
        for field in required_fields:
            if field not in metadata:
                raise ValueError(f"Metadata must include '{field}' field")
        
        file_name = metadata['source']
        chunk_index = str(metadata['chunk_index'])
        doc_id = self._deterministic_uuid(file_name, chunk_index)
        
        new_doc = {
            'id': doc_id,
            'text': document.text,
            'embedding': embedding,
            'metadata': metadata
        }
        
        self.documents.append(new_doc)
        print(f"[Queue] Added document: {file_name} (chunk {chunk_index})")

    def upload(self, index_name: str = None):
        """Calculate index name and upload all queued documents."""
        if not self.documents:
            print("[Upload] No documents to upload.")
            return
        
        # Calculate index name from document sources if not provided
        if index_name is None:
            index_name = self._calculate_index_name()
        
        print(f"[Upload] Starting upload to index: {index_name}")
        
        # Check if index exists, create if not
        if not self._index_exists(index_name):
            self._initialize_index(index_name)
        
        # Filter out documents that already exist
        search_client = SearchClient(
            endpoint=self.settings.search_endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(self.settings.search_key),
        )
        
        upload_candidates = []
        for doc in self.documents:
            if self._document_exists(search_client, doc['id']):
                print(f"[SKIP] Already indexed: {doc['id']}")
                continue
            upload_candidates.append(doc)
        
        if not upload_candidates:
            print("[Upload] All documents already exist in index.")
            return
        
        # Prepare and upload documents
        azure_docs = self._prepare_documents(upload_candidates)
        print(azure_docs)
        self._upload_batches(search_client, azure_docs)
        
        print(f"[Upload] Successfully uploaded {len(azure_docs)} documents to {index_name}")
        
        # Clear the queue after successful upload
        self.documents.clear()

    def _calculate_index_name(self) -> str:
        """Calculate index name based on document sources."""
        if not self.documents:
            return "default-index"
        
        # Get unique source files
        sources = set(doc['metadata']['source'] for doc in self.documents)
        
        # Create a hash of all sources for consistent naming
        sources_str = ''.join(sorted(sources))
        hash_suffix = hashlib.md5(sources_str.encode()).hexdigest()[:8]
        
        return f"documents-{hash_suffix}"

    def _deterministic_uuid(self, file_name: str, chunk_index: str) -> str:
        """Generate a deterministic UUID from file name and chunk index."""
        return str(uuid.UUID(hashlib.md5(f"{file_name}-{chunk_index}".encode()).hexdigest()))

    def _prepare_documents(self, docs: List[Dict]) -> List[dict]:
        """Convert internal document format to Azure Search-compatible structure."""
        azure_docs = []
        for doc in docs:
            azure_doc = {
                "id": doc['id'],
                "file_name": doc['metadata']['source'],
                "chunk_index": doc['metadata']['chunk_index'],
                "title": doc['metadata'].get('section_title', ''),
                "content": doc['text'],
                "contentVector": doc['embedding']
            }
            
            azure_docs.append(azure_doc)
        
        return azure_docs

    def _upload_batches(self, search_client: SearchClient, documents: List[dict]):
        """Upload documents in batches to the Azure Search index."""
        batch_size = 100
        total_succeeded = 0
        total_failed = 0
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            try:
                result = search_client.upload_documents(documents=batch)
                succeeded = sum(r.succeeded for r in result)
                failed = len(batch) - succeeded
                total_succeeded += succeeded
                total_failed += failed
                
                print(f"[Upload] Batch {i // batch_size + 1}: {succeeded} succeeded, {failed} failed")
                
                # Log failed documents
                for r in result:
                    if not r.succeeded:
                        print(f"[ERROR] Failed to upload {r.key}: {r.error_message}")
                        
            except Exception as e:
                print(f"[ERROR] Batch {i // batch_size + 1} failed completely: {e}")
                total_failed += len(batch)
        
        print(f"[Upload] Final totals: {total_succeeded} succeeded, {total_failed} failed")

    def _document_exists(self, search_client: SearchClient, document_id: str) -> bool:
        """Check if a document already exists in the index."""
        try:
            search_client.get_document(key=document_id)
            return True
        except Exception:
            return False

    def _index_exists(self, index_name: str) -> bool:
        """Check if the Azure Search index already exists."""
        client = SearchIndexClient(
            endpoint=self.settings.search_endpoint,
            credential=AzureKeyCredential(self.settings.search_key),
        )
        try:
            client.get_index(index_name)
            return True
        except:
            return False

    def _initialize_index(self, index_name: str):
        """Create a new Azure Search index using field configuration."""
        client = SearchIndexClient(
            endpoint=self.settings.search_endpoint,
            credential=AzureKeyCredential(self.settings.search_key),
        )
        vector_search = VectorSearch(
            profiles=[
                VectorSearchProfile(
                    name="vector_search_profile",
                    algorithm_configuration_name="hnsw_algorithm_config",
                )
            ],
            algorithms=[HnswAlgorithmConfiguration(name="hnsw_algorithm_config")],
        )
        client.create_index(index)
        print(f"[Index] Created new index: {index_name}")

    def _load_fields_config(self, config_path: str) -> List:
        """Load and parse the field definitions for the search index."""
        with open(config_path, "r") as f:
            config_data = json.load(f)

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

    def get_queue_status(self):
        """Get information about queued documents."""
        return {
            'total_documents': len(self.documents),
            'sources': list(set(doc['metadata']['source'] for doc in self.documents)),
            'calculated_index_name': self._calculate_index_name() if self.documents else None
        }