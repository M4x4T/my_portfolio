from typing import Literal

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore

from rag_core.config import configure_llama_index, get_qdrant_client, COLLECTION_NAME, ENABLE_HYBRID_SEARCH, SPARSE_MODEL

AccessLevel = Literal["internal", "customer"]


def load_documents_with_access_level(folder_path: str, access_level: AccessLevel):
    """Loads all documents from a folder and tags each one with
    an access_level metadata field BEFORE chunking, so every
    resulting Node inherits it automatically."""
    documents = SimpleDirectoryReader(folder_path).load_data()
    for doc in documents:
        doc.metadata["access_level"] = access_level
    return documents


def ingest_documents(data_dir: str = "data") -> dict:
    """Loads internal/ and customer/ documents, chunks them,
    embeds them, and upserts everything into a single Qdrant
    collection distinguished by the access_level metadata field."""
    configure_llama_index()

    internal_docs = load_documents_with_access_level(f"{data_dir}/internal", "internal")
    customer_docs = load_documents_with_access_level(f"{data_dir}/customer", "customer")
    all_documents = internal_docs + customer_docs

    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)

    client = get_qdrant_client()
    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME,
    enable_hybrid=ENABLE_HYBRID_SEARCH, 
    fastembed_sparse_model=SPARSE_MODEL,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        all_documents,
        storage_context=storage_context,
        transformations=[splitter],
    )

    return {
        "internal_docs": len(internal_docs),
        "customer_docs": len(customer_docs),
        "collection": COLLECTION_NAME,
    }


if __name__ == "__main__":
    result = ingest_documents()
    print(f"Indexed {result['internal_docs']} internal docs and "
            f"{result['customer_docs']} customer docs into '{result['collection']}'")