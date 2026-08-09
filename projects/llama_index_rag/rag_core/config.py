from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from qdrant_client import QdrantClient

OLLAMA_BASE_URL = "http://localhost:11434"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "support_documents"
SIMILARITY_TOP_K = 5
SPARSE_TOP_K = 5
RERANKER_TOP_N = 3
ENABLE_HYBRID_SEARCH = True


LLM_MODEL = "qwen2:7b"
EMBED_MODEL = "nomic-embed-text"
SPARSE_MODEL = "Qdrant/bm25"
RERANKER_MODEL = "cross-encoder/ms-marco-TinyBERT-L-2-v2"



def configure_llama_index() -> None:
    """Sets global LlamaIndex Settings so every Index/QueryEngine
    created afterwards uses the same LLM and embedding model
    without passing them explicitly each time."""
    Settings.llm = Ollama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, request_timeout=300.0)
    Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL, base_url=OLLAMA_BASE_URL)


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)