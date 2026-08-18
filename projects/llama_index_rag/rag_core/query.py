from rag_core.config import RERANKER_TOP_N, configure_llama_index, get_qdrant_client, COLLECTION_NAME, RERANKER_MODEL, RERANKER_TOP_N, SIMILARITY_TOP_K, SPARSE_TOP_K, ENABLE_HYBRID_SEARCH, SPARSE_MODEL, RELEVANCE_THRESHOLD
from rag_core.models import SourceInfo, QueryResult
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator
from llama_index.core.query_engine import BaseQueryEngine
from rag_core.ingestion import AccessLevel
from llama_index.core.postprocessor import SentenceTransformerRerank

def build_query_engine(user_role: AccessLevel) -> BaseQueryEngine:
    """Builds a QueryEngine that filters results based on the user's role."""
    configure_llama_index()
    client = get_qdrant_client()
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        enable_hybrid=ENABLE_HYBRID_SEARCH, 
        fastembed_sparse_model=SPARSE_MODEL,
    )
    index = VectorStoreIndex.from_vector_store(vector_store)
        # ключевое отличие от ingestion: мы НЕ создаём индекс заново,
        # а подключаемся к уже существующим данным в Qdrant

    if user_role == "internal":
        access_level = ["internal", "customer"]
    elif user_role == "customer":
        access_level = ["customer"]
    else:
        raise ValueError("Invalid user role")

    access_condition = MetadataFilter(
        key = "access_level",
        value = access_level,
        operator = FilterOperator.IN
        
    )
    filters = MetadataFilters(filters=[access_condition])
    reranker = SentenceTransformerRerank(
        model=RERANKER_MODEL,
        top_n=RERANKER_TOP_N
    )
    
    query_engine = index.as_query_engine(
        filters=filters,
        vector_store_query_mode="hybrid",
        similarity_top_k=SIMILARITY_TOP_K,
        sparse_top_k=SPARSE_TOP_K,
        node_postprocessors = [reranker],
    )
    

    
    return query_engine


def ask(question: str, user_role: AccessLevel) -> str:
    """Asks a question to the QueryEngine and returns the answer."""
    query_engine = build_query_engine(user_role)
    response = query_engine.query(question)
    return str(response)


def fallback_response() -> QueryResult:
    return QueryResult(
        answer = "I don't have enough information to answer this question confidently.",
        sources = [],
        escalate = True
    )
        


def ask_with_sources(question: str, user_role: AccessLevel) -> QueryResult:
    """Asks a question to the QueryEngine and returns the answer along with source information."""
    query_engine = build_query_engine(user_role)
    response = query_engine.query(question)
    if response.source_nodes == []:
        return fallback_response()
    best_score = max(node_with_score.score for node_with_score in response.source_nodes)
    if best_score < RELEVANCE_THRESHOLD:
            return fallback_response()
    sources = []
    for node_with_score in response.source_nodes:
        sources.append(SourceInfo(
            file_name=node_with_score.node.metadata.get("file_name", "Unknown"),
            score=node_with_score.score,
            snippet=node_with_score.node.get_text()[:150],
            access_level=node_with_score.node.metadata.get("access_level", "Unknown")
        ))
    return QueryResult(
        answer=str(response),
        sources=sources,
        escalate=False
    )

if __name__ == "__main__":
    # Example usage
    
    example_questions = ["Which tools for internal use are available?",
                        "What is the capital of France?",
                        "How do I reset my password?",
                    ]
    for question in example_questions:
        response = ask_with_sources(question, "internal")
        print(f"Question: {question}")
        for source in response.sources:
            print(f"Source: {source.file_name}, Score: {source.score}")
        print("\n")