from src.embeddings import get_embedding, cosine_similarity


def retrieve_top_chunks(question, chunks, top_n):
    question_embedding = get_embedding(question)
    similarities = [
        (chunk, cosine_similarity(question_embedding, chunk["embedding"]))
        for chunk in chunks
    ]
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_chunks = [chunk for chunk, _ in similarities[:top_n]]
    return top_chunks