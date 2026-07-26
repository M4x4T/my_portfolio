import math
import ollama


def cosine_similarity(vec_a, vec_b):
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def get_embedding(text):
    result = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return result["embedding"]


def embed_chunks(chunks):
    return [{**chunk, "embedding": get_embedding(chunk["chunk"])} for chunk in chunks]