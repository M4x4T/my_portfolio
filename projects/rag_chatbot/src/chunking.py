import nltk


def chunk_text(sentences, chunk_size, overlap):
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(sentences), step):
        chunk = sentences[i : i + chunk_size]
        chunk = " ".join(chunk)
        chunks.append(chunk)
    return chunks


def process_document(parsed_data, chunk_size, overlap):
    all_chunks = []
    for page in parsed_data:
        sentences = nltk.sent_tokenize(page["text"])
        chunks = chunk_text(sentences, chunk_size, overlap)
        for i, chunk in enumerate(chunks):
            dict_chunk = {"page": page["page"], "chunk": chunk}
            all_chunks.append(dict_chunk)
    return all_chunks