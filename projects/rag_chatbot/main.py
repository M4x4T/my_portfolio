from config import CHUNK_SIZE, OVERLAP, TOP_N
from src.pdf_parser import text_parser
from src.chunking import chunk_text, process_document
from src.retrieval import retrieve_top_chunks
from src.embeddings import embed_chunks
from src.generation import build_context, generate_answer


def main():
    path = input("Enter the path to the file: ")
    parsed_data = text_parser(path)
    chunks = process_document(parsed_data, CHUNK_SIZE, OVERLAP)
    chunks_with_embeddings = embed_chunks(chunks)

    print("Document processed and embedded successfully.")

    while True:
        question = input("Enter your question (or type 'exit' to quit): ")
        if question.lower() == "exit":
            break
        top_chunks = retrieve_top_chunks(question, chunks_with_embeddings, TOP_N)
        context = build_context(top_chunks)
        answer = generate_answer(context, question)
        print("Answer:", answer)
        source_pages = [str(chunk["page"]) for chunk in top_chunks]
        print("Source pages:", ", ".join(source_pages))


if __name__ == "__main__":
    main()