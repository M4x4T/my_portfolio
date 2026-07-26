import ollama


def build_context(chunks):
    return "\n\n".join(chunk["chunk"] for chunk in chunks)


def generate_answer(context, question):
    prompt = f"""
        Контекст:
        {context}

        Вопрос: {question}

        Ответь на вопрос, используя только информацию из контекста выше.
    """
    response = ollama.chat(model="qwen2:7b", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]