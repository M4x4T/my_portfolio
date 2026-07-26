# RAG Chatbot (from scratch, no frameworks)

A small project: a chatbot that answers questions about the content of a PDF document. All retrieval logic (chunking, embeddings, cosine similarity, search) is written from scratch in Python, without LangChain/LlamaIndex/smolagents — that's the whole point of this project.

Runs fully locally: answer generation and embeddings are computed through [Ollama](https://ollama.com), with no external APIs and no paid keys.

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/download) installed and running
- ~5 GB of free disk space (for the models)

---

## Installation

### 1. Clone the repository and enter the project folder

```bash
git clone <repository-link>
cd rag_chatbot
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
If PowerShell blocks the script (`execution policy` error):
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Windows (cmd):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

A successful activation shows `(.venv)` at the start of the terminal prompt.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK data (one-time)

```bash
python -c "import nltk; nltk.download('punkt_tab')"
```

### 5. Pull the Ollama models

```bash
ollama pull nomic-embed-text
ollama pull qwen2:7b
```

Check that the models were downloaded:
```bash
ollama list
```

### 6. Make sure Ollama is running

Ollama usually starts automatically as a background service after installation. You can check with:
```bash
ollama list
```
If the command runs without errors, the server is already up — there's no need to run `ollama serve` manually (and it won't work anyway if the port is already taken).

---

## Usage

From the project root (not from the `src/` folder):

```bash
python main.py
```

The program will then ask for the path to a PDF file:
```
Enter the path to the file: C:\Users\name\Documents\report.pdf
```
Enter the path **without quotes**, even if it was copied via "Copy as path" in Windows Explorer.

Once the document is processed (this can take anywhere from a few seconds to a few minutes depending on file size), you'll see:
```
Document processed and embedded successfully.
```

You can then ask questions in a loop:
```
Enter your question (or type 'exit' to quit): what is this document about?
Answer: ...
Source pages: 2, 5, 7
```

Type `exit` to quit.

---

## Project structure

```
rag_chatbot/
├── .env
├── .gitignore
├── requirements.txt
├── config.py
├── main.py
├── README.md
└── src/
    ├── __init__.py
    ├── pdf_parser.py     # PDF → text, page by page
    ├── chunking.py         # text → chunks with overlap
    ├── embeddings.py        # text → vector, cosine similarity
    ├── retrieval.py         # question → top-N relevant chunks
    └── generation.py        # context + question → LLM answer
```

---

## Configuration (`config.py`)

```python
CHUNK_SIZE = 5          # sentences per chunk
OVERLAP = 1              # overlapping sentences between adjacent chunks
TOP_N = 3                # how many chunks are placed into the context
EMBEDDING_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen2:7b"
```

You can swap in other models pulled through Ollama (e.g. `llama3.2:3b` instead of `qwen2:7b` — faster, but weaker answer quality).

---

## Known limitations

- **PDFs with a text layer only.** Scanned pages with no text (just images) are silently skipped — OCR is not implemented.
- **Processing speed.** Embeddings are computed sequentially, one request per chunk — large documents (100+ pages) can take minutes to process, especially without a GPU.
- **No persistence of embeddings between runs.** Every time `main.py` starts, the document is reprocessed from scratch, even if it's the same file as last time.
- **One document at a time.** No support for querying across multiple files at once.
- **`nomic-embed-text` is officially English-only.** On non-English text (a fully Russian or Turkish document, for example), retrieval quality may be less reliable than on English. For robust multilingual support, consider `nomic-embed-text-v2-moe` (supports ~100 languages) — this would require swapping the model in `config.py` and in the `ollama.embeddings()` call.
- **Brute-force search.** `retrieve_top_chunks` compares the question against every chunk with no indexing — at tens of thousands of chunks, a vector database (FAISS/Chroma) would be needed instead of a linear scan.
- **No input error handling.** An invalid file path or a non-running Ollama instance will produce a raw Python traceback rather than a friendly message.

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `LookupError: Resource punkt_tab not found` | NLTK data not downloaded | `python -c "import nltk; nltk.download('punkt_tab')"` |
| `OSError: [Errno 22] Invalid argument` when opening a PDF | Path entered with quotes | Enter the path without quotes |
| `ModuleNotFoundError` when importing from `src` | Missing `src/__init__.py`, or not running from the project root | Create an empty `src/__init__.py`; run `python main.py` from the project root, not from `src/` |
| `Error: listen tcp 127.0.0.1:11434: bind: ...` on `ollama serve` | Ollama is already running in the background | No need to run `ollama serve` manually — check with `ollama list` |
| Model's answer is clearly off-topic | Chunking too coarse, or a poor `TOP_N` value | Experiment with `CHUNK_SIZE`/`OVERLAP`/`TOP_N` in `config.py` |