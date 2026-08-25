# Role-Based RAG Chatbot

A locally-hosted, Retrieval-Augmented Generation chatbot that separates **internal** (employee-facing) and **customer** (public-facing) knowledge through metadata-based access control, with hybrid search, cross-encoder reranking, source citations, and a confidence-based fallback that refuses to answer rather than hallucinate.

Built as a portfolio piece modeled on a real freelance job spec: *"Custom RAG Chatbot for Internal Team and Customer Support"* (Upwork).

---

## Table of Contents

- [Architecture](#architecture)
- [Models Used](#models-used)
- [Why These Choices](#why-these-choices)
- [Features](#features)
- [Project Structure](#project-structure)
- [API Endpoints](#api-endpoints)
- [Setup & Running Locally](#setup--running-locally)
- [Testing](#testing)
- [Known Limitations](#known-limitations)
- [Possible Extensions](#possible-extensions)

---

## Architecture

The system is split into two independent pipelines, following the standard RAG pattern of separating **indexing** (done once, or on document updates) from **querying** (done per user request):

### Indexing pipeline (`rag_core/ingestion.py`)

```
Documents (.txt, extensible to PDF/DOCX)
        │
        ▼
SimpleDirectoryReader   →  loads files, tags each Document with
                            metadata["access_level"] = "internal" | "customer"
        │
        ▼
SentenceSplitter        →  chunks documents into Nodes (chunk_size=512,
                            overlap=50), metadata is inherited by every Node
        │
        ▼
Embedding (dense)       →  nomic-embed-text via Ollama
Embedding (sparse)      →  Qdrant/bm25 via fastembed
        │
        ▼
Qdrant                  →  single collection, dual dense+sparse vectors per Node
```

### Query pipeline (`rag_core/query.py`)

```
User question + user_role ("internal" | "customer")
        │
        ▼
MetadataFilter (IN)     →  internal → ["internal", "customer"]
                            customer → ["customer"]
                            (applied server-side in Qdrant, not post-filtered
                             in Python — customer requests never receive
                             internal payloads over the wire)
        │
        ▼
Hybrid retrieval        →  dense + sparse candidates fused by Qdrant
        │
        ▼
Cross-encoder reranking →  cross-encoder/ms-marco-TinyBERT-L-2-v2
        │
        ▼
Confidence check        →  if no nodes, or best reranker score < threshold:
                            → return fallback (escalate=True), LLM is
                              never called
        │
        ▼
LLM response synthesis  →  qwen2:7b via Ollama, grounded in retrieved Nodes
        │
        ▼
QueryResult (Pydantic)  →  { answer, sources[], escalate }
```

A FastAPI layer (`backend/`) wraps this core with `POST /query`, `GET /health`, and a `POST /ingest` admin stub, using the same Pydantic models as the response contract — so the OpenAPI/Swagger schema is generated directly from the RAG core's own types.

---

## Models Used

| Purpose | Model | Served via |
|---|---|---|
| Text generation | `qwen2:7b` | Ollama (local) |
| Dense embeddings | `nomic-embed-text` | Ollama (local) |
| Sparse embeddings (hybrid search) | `Qdrant/bm25` | fastembed (local) |
| Reranking | `cross-encoder/ms-marco-TinyBERT-L-2-v2` | sentence-transformers (local, CPU) |
| Vector database | Qdrant | Docker (local) |

Everything runs **locally with no external API calls and no API keys** — a deliberate choice explained below.

---

## Why These Choices

**Ollama instead of a hosted API (OpenAI/Claude).** For local development and portfolio purposes, a fully local stack means zero marginal cost per query and no dependency on external service availability. The pipeline itself is provider-agnostic: `Settings.llm` and `Settings.embed_model` are the only two points where a model is bound, so swapping to a hosted API is a config change, not an architecture change.

**Qdrant instead of Chroma.** The two central requirements of this project — role-based document separation and hybrid (dense + sparse) search — are exactly the two things Qdrant supports natively and Chroma does not handle as well out of the box. Chroma's metadata filtering is simpler and it has no built-in sparse/dense fusion. Since the project's core teaching goal *was* those two features, Qdrant was the better fit even though it requires a Docker container instead of Chroma's embedded, zero-infra mode.

**Single Qdrant collection with an `access_level` metadata field**, rather than two separate collections. This follows Qdrant's own multi-tenancy guidance: a single collection with a payload-based tenant filter scales better and is simpler to maintain than N collections, and it makes an "internal user sees both internal + customer" access pattern trivial (a two-value `IN` filter) instead of requiring cross-collection queries.

**Hybrid search (dense + sparse), not dense-only.** Dense embeddings capture semantic meaning well but are weak on exact terms (product codes, specific policy names). Sparse (BM25-style) vectors catch exact keyword matches that dense embeddings can blur. Qdrant fuses both at query time.

**Reranking as a second stage, not a replacement for retrieval.** Retrieval (even hybrid) is a fast, coarse first pass across the whole collection; a cross-encoder reranker is far more accurate but too slow to run against every document, so it's applied only to the already-narrowed candidate set. `top_n=3` after `similarity_top_k=5` / `sparse_top_k=5` — the reranker narrows, it doesn't need to receive more candidates than retrieval produced.

**`ms-marco-TinyBERT-L-2-v2` specifically** (not a larger cross-encoder) — the whole stack runs on CPU with no GPU acceleration, verified via Ollama server logs during development. The smallest well-supported cross-encoder was chosen deliberately to keep reranking latency bounded on constrained hardware, matching the "explain why the selected models ... are suitable for our expected usage and budget" requirement in the source job spec.

**Confidence-based fallback via reranker score, not LLM self-reporting.** An LLM prompted to "say you don't know" is not a hard guarantee — it's a statistical tendency. The fallback here is a program-level `if` check on the reranker's own relevance score before the LLM is ever invoked. If retrieval found nothing, or the best result scores below `RELEVANCE_THRESHOLD`, a fixed, hardcoded response is returned and the LLM call is skipped entirely — removing the possibility of the model improvising an answer for an out-of-scope question.

**Threshold value (`-8`) is empirically calibrated, not guessed.** The reranker returns raw logits (this specific cross-encoder is not sigmoid-calibrated to a 0–1 range — confirmed by observing negative scores in practice), so a fixed "0.5 confidence" heuristic would be meaningless here. The threshold was set by running three calibration probes: a clearly relevant question (best score ≈ ‑2.9), a completely unrelated question (best score ≈ ‑11.6, all three candidates converging to nearly the same low score), and a second relevant question (best score ≈ +7.2). `-8` sits inside the wide gap between "relevant" and "irrelevant" observed on this small probe set — see [Known Limitations](#known-limitations) for the caveats on this.

**Pydantic models (`SourceInfo`, `QueryResult`) instead of raw dicts** as the public contract of `rag_core`. FastAPI validates request/response bodies through Pydantic, so defining the contract this way lets the API layer use these types directly as `response_model`, gets free request/response validation, and gets an auto-generated OpenAPI schema — with zero manual serialization code.

---

## Features

- **Role-based access control** — `internal` sees internal + customer documents; `customer` sees customer-only documents, enforced via a server-side Qdrant metadata filter (not a Python-side post-filter, so restricted payloads never leave the database for a customer request).
- **Hybrid retrieval** — dense (`nomic-embed-text`) + sparse (`Qdrant/bm25`) search fused by Qdrant.
- **Cross-encoder reranking** for higher-precision ordering of retrieved chunks.
- **Source citations** — every answer returns `file_name`, `relevance score`, a text `snippet`, and the `access_level` of each contributing chunk.
- **Confidence-based fallback / escalation** — refuses to answer (and flags `escalate: true`) when retrieval confidence is below threshold, instead of letting the LLM guess.
- **Automated tests** covering role isolation (customer never receives internal-tagged sources; internal can), invalid-role error handling, and fallback behavior — run via `pytest`.
- **FastAPI backend** exposing the pipeline as a REST API with auto-generated Swagger docs.

---

## Project Structure

```
llama_index_rag/
├── rag_core/
│   ├── __init__.py
│   ├── config.py          # Settings, model names, thresholds — single source of truth
│   ├── ingestion.py        # Document loading, chunking, embedding, indexing
│   ├── query.py            # Role-filtered retrieval, reranking, fallback, generation
│   └── models.py           # Pydantic contract: SourceInfo, QueryResult
├── backend/
│   ├── main.py              # FastAPI app: /query, /health, /ingest
│   ├── requirements.txt
│   └── tests/
│       └── test_main.py     # API-layer tests (mocked rag_core — no live Qdrant/Ollama needed)
├── data/
│   ├── internal/            # Sample internal-only documents
│   └── customer/            # Sample customer-facing documents
├── tests/
│   └── test_query.py       # pytest suite: role isolation, fallback, error handling
├── docker-compose.yml       # Qdrant service definition
├── requirements.txt
└── README.md
```

---

## API Endpoints

Once the FastAPI backend is running (`http://localhost:8000`), interactive docs are available at `/docs`.

### `POST /query`

```json
// Request
{
  "question": "Which tools for internal use are available?",
  "user_role": "internal"
}
```

```json
// Response (QueryResult)
{
  "answer": "The available tools for internal use include ...",
  "sources": [
    {
      "file_name": "onboarding_procedure_internal.txt",
      "score": -2.9068,
      "snippet": "Internal Tools Access Required: ...",
      "access_level": "internal"
    }
  ],
  "escalate": false
}
```

An invalid `user_role` returns HTTP 400. Note that `QueryRequest.user_role` is typed as a plain `str` at the API layer rather than a `Literal` — role validation is deliberately left to `ask_with_sources` itself, so the validation rule lives in exactly one place (`rag_core`) instead of being duplicated in the API schema.

### `GET /health`

Simple liveness check — `{"status": "ok"}`.

### `POST /ingest`

Admin-only stub that triggers `rag_core.ingestion.ingest_documents()`. No authentication is implemented yet (see [Known Limitations](#known-limitations)).

---

## Setup & Running Locally

### Prerequisites

- Python 3.13 (see note below on Python version)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (for Qdrant)
- [Ollama](https://ollama.com/) installed and running locally

> **Python version note:** As of this project's development, `llama-index-vector-stores-qdrant` and its dependencies did not yet publish wheels for Python 3.14. Use **Python 3.13** to avoid dependency resolution failures.

### 1. Clone and set up the environment

```bash
git clone https://github.com/M4x4T/my_portfolio.git
cd my_portfolio/projects/llama_index_rag

python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Pull the required Ollama models

```bash
ollama pull qwen2:7b
ollama pull nomic-embed-text
```

### 3. Start Qdrant

```bash
docker compose up -d
```

Verify at `http://localhost:6333/dashboard`.

### 4. Ingest the sample documents

```bash
python -m rag_core.ingestion
```

This loads the documents in `data/internal/` and `data/customer/`, chunks and embeds them (dense + sparse), and writes them into a Qdrant collection.

### 5. Query the pipeline directly (no API layer)

```bash
python -m rag_core.query
```

### 6. Run the FastAPI backend

```bash
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload
```

Open `http://localhost:8000/docs` to try `/query` interactively.

---

## Testing

The project uses two intentionally different testing layers:

### `rag_core` tests — integration-style, real dependencies

```bash
python -m pytest tests/ -v
```

Each test exercises the **real** pipeline — Qdrant, Ollama, and the reranker must all be running — so a full run takes minutes rather than seconds. Coverage includes:

- `test_customer_never_receives_internal_nodes` — the core security guarantee: a customer-role query never returns a source tagged `internal`.
- `test_internal_can_receive_internal_nodes` — confirms the filter isn't over-restrictive.
- `test_invalid_role_raises_error` — an unrecognized role raises a clear `ValueError` rather than failing silently.
- `test_fallback_for_irrelevant_question` — an out-of-scope question triggers the fixed fallback response with an empty source list, rather than an LLM-generated guess.

### `backend` tests — unit-style, mocked

```bash
python -m pytest backend/tests/ -v
```

These tests **do not** require Qdrant or Ollama to be running. `ask_with_sources` and `ingest_documents` are replaced with `monkeypatch` fakes (patched at their point of *use*, `backend.main`, rather than their point of *definition* — the correct target for `monkeypatch.setattr` in Python), so the API layer's routing, request validation, and error-handling logic can be verified in isolation, in well under a second. Coverage includes:

- `test_health` — liveness check.
- `test_query_success` — a mocked `QueryResult` round-trips correctly through the endpoint into JSON.
- `test_query_invalid_role` — a `ValueError` from the core is correctly translated into an HTTP 400.
- `test_query_missing_fields` — malformed request bodies are rejected by Pydantic validation (HTTP 422) before the core is ever called.
- `test_ingest` — the ingest endpoint returns the (mocked) ingestion result.

This split mirrors a common real-world pattern: fast, dependency-free tests for the API contract, and slower, high-fidelity tests for the actual retrieval/generation behavior.

---

## Known Limitations

- **CPU-only inference.** No GPU was used during development; a single query (hybrid retrieval + reranking + generation) can take on the order of a minute on modest hardware. Latency-sensitive deployments would need GPU inference or a hosted LLM API.
- **Relevance threshold calibrated on a very small probe set (3 questions).** `-8` is a reasonable working value given the observed score gap, but it has not been validated against a proper evaluation dataset with dozens of labeled questions. This is the natural next step before treating the threshold as production-ready.
- **No real authentication.** `user_role` is currently passed directly in the request body with no verification — anyone calling the API can claim to be `"internal"`. Production use requires real authentication (e.g., JWT with role claims) in front of this endpoint.
- **No conversation history / multi-turn memory.** Each query is stateless.
- **No document versioning or duplicate-content detection.** Re-ingesting a changed document adds new chunks rather than replacing old ones.
- **No query rewriting** (e.g., HyDE, multi-query expansion) — retrieval quality depends on the user's question being reasonably well-formed.
- **Sample documents are synthetic**, written specifically to test the internal/customer separation (both sets cover overlapping topics like refunds, at different levels of detail) rather than sourced from a real company's documentation.
- **Admin panel is not built.** `POST /ingest` exists as a stub; there is no UI for document management, re-indexing status, or reviewing unanswered questions.

---

## Possible Extensions

- Real evaluation dataset (expected question → expected source document → acceptable answer criteria) to properly calibrate `RELEVANCE_THRESHOLD` and measure retrieval quality over time.
- JWT-based authentication mapping to `user_role`.
- Conversation history support.
- Query rewriting for ambiguous or poorly-formed questions.
- Swap `nomic-embed-text` / `qwen2:7b` for hosted equivalents (OpenAI, Claude, Cohere) via a config change — the architecture was built provider-agnostic specifically to make this a drop-in swap.
- Admin UI for document management, re-indexing, and reviewing escalated/unanswered queries.