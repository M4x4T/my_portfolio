from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_core.query import ask_with_sources
from rag_core.models import QueryResult
from rag_core.ingestion import ingest_documents

app = FastAPI(
    title="RAG API",
    description="FastAPI backend for the role-based RAG system",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    user_role: str  # намеренно str, не Literal — валидацию делает ask_with_sources


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResult)
def query(request: QueryRequest):
    try:
        return ask_with_sources(request.question, request.user_role)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/ingest")
def ingest():
    # TODO: Add an auth check (admin-only) when the JWT appears
    return ingest_documents()