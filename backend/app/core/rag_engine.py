"""
RAG Engine — LangChain orchestration for retrieval-augmented generation.

Supports both OpenAI and Anthropic as LLM providers with a unified interface.
"""

import logging
import time
from typing import Any

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult

from app.core.config import get_settings
from app.core.vector_store import get_vector_store

logger = logging.getLogger(__name__)
settings = get_settings()


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise and helpful document assistant. \
Your job is to answer the user's question using ONLY the context excerpts provided below.

Rules:
- Base your answer strictly on the provided context. Do not use outside knowledge.
- If the context does not contain enough information to answer, say so clearly.
- Be concise but thorough. Cite the source document name when referencing specific facts.
- If multiple documents are relevant, synthesize the information coherently.

Context:
{context}
"""


# ── LLM factory ──────────────────────────────────────────────────────────────

def get_llm(streaming: bool = False):
    """Return the configured LLM instance."""
    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=settings.anthropic_model,
            anthropic_api_key=settings.anthropic_api_key,
            streaming=streaming,
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.llm_model,
            openai_api_key=settings.openai_api_key,
            streaming=streaming,
        )


# ── Main RAG query function ───────────────────────────────────────────────────

async def run_rag_query(question: str, top_k: int = None) -> dict[str, Any]:
    """
    Run a full RAG pipeline:
      1. Embed the question
      2. Retrieve top-k most relevant chunks from ChromaDB
      3. Stuff chunks into context + call LLM
      4. Return structured response with sources

    Returns:
        {
            "answer": str,
            "sources": [{"document": str, "page": int, "excerpt": str}],
            "model": str,
            "tokens_used": int,
            "latency_ms": int,
        }
    """
    k = top_k or settings.top_k_retrieval
    start = time.perf_counter()

    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": k})

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])

    llm = get_llm(streaming=False)

    # Track token usage via callback
    token_tracker = _TokenUsageCallback()
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    result = await rag_chain.ainvoke(
        {"input": question},
        config={"callbacks": [token_tracker]},
    )

    latency_ms = int((time.perf_counter() - start) * 1000)

    # Build source citations
    sources = []
    seen = set()
    for doc in result.get("context", []):
        meta = doc.metadata
        source_key = (meta.get("source", "unknown"), meta.get("page", 0))
        if source_key not in seen:
            seen.add(source_key)
            sources.append({
                "document": meta.get("source", "unknown"),
                "page": meta.get("page", 0),
                "excerpt": doc.page_content[:300].strip(),
            })

    model_name = (
        settings.anthropic_model
        if settings.llm_provider == "anthropic"
        else settings.llm_model
    )

    return {
        "answer": result["answer"],
        "sources": sources,
        "model": model_name,
        "tokens_used": token_tracker.total_tokens,
        "latency_ms": latency_ms,
    }


async def stream_rag_query(question: str, top_k: int = None):
    """
    Async generator that streams tokens from the RAG pipeline via SSE.

    Yields strings — either token chunks or a final JSON metadata event.
    """
    k = top_k or settings.top_k_retrieval
    start = time.perf_counter()

    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": k})

    # Retrieve docs first (non-streaming)
    docs = await retriever.ainvoke(question)

    context_text = "\n\n---\n\n".join(
        f"[Source: {d.metadata.get('source', '?')}, p.{d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in docs
    )

    messages = [
        ("system", SYSTEM_PROMPT.replace("{context}", context_text)),
        ("human", question),
    ]

    prompt = ChatPromptTemplate.from_messages(messages)
    llm = get_llm(streaming=True)
    chain = prompt | llm

    async for chunk in chain.astream({}):
        token = chunk.content if hasattr(chunk, "content") else str(chunk)
        if token:
            yield token

    latency_ms = int((time.perf_counter() - start) * 1000)
    sources = [
        {
            "document": d.metadata.get("source", "unknown"),
            "page": d.metadata.get("page", 0),
            "excerpt": d.page_content[:200].strip(),
        }
        for d in docs
    ]
    import json
    yield f"\n\n__SOURCES__{json.dumps({'sources': sources, 'latency_ms': latency_ms})}"


# ── Token tracking callback ───────────────────────────────────────────────────

class _TokenUsageCallback(AsyncCallbackHandler):
    """Accumulates token counts across LLM calls."""

    def __init__(self):
        self.total_tokens = 0

    async def on_llm_end(self, response: LLMResult, **kwargs):
        usage = response.llm_output or {}
        token_usage = usage.get("token_usage", {})
        self.total_tokens += token_usage.get("total_tokens", 0)
