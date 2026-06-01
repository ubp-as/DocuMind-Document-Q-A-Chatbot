"""
ChromaDB vector store — singleton with persistent storage.
"""

import logging
from functools import lru_cache

import chromadb
from langchain_chroma import Chroma

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def _get_embeddings():
    """Return the configured embeddings model."""
    if settings.llm_provider == "anthropic":
        # Anthropic doesn't provide embeddings; fall back to OpenAI
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
        )
    else:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
        )


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    """Return (or create) the persistent ChromaDB vector store."""
    logger.info(
        "Initialising ChromaDB at '%s' (collection: %s)",
        settings.chroma_persist_dir,
        settings.chroma_collection_name,
    )
    embeddings = _get_embeddings()
    return Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )


def reset_vector_store_cache():
    """Clear the lru_cache — useful after adding documents."""
    get_vector_store.cache_clear()
