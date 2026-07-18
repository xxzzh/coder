"""General hybrid RAG retriever plugin marker."""

from __future__ import annotations

from .base import DomainRetriever, RetrieverHandler


def create(handler: RetrieverHandler) -> DomainRetriever:
    return DomainRetriever("general_rag", handler)
