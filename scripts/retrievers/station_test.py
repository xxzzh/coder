"""Station-test table retriever plugin."""

from __future__ import annotations

from .base import DomainRetriever, RetrieverHandler


def create(handler: RetrieverHandler) -> DomainRetriever:
    return DomainRetriever("station_test", handler)
