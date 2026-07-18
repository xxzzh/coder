"""Abbreviation dictionary retriever plugin."""

from __future__ import annotations

from .base import DomainRetriever, RetrieverHandler


def create(handler: RetrieverHandler) -> DomainRetriever:
    return DomainRetriever("abbreviation", handler)
