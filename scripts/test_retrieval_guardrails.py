#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline regression checks for retrieval relevance guardrails."""

from __future__ import annotations

import query_knowledge_base as query_module
from query_knowledge_base import (
    filter_web_sources,
    merge_with_web_result,
    normalize_search_question,
    web_query,
    web_source_quality_score,
)


def main() -> None:
    question = "量子纠缠时什么"
    industrial_sources = [
        {
            "title": "Innovative Industrial Solutions",
            "url": "https://example.com/industrial",
            "snippet": "High-precision hydro demolition using 40,000 psi water systems.",
        },
        {
            "title": "Industrial Fabrication Services",
            "url": "https://example.com/fabrication",
            "snippet": "Equipment, servicing, repairs, and training.",
        },
    ]
    quantum_sources = [
        {
            "title": "量子纠缠",
            "url": "https://example.com/entanglement",
            "snippet": "量子纠缠是量子力学中的一种现象，多个粒子的状态之间存在关联。",
        },
        {
            "title": "量子纠错",
            "url": "https://example.com/error-correction",
            "snippet": "量子纠错用于保护量子信息免受噪声影响。",
        },
    ]

    assert normalize_search_question(question) == "量子纠缠是什么"
    assert web_query(question) == "量子纠缠"
    assert web_query("acwing是什么") == "acwing"
    assert [source["title"] for source in filter_web_sources(
        "acwing是什么",
        [{"title": "AcWing", "url": "https://www.acwing.com/", "snippet": "AcWing 算法交流平台。"}],
        5,
    )] == ["AcWing"]
    assert web_source_quality_score(
        "acwing是什么",
        {"title": "AcWing 百度百科", "snippet": "AcWing 是一个算法交流平台。", "relevance_score": 1.0},
    ) > web_source_quality_score(
        "acwing是什么",
        {"title": "AcWing", "snippet": "版权所有 | 隐私政策 | 联系我们", "relevance_score": 1.0},
    )
    original_fetch_url_text = query_module.fetch_url_text
    try:
        query_module.fetch_url_text = lambda request: """
### [App](http://www.baidu.com/link?url=app)

acwing 是一款同名校园工具 app。

### [AcWing 百度百科](http://www.baidu.com/link?url=baike)

AcWing 是一个算法交流平台。
"""
        baidu_sources = query_module.jina_baidu_search("acwing", 5)
        assert [source["title"] for source in baidu_sources] == ["AcWing 百度百科"]
    finally:
        query_module.fetch_url_text = original_fetch_url_text
    assert filter_web_sources(question, industrial_sources, 5) == []
    filtered = filter_web_sources(question, quantum_sources, 5)
    assert [source["title"] for source in filtered] == ["量子纠缠"]
    relaxed = filter_web_sources(
        "量子纠缠在量子通信中的实际应用",
        [{"title": "量子通信", "url": "https://example.com/communication", "snippet": "量子纠缠可用于量子通信。"}],
        5,
        allow_relaxed=True,
    )
    assert [source["title"] for source in relaxed] == ["量子通信"]

    english = filter_web_sources(
        "OpenAI latest model",
        [
            {
                "title": "Models - OpenAI API",
                "url": "https://example.com/models",
                "snippet": "All latest OpenAI models support text and image input.",
            },
            {
                "title": "Industrial systems",
                "url": "https://example.com/irrelevant",
                "snippet": "Equipment repairs and training.",
            },
        ],
        5,
    )
    assert [source["title"] for source in english] == ["Models - OpenAI API"]
    english_noise = filter_web_sources(
        "What is Ohm's law?",
        [
            {"title": "Ohm Cafe and Bar", "url": "https://example.com/cafe", "snippet": "Coffee and drinks."},
            {
                "title": "Ohm's law",
                "url": "https://example.com/ohms-law",
                "snippet": "Ohm's law relates voltage, current, and resistance.",
            },
        ],
        5,
        allow_relaxed=True,
    )
    assert [source["title"] for source in english_noise] == ["Ohm's law"]

    providers = (
        "jina_baidu_search",
        "bing_china_search",
        "duckduckgo_search",
        "wikipedia_search",
        "jina_bing_search",
        "bing_rss_search",
    )
    originals = {name: getattr(query_module, name) for name in providers}
    try:
        setattr(query_module, "duckduckgo_search", lambda question, limit: quantum_sources[:1])
        setattr(query_module, "jina_baidu_search", lambda question, limit: [])
        setattr(query_module, "bing_china_search", lambda question, limit: [])
        setattr(
            query_module,
            "wikipedia_search",
            lambda question, limit: [
                quantum_sources[0],
                {
                    "title": "量子纠缠应用",
                    "url": "https://example.com/entanglement-applications",
                    "snippet": "量子纠缠可用于量子通信与量子计算研究。",
                },
            ],
        )
        setattr(query_module, "jina_bing_search", lambda question, limit: [])
        setattr(query_module, "bing_rss_search", lambda question, limit: [])
        result = query_module.web_search(question, 5)
        assert len(result["sources"]) == 2
        assert set(result["web_search_providers"]) == {"duckduckgo", "wikipedia"}

        hybrid = merge_with_web_result(
            {
                "source_type": "knowledge_base",
                "sources": [
                    {
                        "source_id": 1,
                        "file": "local.md",
                        "file_path": "knowledge_base/raw/local.md",
                        "chunk_id": "local-1",
                        "citation": "量子纠缠是量子系统中的关联现象。",
                    }
                ],
            },
            result,
            5,
        )
        assert hybrid["source_type"] == "hybrid_web_knowledge"
        origins = [source["source_origin"] for source in hybrid["sources"]]
        assert origins[0] == "local"
        assert origins.count("local") == 1
        assert origins.count("web") == 2

        irrelevant = lambda question, limit: industrial_sources  # noqa: E731
        for name in providers:
            setattr(query_module, name, irrelevant)
        result = query_module.web_search(question, 5)
        assert result["web_search_reason"] == "no_relevant_results"
        assert "没有找到与问题足够相关的可信结果" in result["answer"]

        def unavailable(question: str, limit: int) -> list[dict[str, str]]:
            raise RuntimeError("offline")

        for name in providers:
            setattr(query_module, name, unavailable)
        result = query_module.web_search(question, 5)
        assert result["web_search_reason"] == "providers_unavailable"
        assert "搜索源暂时无法访问" in result["answer"]
    finally:
        for name, provider in originals.items():
            setattr(query_module, name, provider)
    print("retrieval guardrails: ok")


if __name__ == "__main__":
    main()
