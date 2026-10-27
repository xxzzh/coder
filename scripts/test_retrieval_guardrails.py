#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline regression checks for retrieval relevance guardrails."""

from __future__ import annotations

import json
from pathlib import Path
import os
import re
import sqlite3
import tempfile
import zipfile

os.environ["LKA_API_USAGE_LOG"] = "off"  # test calls must not land in the real usage log

import ingest_knowledge_base as ingest_module
from kb_ingest import ocr as ocr_module  # OCR code looks shutil.which up here
from kb_ingest import pdf as pdf_module  # pdftotext reading looks find_pdftotext and subprocess up here
import model_capabilities
import query_knowledge_base as query_module
from kb import web as web_module  # the web search code looks its providers and fetchers up here
import vector_store
from ingest_knowledge_base import read_xlsx
from query_knowledge_base import (
    chunk_information_quality,
    citation_for_text,
    extract_station_table_items,
    extract_station_table_records,
    filter_web_sources,
    best_summary_source_ids,
    human_text,
    merge_with_web_result,
    normalize_search_question,
    polish_internal_answer_text,
    web_query,
    web_evidence_quality_score,
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
    assert query_module.normalized_retrieval_question("SR是什么意思") == "SR"
    assert query_module.abbreviation_query_term("SR是什么意思") == "SR"
    assert web_query(question) == "量子纠缠"
    assert web_query("acwing是什么") == "acwing"
    assert web_query("雷军是谁") == "雷军"
    assert web_query("雷军是什么") == "雷军"
    polished = polish_internal_answer_text(
        "根据本地知识库，├── STPM.sln / STPM.exe.config | USB2.0/USB3.2 \\ ProgramData ![Image 1](x) [1]。"
    )
    assert "|" not in polished
    assert "├" not in polished and "Image" not in polished
    assert "STPM.sln / STPM.exe.config" in polished and "USB2.0/USB3.2" in polished
    assert "[1]" in polished
    # Identifiers, paths and numeric ranges must survive answer polishing.
    polished = polish_internal_answer_text(
        "缓存 `Output/LocalData/`，**EnableReadLocalStatus**，GetWorkOrderData_V20()，"
        "C:\\ProgramData\\SAMPLELIST.txt，SAMPLE_ENUM_VALUE，[1.25,3.75]，接口*2，80%*标称值 [1]"
    )
    for fragment in (
        "Output/LocalData/",
        "EnableReadLocalStatus",
        "GetWorkOrderData_V20()",
        "C:\\ProgramData\\SAMPLELIST.txt",
        "SAMPLE_ENUM_VALUE",
        "[1.25,3.75]",
        "接口*2",
        "80%*标称值",
    ):
        assert fragment in polished, (fragment, polished)
    assert "**" not in polished and "`" not in polished
    polished = polish_internal_answer_text("LogMsgType 枚举 | 值 | 颜色 | |----|------| | INFO | 蓝色 |")
    assert "---" not in polished and "INFO，蓝色" in polished
    assert human_text("public enum EnumSample { SAMPLE_ENUM_VALUE, // 示例值 }").count("SAMPLE_ENUM_VALUE") == 1
    assert query_module.requested_station_name("FRT站位有哪些测试项") == "FRT"
    assert query_module.requested_station_name("FRT有哪些测试项") == "FRT"
    assert query_module.requested_station_name("FRT测试项有哪些") == "FRT"
    assert query_module.requested_station_name("FRT的测试项目有什么") == "FRT"
    assert query_module.canonical_question_for_cache("FRT站位有哪些测试项") == "station:frt:items"
    assert query_module.canonical_question_for_cache("FRT有哪些测试项") == "station:frt:items"
    assert query_module.cache_key("FRT站位有哪些测试项", 5, False) == query_module.cache_key("FRT有哪些测试项", 5, False)
    assert query_module.normalized_retrieval_question("工单数据数据获取流程") == "工单数据获取流程"
    assert query_module.meaningful_query_terms("客户类型有那些") == ["客户类型"]
    assert query_module.meaningful_query_terms("LogMsgType有那些值") == ["logmsgtype"]
    assert query_module.cache_key("FFT测试", 5, False) == query_module.cache_key("人工功能测试", 5, False)
    assert query_module.cache_key("FFT测试项", 5, False) == query_module.cache_key("FFT测试", 5, False)
    # Different questions that merely share a topic word must not share one cached answer.
    assert query_module.cache_key("本地缓存的配置开关", 5, False) != query_module.cache_key("本地缓存的完整生命周期", 5, False)
    assert query_module.cache_key("人工功能测试的IR Camera判定标准", 5, False) != query_module.cache_key("人工功能测试", 5, False)
    # Relevance gate: a paraphrase of a section title passes, a known subject with an unknown attribute does not.
    assert query_module.local_result_is_relevant(
        [{"rerank_score": 0.48, "keyword_score": 1.0, "fts_score": 0.1, "text": "十一、添加新设备完整步骤 共需修改 3 处"}],
        "如何添加一个新设备",
    )
    assert not query_module.local_result_is_relevant(
        [{"rerank_score": 0.6, "keyword_score": 1.0, "fts_score": 0.3, "text": "AB123装备测试计划 测试流程图 单板测试"}],
        "AB123的售价是多少",
    )
    assert query_module.query_cache_variant({}, allow_api=False, use_web=False).startswith("local-only:")
    assert query_module.query_cache_variant({}, allow_api=False, use_web=True).startswith("web-enabled:")
    assert query_module.query_cache_variant({}, allow_api=False, use_web=False) != query_module.query_cache_variant(
        {}, allow_api=False, use_web=True
    )
    assert "embedding-disabled" in query_module.query_cache_variant({}, allow_api=False, use_web=False)
    assert model_capabilities.classify_query("PR")["type"] == "abbreviation"
    assert model_capabilities.classify_query("FRT站位有哪些测试项")["type"] == "excel_table"
    assert model_capabilities.classify_query("LogMsgType有哪些值")["type"] == "code_or_document"
    with tempfile.TemporaryDirectory() as temp_vector_dir:
        vector_status = vector_store.vector_status(sqlite_chunk_count=0, vector_dir=Path(temp_vector_dir))
        assert vector_status["ready"] is False
        assert vector_status["reason"] in {"faiss_dependency_missing", "vector_index_missing"}
    assert query_module.station_test_item_question("FFT站位测试")
    assert query_module.canonical_question_for_cache("FFT站位测试") == "station:fft:items"
    assert query_module.canonical_question_for_cache("FFT站位测试项") == "station:fft:items"
    assert query_module.abbreviation_query_term("PR") == "PR"
    assert query_module.abbreviation_query_term("PR是什么") == "PR"
    assert query_module.meaningful_query_terms("GW") == ["gw"]
    assert query_module.suggested_local_questions("SWDL2")[0] == "SWDL2有哪些测试项"
    assert query_module.suggested_local_questions("swdl2")[0] == "SWDL2有哪些测试项"
    assert query_module.suggested_local_questions("hpd test")[0] == "HPD有哪些测试项"
    assert query_module.suggested_local_questions("电源键测试")[0] == "电源键测试是什么"
    assert query_module.suggested_local_questions("我的个人密码是")[0] == "个人密码是什么"
    assert query_module.suggested_local_questions("请问我的个人密码是什么")[0] == "个人密码是什么"
    # These checks need an index that contains the abbreviation SR; other users' indexes skip them.
    index_has_sr = query_module.DB_PATH.exists() and query_module.structured_abbreviation_result(query_module.DB_PATH, "SR") is not None
    if query_module.DB_PATH.exists():
        assert query_module.suggested_local_questions("ZZZ999有哪些测试项", query_module.DB_PATH) == []
    if index_has_sr:
        assert query_module.suggested_local_questions("SR是什么意思", query_module.DB_PATH)[0] == "SR是什么"
        assert query_module.suggested_local_questions("SR有哪些测试项?", query_module.DB_PATH)[0] == "SR是什么"
    guidance = query_module.no_index_user_guidance("SWDL2")
    assert guidance["severity"] == "error"
    assert "与「SWDL2」相关的索引和 FAQ" in guidance["message"]
    assert "SWDL2有哪些测试项" in guidance["message"]
    assert "HPD有哪些测试项" in query_module.no_index_user_guidance("hpd test")["message"]
    no_match_guidance = query_module.local_no_match_user_guidance("我的个人密码是")
    assert no_match_guidance["kind"] == "local_no_match"
    assert "与「个人密码」相关" in no_match_guidance["message"]
    assert "个人密码是什么" in no_match_guidance["message"]
    if index_has_sr:
        sr_guidance = query_module.local_no_match_user_guidance("SR是什么意思", query_module.DB_PATH)
        assert "SR是什么" in sr_guidance["message"]
        assert "SR有哪些测试项" not in sr_guidance["message"]
        sr_test_guidance = query_module.local_no_match_user_guidance("SR有哪些测试项?", query_module.DB_PATH)
        assert "SR是什么" in sr_test_guidance["message"]
        assert "SR有哪些测试项" not in sr_test_guidance["message"]
    if query_module.DB_PATH.exists():
        missing_guidance = query_module.local_no_match_user_guidance("ZZZ999有哪些测试项", query_module.DB_PATH)
        assert missing_guidance["suggested_questions"] == []
        assert "你是否想问" not in missing_guidance["message"]
    original_processed_dir = query_module.PROCESSED_DIR
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            query_module.PROCESSED_DIR = Path(temp_dir) / "processed"
            missing_index_result = query_module.query(Path(temp_dir) / "missing.db", "SWDL2", use_web=False, allow_api=False)
            assert missing_index_result["degrade_reason"] == "local_index_missing"
            assert missing_index_result["user_guidance"]["kind"] == "no_index_no_faq"
            assert "SWDL2有哪些测试项" in missing_index_result["user_guidance"]["message"]
    finally:
        query_module.PROCESSED_DIR = original_processed_dir
    with tempfile.TemporaryDirectory() as temp_dir:
        raw_dir = Path(temp_dir)
        sample = raw_dir / "same-size.txt"
        fixed_ns = 1_700_000_000_123_456_789
        sample.write_text("alpha", encoding="utf-8")
        os.utime(sample, ns=(fixed_ns, fixed_ns))
        first_fingerprint = ingest_module.source_fingerprint(raw_dir)
        sample.write_text("omega", encoding="utf-8")
        os.utime(sample, ns=(fixed_ns, fixed_ns))
        assert ingest_module.source_fingerprint(raw_dir) != first_fingerprint
    dictionary_text = (
        "P | P | PR | | Pilot Run | | | | | | | 试运行 | | | | | | | Back\n"
        "P | P | PR | | Public Relations | | | | | | | 公共关系 | | | | | | | Back\n"
        "G | G | GW | | Gross Weight | | | | | | | 毛重 | | | | | | | Back\n"
    )
    assert query_module.dictionary_entry_rows(dictionary_text, "PR") == [
        {
            "abbr": "PR",
            "english": "Pilot Run",
            "chinese": "试运行",
            "line": "P | P | PR | | Pilot Run | | | | | | | 试运行 | | | | | | | Back",
        },
        {
            "abbr": "PR",
            "english": "Public Relations",
            "chinese": "公共关系",
            "line": "P | P | PR | | Public Relations | | | | | | | 公共关系 | | | | | | | Back",
        },
    ]
    fft_manual_text = (
        "14 | 人工功能测试 MES：FFT | IR CameraTest | IR Camera Test | IQCapture |\n"
        "15 | | Sensor Test | Lid Hall Sensor Test | SensorTest.dll |\n"
        "16 | | LCD明暗度测试 | LCD明暗度测试 | LCDTest.dll |\n"
        "17 | | Power Button | 开机键测试 | STPM.exe |\n"
        "18 | | G-sensor Test | 零偏校准测试 | STPM.exe |\n"
        "| | 手写笔功能测试 | 充电功能测试 | STPM.exe |\n"
        "19 | | 异音测试 | 风扇异音测试 | STPM.exe |\n"
        "20 | | | 开合转轴异响 | NA |\n"
        "21 | | RSSI | 天线信号强度测试 | STPM.exe |\n"
        "22 | | WIFI吞吐量测试 | T-put测试 | STPM.exe |\n"
        "23 | | FFT 过站 | FFT扫描过站 | MES |\n"
    )
    assert query_module.extract_fft_manual_items(fft_manual_text) == [
        "IR CameraTest / IR Camera Test",
        "Sensor Test / Lid Hall Sensor Test",
        "LCD明暗度测试",
        "Power Button / 开机键测试",
        "G-sensor Test / 零偏校准测试",
        "手写笔功能测试 / 充电功能测试",
        "异音测试 / 风扇异音测试",
        "异音测试 / 开合转轴异响",
        "RSSI / 天线信号强度测试",
        "WIFI吞吐量测试 / T-put测试",
        "FFT 过站 / FFT扫描过站",
    ]
    assert query_module.extract_fft_station_items(
        "| | 测试项 | 测试子项 | 工具名称 |\n"
        "1 | 自动化测试 | 寸动线自动化测试 | 清洁屏幕 |\n"
        "2 | 自动化功能测试 MES:ATS | 摄像头测试 | 白板灯测试 棋盘格测试 |\n"
        "3 | | LCD AOI 测试 | LCD 点.线 |\n"
        "14 | 人工功能测试 MES：FFT | IR CameraTest | IR Camera Test |\n"
        "| | 手写笔功能测试 | 充电功能测试 | STPM.exe |\n"
        "19 | | 异音测试 | 风扇异音测试 | STPM.exe |\n"
        "20 | | | 开合转轴异响 | NA |\n"
        "23 | | FFT 过站 | FFT扫描过站 | MES |\n"
    ) == [
        "自动化测试 / 寸动线自动化测试 / 清洁屏幕",
        "自动化功能测试 MES:ATS / 摄像头测试 / 白板灯测试 棋盘格测试",
        "自动化功能测试 MES:ATS / LCD AOI 测试 / LCD 点.线",
        "人工功能测试 MES：FFT / IR CameraTest / IR Camera Test",
        "人工功能测试 MES：FFT / 手写笔功能测试 / 充电功能测试",
        "人工功能测试 MES：FFT / 异音测试 / 风扇异音测试",
        "人工功能测试 MES：FFT / 异音测试 / 开合转轴异响",
        "人工功能测试 MES：FFT / FFT 过站 / FFT扫描过站",
    ]
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
    lei_sources = [
        {
            "source_id": 1,
            "title": "雷军 百度百科",
            "snippet": "雷军，1969年12月16日出生，小米科技有限责任公司创始人、董事长、首席执行官（CEO）。",
            "relevance_score": 1.0,
            "quality_score": 1.8,
        },
        {
            "source_id": 2,
            "title": "管理团队-小米商城",
            "citation": "Xiaomi Pad 8 Pro 2699元起 服务中心 预约维修 7天无理由退货 京ICP备10046444号 营业执照 探索黑科技，小米为发烧而生",
            "relevance_score": 1.0,
            "quality_score": 1.8,
        },
        {
            "source_id": 3,
            "title": "雷军 - 百度图片",
            "snippet": "![Image 34 变清晰](https://image.baidu.com/search/detail?word=雷军)",
            "relevance_score": 1.0,
            "quality_score": 1.0,
        },
    ]
    assert web_evidence_quality_score("雷军是谁", lei_sources[0]) > web_evidence_quality_score("雷军是谁", lei_sources[1])
    assert best_summary_source_ids("雷军是谁", lei_sources) == [1]
    original_fetch_url_text = web_module.fetch_url_text
    try:
        web_module.fetch_url_text = lambda request: """
### [App](http://www.baidu.com/link?url=app)

acwing 是一款同名校园工具 app。

### [AcWing 百度百科](http://www.baidu.com/link?url=baike)

AcWing 是一个算法交流平台。
"""
        baidu_sources = query_module.jina_baidu_search("acwing", 5)
        assert [source["title"] for source in baidu_sources] == ["AcWing 百度百科"]
    finally:
        web_module.fetch_url_text = original_fetch_url_text
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
    assert chunk_information_quality("站位 | 测试项目 | 工具名称 | 测试门限 | 卡关/判定方式") < 0.6
    assert human_text("采集范围 0~50V") == "采集范围 0~50V"
    assert "CN2:8路电压采集通道，采集范围0~50V" in citation_for_text(
        "ZDE-DAQ-V1采集板说明\nCN1:USB通讯接口\nCN2:8路电压采集通道，采集范围0~50V",
        "ZDE-DAQ-V1采集板的CN2接口是什么",
    )
    assert not query_module.local_result_is_relevant(
        [{"rerank_score": 0.8, "keyword_score": 1.0, "fts_score": 1.0, "text": "Extended by quantum mechanics."}],
        "quantum entanglement",
    )
    assert query_module.lexical_query_coverage("恢复机制逻辑", "八、本地缓存与恢复机制") == 1.0
    assert chunk_information_quality("## 目录\n- [恢复机制](#恢复机制)\n- [配置系统](#配置系统)\n- [运行流程](#运行流程)\n- [API](#api)") < 0.5
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE chunks(chunk_id TEXT PRIMARY KEY, doc_id TEXT, chunk_index INTEGER, text TEXT);
        INSERT INTO chunks VALUES ('guide-0122', 'guide', 122, '## 八、本地缓存与恢复机制\n### 用途\n支持断点续测。');
        INSERT INTO chunks VALUES ('guide-0123', 'guide', 123, '### 缓存文件\nOutput/LocalData/');
        INSERT INTO chunks VALUES ('guide-0124', 'guide', 124, '### 状态值\n-1 等待测试，0 测试失败，1 测试通过。');
        INSERT INTO chunks VALUES ('guide-0125', 'guide', 125, '### 恢复逻辑\n若 EnableReadLocalStatus = true，状态 = 1 时直接标记 PASS，跳过执行。');
        INSERT INTO chunks VALUES ('guide-0126', 'guide', 126, '## 九、下一个章节\n不应拼接。');
        """
    )
    expanded_rows = query_module.expand_section_context(
        conn,
        [
            {
                "chunk_id": "guide-0122",
                "doc_id": "guide",
                "chunk_index": 122,
                "file_type": "md",
                "text": "## 八、本地缓存与恢复机制\n### 用途\n支持断点续测。",
                "rerank_score": 0.8,
                "keyword_score": 1.0,
                "fts_score": 1.0,
            }
        ],
        "恢复机制是什么",
    )
    conn.close()
    assert expanded_rows[0]["context_chunk_ids"] == ["guide-0122", "guide-0123", "guide-0124", "guide-0125"]
    assert "状态 = 1" in query_module.citation_for_retrieval_row(expanded_rows[0], "恢复机制是什么")
    assert "下一个章节" not in expanded_rows[0]["text"]
    assert "INFO" in query_module.citation_for_retrieval_row(
        {"text": "LogMsgType 枚举 | 值 | 颜色 | 用途 |\n| INFO | 蓝色 | 正常信息 |\n| WARNING | 黄色 | 警告 |"},
        "LogMsgType有那些值",
    )
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT);
        INSERT INTO metadata VALUES ('indexed_at', 'test-index');
        CREATE TABLE documents(doc_id TEXT PRIMARY KEY, file_name TEXT, file_path TEXT, file_type TEXT);
        CREATE TABLE chunks(
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT,
            chunk_index INTEGER,
            text TEXT,
            embedding TEXT,
            indexed_at TEXT
        );
        CREATE TABLE embedding_lsh(bucket TEXT, chunk_id TEXT, doc_id TEXT, indexed_at TEXT);
        """
    )
    conn.executemany(
        "INSERT INTO documents VALUES (?, ?, ?, ?)",
        [
            ("doc-physics", "physics.md", "physics.md", "md"),
            ("doc-cooking", "cooking.md", "cooking.md", "md"),
        ],
    )

    def insert_lsh_chunk(chunk_id: str, doc_id: str, text: str) -> None:
        embedding = query_module.local_embedding(text)
        raw_embedding = [[index, value] for index, value in sorted(embedding.items())]
        conn.execute("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)", (chunk_id, doc_id, 1, text, json.dumps(raw_embedding), "test-index"))
        for bucket in query_module.embedding_lsh_buckets(embedding):
            conn.execute("INSERT INTO embedding_lsh VALUES (?, ?, ?, ?)", (bucket, chunk_id, doc_id, "test-index"))

    insert_lsh_chunk("physics-1", "doc-physics", "量子纠缠是量子力学中的关联现象，多个粒子的状态之间存在关联。")
    insert_lsh_chunk("cooking-1", "doc-cooking", "番茄炒蛋需要番茄、鸡蛋和少量盐。")
    from kb import retrieval as retrieval_module

    original_vector_threshold = retrieval_module.VECTOR_INDEX_THRESHOLD_CHUNKS
    original_vector_fallback_min = retrieval_module.VECTOR_CANDIDATE_FALLBACK_MIN
    try:
        retrieval_module.VECTOR_INDEX_THRESHOLD_CHUNKS = 1
        retrieval_module.VECTOR_CANDIDATE_FALLBACK_MIN = 1
        semantic_rows = query_module.semantic_candidates(conn, "量子纠缠是什么", 5, {"indexed_at": "test-index"})
    finally:
        retrieval_module.VECTOR_INDEX_THRESHOLD_CHUNKS = original_vector_threshold
        retrieval_module.VECTOR_CANDIDATE_FALLBACK_MIN = original_vector_fallback_min
        conn.close()
    assert semantic_rows[0]["chunk_id"] == "physics-1"
    assert semantic_rows[0]["semantic_candidate_mode"] == "lsh"
    title, page_text = query_module.extract_webpage_text(
        """
        <html><head><title>量子纠缠说明</title><script>ignore()</script></head>
        <body><header>导航文字</header><main><h1>量子纠缠</h1>
        <p>量子纠缠是量子力学中的一种关联现象。</p>
        <p>它可用于量子通信与量子计算研究。</p></main><footer>版权信息</footer></body></html>
        """
    )
    assert title == "量子纠缠说明"
    assert "量子纠缠是量子力学中的一种关联现象" in page_text
    assert "ignore" not in page_text
    assert "导航文字" not in page_text
    assert "版权信息" not in page_text
    original_fetch_public_webpage_text = web_module.fetch_public_webpage_text
    try:
        web_module.fetch_public_webpage_text = lambda *args, **kwargs: (
            "<main><p>量子纠缠是量子力学中的一种现象，多个粒子的状态之间存在关联。</p></main>",
            "https://example.com/entanglement",
            "text/html",
        )
        researched = query_module.research_web_source("量子纠缠是什么", quantum_sources[0])
        assert researched["page_fetch_status"] == "ok"
        assert researched["content_source"] == "webpage"
        assert "多个粒子的状态之间存在关联" in researched["citation"]
    finally:
        web_module.fetch_public_webpage_text = original_fetch_public_webpage_text
    assert [citation["source_id"] for citation in query_module.synthesis_citations(
        {
            "source_type": "web_search",
            "web_pages_extracted": 1,
            "sources": [
                {"source_id": 1, "source_origin": "web", "snippet": "搜索摘要"},
                {"source_id": 2, "source_origin": "web", "content_source": "webpage", "citation": "网页正文证据"},
            ],
        }
    )] == [2]
    fat_items = extract_station_table_items(
        "工序名称：FAT\n"
        "岗位资源： | MES Station Check | MES站别检测 | STPM.exe | NA |\n"
        "6 | | ATO 系统时间同步 | 系统时间同步 | STPM.exe | NA |\n"
        "7 | | FAN Test | FAN speed | STPM.exe | NA |\n"
        "工作表 sheet6\n",
        "FAT",
    )
    assert fat_items == ["MES Station Check / MES站别检测", "ATO 系统时间同步 / 系统时间同步", "FAN Test / FAN speed"]
    fat_records = extract_station_table_records(
        "工序名称：FAT\n"
        "岗位资源： | MES Station Check | MES站别检测 | STPM.exe | NA | LC | NA | 1.读取主板序列号；\n"
        "2.查询MES站点状态； | MES | 待导入 |\n"
        "6 | | FAN Test | FAN speed | STPM.exe | NA | LC | NA | 1.读取风扇转速； | 程序 | 待导入 |\n"
        "工作表 sheet6\n",
        "FAT",
    )
    assert fat_records == [
        {"item": "MES Station Check / MES站别检测", "scheme": "1.读取主板序列号； 2.查询MES站点状态；"},
        {"item": "FAN Test / FAN speed", "scheme": "1.读取风扇转速；"},
    ]
    frt_records = extract_station_table_records(
        "工作表 老化\n"
        "1 | FRT | AC和网络在位Check | AC和网络在位Check | STPM.exe | NA | LC | NA | 检查AC与网络； | 程序 |\n"
        "2 |  | Mainboard Test | AC in Check | STPM.exe | NA | LC | NA | 检查AC； | 程序 |\n"
        "3 |  |  | RTC Test | STPM.exe | NA | LC | NA | 检查RTC； | 程序 |\n"
        "4 |  | Write Number Check | Check_DMI（SMBIOS） | STPM.exe | NA | LC | NA | 检查DMI； | 程序 |\n",
        "FRT",
    )
    assert [record["item"] for record in frt_records] == [
        "AC和网络在位Check",
        "Mainboard Test / AC in Check",
        "Mainboard Test / RTC Test",
        "Write Number Check / Check_DMI（SMBIOS）",
    ]
    with tempfile.TemporaryDirectory() as temp_dir:
        xlsx_path = Path(temp_dir) / "coordinates.xlsx"
        with zipfile.ZipFile(xlsx_path, "w") as archive:
            archive.writestr(
                "xl/workbook.xml",
                """<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
                xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
                <sheets><sheet name="Ordered Sheet" sheetId="1" r:id="rId1"/></sheets></workbook>""",
            )
            archive.writestr(
                "xl/_rels/workbook.xml.rels",
                """<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
                <Relationship Id="rId1" Target="worksheets/sheet10.xml"
                Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"/>
                </Relationships>""",
            )
            archive.writestr(
                "xl/worksheets/sheet10.xml",
                """<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
                <sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>A</t></is></c>
                <c r="C1" t="inlineStr"><is><t>C</t></is></c></row></sheetData></worksheet>""",
            )
        assert read_xlsx(xlsx_path) == "工作表 Ordered Sheet\nA |  | C"

    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_path = Path(temp_dir) / "sample.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n")
        original_find_pdftotext = pdf_module.find_pdftotext
        original_subprocess_run = pdf_module.subprocess.run

        def fake_run(command, **kwargs):
            Path(command[-1]).write_text("4.2 HPD (Human Presence Detection) Test", encoding="utf-8")

            class Completed:
                returncode = 0

            return Completed()

        try:
            pdf_module.find_pdftotext = lambda: "pdftotext"
            pdf_module.subprocess.run = fake_run
            assert pdf_module._read_pdf_with_pdftotext(pdf_path) == "4.2 HPD (Human Presence Detection) Test"
        finally:
            pdf_module.find_pdftotext = original_find_pdftotext
            pdf_module.subprocess.run = original_subprocess_run

    providers = (
        "jina_baidu_search",
        "bing_china_search",
        "duckduckgo_search",
        "wikipedia_search",
        "jina_bing_search",
        "bing_rss_search",
    )
    originals = {name: getattr(web_module, name) for name in providers}
    try:
        setattr(web_module, "duckduckgo_search", lambda question, limit: quantum_sources[:1])
        setattr(web_module, "jina_baidu_search", lambda question, limit: [])
        setattr(web_module, "bing_china_search", lambda question, limit: [])
        setattr(
            web_module,
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
        setattr(web_module, "jina_bing_search", lambda question, limit: [])
        setattr(web_module, "bing_rss_search", lambda question, limit: [])
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
            setattr(web_module, name, irrelevant)
        result = query_module.web_search(question, 5)
        assert result["web_search_reason"] == "no_relevant_results"
        assert "没有找到与问题足够相关的可信结果" in result["answer"]

        def unavailable(question: str, limit: int) -> list[dict[str, str]]:
            raise RuntimeError("offline")

        for name in providers:
            setattr(web_module, name, unavailable)
        result = query_module.web_search(question, 5)
        assert result["web_search_reason"] == "providers_unavailable"
        assert "搜索源暂时无法访问" in result["answer"]
    finally:
        for name, provider in originals.items():
            setattr(web_module, name, provider)

    api_config = {
        "LKA_USE_API": "true",
        "LKA_API_PROVIDER": "openai-compatible",
        "LKA_API_BASE_URL": "https://example.com/v1",
        "LKA_API_KEY": "test-key",
        "LKA_API_MODEL": "test-model",
    }
    original_chat_completion = query_module.chat_completion
    original_web_search = query_module.web_search
    original_enrich_web_search_result = query_module.enrich_web_search_result
    try:
        steps: list[str] = []
        query_module.web_search = lambda *args, **kwargs: (
            steps.append("search")
            or {
                "answer": "网页搜索摘要",
                "source_type": "web_search",
                "sources": [{"title": "外部资料", "url": "https://example.com/", "snippet": "外部资料摘要"}],
                "web_search_used": True,
            }
        )
        query_module.enrich_web_search_result = lambda *args, **kwargs: (
            steps.append("fetch")
            or {
                **args[1],
                "sources": [{"title": "外部资料", "url": "https://example.com/", "citation": "整理后的网页证据"}],
                "retrieval_mode": "web_search_page_rag",
            }
        )
        query_module.chat_completion = lambda *args, **kwargs: steps.append("api") or "基于网页证据的 API 精炼答案"
        web_grounded = query_module.no_local_answer(
            True,
            "外部问题",
            "本地知识库没有找到足够依据。",
            config=api_config,
            allow_api=True,
        )
        assert steps == ["search", "fetch", "api"]
        assert web_grounded["source_type"] == "web_search"
        assert web_grounded["web_search_used"] is True
        assert web_grounded["api_used"] is True
        assert web_grounded["retrieval_mode"] == "web_research_grounded_api_summary"
        assert web_grounded["answer"] == "基于网页证据的 API 精炼答案"

        steps.clear()
        query_module.chat_completion = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("api offline"))
        web_fallback = query_module.no_local_answer(
            True,
            "外部问题",
            "本地知识库没有找到足够依据。",
            config=api_config,
            allow_api=True,
        )
        assert web_fallback["source_type"] == "web_search"
        assert web_fallback["web_search_used"] is True
        assert web_fallback["retrieval_mode"] == "web_research_local_summary_after_api_error"
        assert "api offline" in web_fallback["api_error"]

        steps.clear()
        no_api = query_module.no_local_answer(
            True,
            "外部问题",
            "本地知识库没有找到足够依据。",
            config=api_config,
            allow_api=False,
        )
        assert steps == ["search", "fetch"]
        assert no_api["api_used"] is False
        assert query_module.freshness_sensitive_question("DeepSeek 什么模型名即将停止使用？")
    finally:
        query_module.chat_completion = original_chat_completion
        query_module.web_search = original_web_search
        query_module.enrich_web_search_result = original_enrich_web_search_result

    original_search = query_module.search
    original_trigger_async_update = query_module.trigger_async_update
    original_web_search = query_module.web_search
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "knowledge.db"
        conn = sqlite3.connect(db_path)
        conn.executescript(
            """
            CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT);
            INSERT INTO metadata(key, value) VALUES ('indexed_at', 'test-index');
            CREATE TABLE documents(
                doc_id TEXT PRIMARY KEY,
                file_name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_type TEXT NOT NULL,
                text TEXT NOT NULL
            );
            CREATE TABLE query_cache(
                cache_key TEXT PRIMARY KEY,
                question TEXT NOT NULL,
                answer_json TEXT NOT NULL,
                hit_count INTEGER NOT NULL,
                index_version TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        conn.execute(
            "INSERT INTO documents(doc_id, file_name, file_path, file_type, text) VALUES (?, ?, ?, ?, ?)",
            (
                "fat-doc",
                "fat.xlsx",
                "excel/fat.xlsx",
                "xlsx",
                "工序名称：FAT\n"
                "岗位资源： | MES Station Check | MES站别检测 | STPM.exe | NA | LC | NA | 1.读取主板序列号； | MES | 待导入 |\n"
                "6 | | FAN Test | FAN speed | STPM.exe | NA | LC | NA | 1.读取风扇转速； | 程序 | 待导入 |\n"
                "工作表 sheet6\n",
            ),
        )
        conn.commit()
        conn.close()
        try:
            query_module.trigger_async_update = lambda *args, **kwargs: False
            query_module.web_search = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("structured local answer must not search web"))
            structured = query_module.query(
                db_path,
                "FAT 站位包含哪些测试项及其对应的测试方案是什么",
                use_web=True,
                allow_api=False,
            )
            assert structured["source_type"] == "knowledge_base"
            assert structured["retrieval_mode"] == "local_structured_table"
            assert structured["web_search_used"] is False
            assert structured["answer"].count("\n- ") == 2
            assert "1.读取主板序列号" in structured["answer"]
            shorthand = query_module.query(
                db_path,
                "FAT有哪些测试项及其对应的测试方案是什么",
                use_web=True,
                allow_api=False,
            )
            assert shorthand["retrieval_mode"] == "local_structured_table"
            assert shorthand["answer"] == structured["answer"]
            assert shorthand["cache_hit"] is True
        finally:
            query_module.trigger_async_update = original_trigger_async_update
            query_module.web_search = original_web_search

    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "knowledge.db"
        sqlite3.connect(db_path).close()
        try:
            query_module.trigger_async_update = lambda *args, **kwargs: True
            query_module.search = lambda *args, **kwargs: (
                [
                    {
                        "chunk_id": "fat-1",
                        "file_name": "fat.xlsx",
                        "file_path": "excel/fat.xlsx",
                        "file_type": "xlsx",
                        "text": "FAT站位测试项目包含MES Station Check。",
                        "rerank_score": 0.9,
                        "semantic_score": 0.8,
                        "keyword_score": 1.0,
                        "fts_score": 1.0,
                    }
                ],
                {},
            )
            query_module.web_search = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("local answer must not search web"))
            local_first = query_module.query(db_path, "FAT站位有哪些测试项", use_web=True, allow_api=False)
            assert local_first["source_type"] == "knowledge_base"
            assert local_first["web_search_used"] is False
            assert local_first["async_update_scheduled"] is True
            assert local_first["index_status"] == "updating"
            assert "后台更新" in local_first["index_message"]
        finally:
            query_module.search = original_search
            query_module.trigger_async_update = original_trigger_async_update
            query_module.web_search = original_web_search
    check_embedding_cache()
    check_block_citations()
    check_structured_xlsx()
    check_query_terms()
    check_unknown_names()
    check_agent_loop()
    check_ocr_pipeline()
    check_xlsx_drawings()
    check_offline_translation_setup()
    check_process_answers()
    check_local_answer_layout()
    check_gate_synonyms()
    check_gate_asked_part()
    print("retrieval guardrails: ok")


def check_process_answers() -> None:
    """A process question shows supporting sections whole, each as its own list; other questions keep snippets."""
    assert query_module.process_question("工单数据获取流程") and query_module.process_question("单个测试项要经历哪些步骤")
    assert not query_module.process_question("客户类型有哪些")
    overview = "\n".join(["处理总览"] + [f"路径 {name}：读取步骤说明，这一步的输入、输出和异常处理都写在这里" for name in "ABCDEFGHIJ"])
    row = {"text": overview, "context_expanded": True}
    assert query_module.citation_for_retrieval_row(row, "数据获取流程").count("路径") == 10  # whole section
    assert len(query_module.citation_for_retrieval_row(row, "数据获取有哪些路径说明")) <= query_module.PROCESS_SECONDARY_BLOCK_MAX_CHARS
    assert len(query_module.citation_for_retrieval_row(row, "客户类型有哪些")) <= query_module.SECONDARY_BLOCK_MAX_CHARS + 20
    result = query_module.chinese_grounded_fallback(
        {
            "source_type": "knowledge_base",
            "sources": [
                {"source_id": 1, "file": "a.md", "citation": "流程一\n第一步\n第二步"},
                {"source_id": 2, "file": "b.md", "citation": overview},
                {"source_id": 3, "file": "c.md", "citation": "一句简短的补充说明"},
            ],
        }
    )
    answer = result["answer"]
    # The opening names what was found; each section is a group named by its title, keeping its list.
    assert answer.startswith("以下是本地知识库中「流程一」的内容："), answer
    assert "\n- 第一步\n- 第二步 [1]" in answer, answer
    assert "处理总览：\n- 路径 A：" in answer and "- 路径 J：" in answer, answer
    assert "补充：\n- 一句简短的补充说明 [3]" in answer, answer  # a short snippet goes under 补充


def check_gate_synonyms() -> None:
    """The local gate accepts evidence written in the documents' words (路径 for 目录, 并行 for
    同时执行) and ignores words that only shape the question, but still refuses when the asked
    attribute is missing."""
    from kb import citations, relevance

    units = relevance.query_content_units("样例缓存文件放在哪个目录")
    assert ("concept", "目录|路径|文件夹|directory|folder", 1.0) in units, units
    assert not any(term in {"放在", "件放"} for _, term, _ in units), units
    assert relevance.content_coverage(units, "样例缓存文件 | 路径 | 内容") > relevance.LOCAL_CONTENT_COVERAGE_THRESHOLD
    units = relevance.query_content_units("测试通过后让设备冷启动是哪个参数控制的")
    assert ("concept", "冷启动|coldboot|cold boot", 1.0) in units and not any("参数" in term for _, term, _ in units), units
    assert relevance.concept_variant_found("coldboot", "if (samplecoldbootflag) coldboot();", "")
    assert not relevance.concept_variant_found("dut", "dutyratio", "")  # short variants need a word boundary
    # A missing attribute still refuses: the subject alone covers only half of the question.
    units = relevance.query_content_units("SAMPLE01的售价是多少")
    assert relevance.content_coverage(units, "SAMPLE01 测试计划 封面") <= relevance.LOCAL_CONTENT_COVERAGE_THRESHOLD
    # The excerpt picks the line saying the asked idea in the documents' words.
    table = "| 字段 | 说明 |\n| CaseTitle | 测试项标题 |\n| GroupNo | 分组号，相同值的测试项会并行执行 |"
    assert "并行" in citations.citation_for_text(table, "怎么让两个测试项同时执行"), citations.citation_for_text(table, "怎么让两个测试项同时执行")


def check_gate_asked_part() -> None:
    """A name that a large share of the index mentions does not make a chunk evidence for whatever
    is asked about it, and in 「X支持Y吗」 the evidence must mention Y."""
    from kb import relevance

    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "knowledge.db"
        conn = sqlite3.connect(db_path)
        conn.executescript("CREATE TABLE chunks (text TEXT); CREATE TABLE metadata (key TEXT, value TEXT);")
        # 30 chunks: SAMPLEFW in 4 (broad, 13%), SampleWidget in 1 (specific, 3%), 测试 in most.
        texts = [f"SAMPLEFW 测试说明第 {index} 节" for index in range(4)] + [f"测试记录 {index}" for index in range(22)]
        texts += [f"其他说明 {index}" for index in range(3)] + ["SampleWidget 枚举：A、B"]
        conn.executemany("INSERT INTO chunks (text) VALUES (?)", [(text,) for text in texts])
        conn.execute("INSERT INTO metadata VALUES ('indexed_at', 't1')")
        conn.commit()
        conn.close()
        assert relevance.name_share("samplefw", db_path) >= relevance.BROAD_NAME_SHARE > relevance.name_share("samplewidget", db_path)
        assert relevance.term_share("测试", db_path) >= relevance.UBIQUITOUS_TERM_SHARE

        def row(text: str) -> dict:
            return {"text": text, "rerank_score": 0.7, "keyword_score": 1.0, "fts_score": 0.5}

        # A broad name plus a ubiquitous word is no evidence for the asked 仓库地址 / 覆盖率.
        assert not relevance.local_result_is_relevant([row("SAMPLEFW 测试源码说明，代码以源码为准")], "SAMPLEFW的代码仓库地址是什么", db_path)
        assert not relevance.local_result_is_relevant([row("SAMPLEFW 测试覆盖说明")], "SAMPLEFW的单元测试覆盖率是多少", db_path)
        assert relevance.local_result_is_relevant([row("SAMPLEFW 代码仓库地址：内部服务器")], "SAMPLEFW的代码仓库地址是什么", db_path)
        # A specific name still pins the question to its evidence.
        assert relevance.local_result_is_relevant([row("SampleWidget 枚举：A、B")], "SampleWidget平台枚举有哪些值", db_path)
        # 「X支持Y吗」 needs Y in the evidence.
        assert not relevance.local_result_is_relevant([row("样例箱模式：循环执行测试")], "样例箱模式支持双屏测试吗", db_path)
        assert relevance.local_result_is_relevant([row("样例箱模式支持并行测试，组号相同即可")], "样例箱模式支持并行测试吗", db_path)


def check_local_answer_layout() -> None:
    """Local answers turn the evidence's structure into readable lines: table rows, Excel records
    with their row number and empty fields dropped, numbered steps, and code kept as a code block."""
    from kb import answer_text

    table = "LevelType 枚举\n值 | 颜色 | 用途\nLOW | 绿色 | 普通提示\nNONE | — | 不显示"
    kind, title, lines = answer_text.evidence_layout(table)
    assert (kind, title) == ("section", "LevelType 枚举"), (kind, title)
    assert lines == ["- LOW — 颜色：绿色；用途：普通提示", "- NONE — 用途：不显示"], lines
    record = "样例清单 | #：7 | 类别：电源 | 内容：电压范围 | 状态：— | 负责人：甲组 | 负责人：乙组 | 方案：1.接通电源 2.读取电压 3.比较范围"
    kind, title, lines = answer_text.evidence_layout(record)
    assert (kind, title) == ("record", "样例清单"), (kind, title)
    assert lines == ["- 类别：电源", "- 内容：电压范围", "- 负责人：甲组，乙组", "", "方案", "- 1.接通电源", "- 2.读取电压", "- 3.比较范围"], lines
    assert answer_text.numbered_steps("电压 2.5V，版本 1.0") == []  # numbers are not steps
    code = "步骤 1：新建示例类\nusing Demo.Core;\npublic class Sample : SampleBase\nget { return value; }\n<Target>Release</Target>"
    kind, title, lines = answer_text.evidence_layout("新增示例\n" + code)
    assert lines[:2] == ["", "步骤 1：新建示例类"] and lines[2] == "```" and lines[-1] == "```", lines
    assert "public class Sample : SampleBase" in lines and "<Target>Release</Target>" in lines, lines
    # Polishing leaves code untouched: pipes, semicolons and commas are not punctuation there.
    polished = answer_text.polish_internal_answer_text("说明：\n```\nint a = b | c;\nf(x, y);\n```\n- 下一步，继续 [1]")
    assert "int a = b | c;\nf(x, y);" in polished and "- 下一步，继续 [1]" in polished, polished
    answer = answer_text.local_structured_answer([(record, 1), ("接通电源后读取电压。", 2)], "knowledge_base")
    assert answer.startswith("在本地知识库的「样例清单」中找到这条记录："), answer
    assert "#" not in answer and "补充：\n- 接通电源后读取电压。 [2]" in answer, answer

def check_offline_translation_setup() -> None:
    """Stanza pipelines started for translation reuse local resources instead of re-downloading them."""
    import sys
    import types

    import project_runtime

    calls: list[dict] = []
    fake_stanza = types.ModuleType("stanza")
    fake_pipeline_package = types.ModuleType("stanza.pipeline")
    fake_core = types.ModuleType("stanza.pipeline.core")
    fake_core.DownloadMethod = types.SimpleNamespace(REUSE_RESOURCES="reuse", DOWNLOAD_RESOURCES="download")
    fake_stanza.Pipeline = lambda *args, **kwargs: calls.append(kwargs) or "pipeline"
    saved = {name: sys.modules.get(name) for name in ("stanza", "stanza.pipeline", "stanza.pipeline.core")}
    sys.modules.update({"stanza": fake_stanza, "stanza.pipeline": fake_pipeline_package, "stanza.pipeline.core": fake_core})
    try:
        project_runtime.reuse_local_stanza_resources()
        project_runtime.reuse_local_stanza_resources()  # applying twice does not wrap twice
        assert fake_stanza.Pipeline(lang="en", processors="tokenize") == "pipeline"
        assert calls == [{"lang": "en", "processors": "tokenize", "download_method": "reuse"}], calls
        fake_stanza.Pipeline(lang="en", download_method="download")  # an explicit choice is kept
        assert calls[-1]["download_method"] == "download"
        # Code is never machine-translated (it comes back garbled); English prose is.
        assert query_module.looks_like_code("ShowStatus(_status). Ready = (_status == Status.OK).")
        assert query_module.looks_like_code("if (x > 0) { return y; }")
        assert not query_module.looks_like_code("The item is retried when RetryCount is greater than zero.")
        assert query_module.local_translate_to_chinese("Ready = (_status == Status.OK);") == ""
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

def check_xlsx_drawings() -> None:
    """Text boxes, charts and SmartArt are extracted with their sheet; only what cannot be read needs review."""
    main = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    pkg = "http://schemas.openxmlformats.org/package/2006/relationships"
    xdr = "http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing"
    a = "http://schemas.openxmlformats.org/drawingml/2006/main"
    c = "http://schemas.openxmlformats.org/drawingml/2006/chart"

    def text_body(*paragraphs: str) -> str:
        return "".join(f"<a:p><a:r><a:t>{text}</a:t></a:r></a:p>" for text in paragraphs)

    def points(*values: str) -> str:
        return "".join(f'<c:pt idx="{index}"><c:v>{value}</c:v></c:pt>' for index, value in enumerate(values))

    drawing = (
        f'<xdr:wsDr xmlns:xdr="{xdr}" xmlns:a="{a}" xmlns:c="{c}" xmlns:r="{rel}">'
        f"<xdr:twoCellAnchor><xdr:sp><xdr:txBody>{text_body('示例文本框', '第二段说明')}</xdr:txBody></xdr:sp></xdr:twoCellAnchor>"
        '<xdr:twoCellAnchor><xdr:sp><xdr:nvSpPr><xdr:cNvPr id="3" name="Object 1" hidden="1"/></xdr:nvSpPr></xdr:sp></xdr:twoCellAnchor>'
        '<xdr:twoCellAnchor><xdr:graphicFrame><a:graphic><a:graphicData><c:chart r:id="rId1"/></a:graphicData></a:graphic>'
        "</xdr:graphicFrame></xdr:twoCellAnchor></xdr:wsDr>"
    )
    chart = (
        f'<c:chartSpace xmlns:c="{c}" xmlns:a="{a}"><c:chart>'
        f"<c:title><c:tx><c:rich>{text_body('月度良率')}</c:rich></c:tx></c:title><c:plotArea><c:barChart><c:ser>"
        f"<c:tx><c:strRef><c:strCache>{points('良率')}</c:strCache></c:strRef></c:tx>"
        f"<c:cat><c:strRef><c:strCache>{points('1月', '2月')}</c:strCache></c:strRef></c:cat>"
        f"<c:val><c:numRef><c:numCache>{points('98', '97')}</c:numCache></c:numRef></c:val>"
        f"</c:ser></c:barChart><c:valAx><c:title><c:tx><c:rich>{text_body('百分比')}</c:rich></c:tx></c:title></c:valAx>"
        "</c:plotArea></c:chart></c:chartSpace>"
    )
    diagram = f'<dgm:dataModel xmlns:dgm="d" xmlns:a="{a}"><dgm:ptLst><dgm:pt><dgm:t>{text_body("步骤一", "步骤二")}</dgm:t></dgm:pt></dgm:ptLst></dgm:dataModel>'

    def workbook(path: Path, chart_xml: str = chart, attachment: bool = False) -> None:
        ole = f'<oleObjects><oleObject progId="Package" shapeId="1025" r:id="rId9"/></oleObjects>' if attachment else ""
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("xl/workbook.xml", f'<workbook xmlns="{main}" xmlns:r="{rel}"><sheets><sheet name="示例" sheetId="1" r:id="rId1"/></sheets></workbook>')
            archive.writestr("xl/_rels/workbook.xml.rels", f'<Relationships xmlns="{pkg}"><Relationship Id="rId1" Target="worksheets/sheet1.xml" Type="{rel}/worksheet"/></Relationships>')
            archive.writestr(
                "xl/worksheets/sheet1.xml",
                f'<worksheet xmlns="{main}" xmlns:r="{rel}"><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>表头</t></is></c></row></sheetData>{ole}</worksheet>',
            )
            archive.writestr("xl/worksheets/_rels/sheet1.xml.rels", f'<Relationships xmlns="{pkg}"><Relationship Id="rId2" Target="../drawings/drawing1.xml" Type="{rel}/drawing"/></Relationships>')
            archive.writestr("xl/drawings/drawing1.xml", drawing)
            archive.writestr(
                "xl/drawings/_rels/drawing1.xml.rels",
                f'<Relationships xmlns="{pkg}"><Relationship Id="rId1" Target="../charts/chart1.xml" Type="{rel}/chart"/>'
                f'<Relationship Id="rId3" Target="../diagrams/data1.xml" Type="{rel}/diagramData"/></Relationships>',
            )
            archive.writestr("xl/charts/chart1.xml", chart_xml)
            archive.writestr("xl/diagrams/data1.xml", diagram)
            if attachment:
                archive.writestr("xl/embeddings/oleObject1.bin", b"\xd0\xcf\x11\xe0")

    with tempfile.TemporaryDirectory() as temp_dir:
        readable = Path(temp_dir) / "drawings.xlsx"
        workbook(readable)
        text, report = ingest_module.read_xlsx_with_report(readable)
        assert "[工作表 示例 文本框] 示例文本框 第二段说明" in text, text
        assert "[工作表 示例 图表] 标题：月度良率；数值轴：百分比；良率：1月=98，2月=97" in text, text
        assert "[工作表 示例 SmartArt] 步骤一 步骤二" in text, text
        assert "drawing_text" in report["method"] and not report["requires_review"] and not report["warnings"], report
        assert "示例文本框" in report[ingest_module.CHUNK_TEXT_REPORT_KEY]  # chunks get the drawing text too
        # An attached file and an unreadable chart are what still needs a person to look.
        attached = Path(temp_dir) / "attached.xlsx"
        workbook(attached, attachment=True)
        _, report = ingest_module.read_xlsx_with_report(attached)
        assert report["requires_review"] and "1 embedded object(s) not extracted (attached file on sheet 示例)" in report["warnings"][0], report
        broken = Path(temp_dir) / "broken.xlsx"
        workbook(broken, chart_xml="<c:chartSpace")
        text, report = ingest_module.read_xlsx_with_report(broken)
        assert report["requires_review"] and "1 chart/drawing part(s) could not be read" in report["warnings"][0], report
        assert "示例文本框" in text  # the rest of the drawing is still extracted

def check_ocr_pipeline() -> None:
    """Scanned PDFs render without Poppler, and OCR keeps the page-segmentation mode Tesseract is surest of."""
    import shutil

    original_which = ocr_module.shutil.which
    ocr_module.shutil.which = lambda name, *args, **kwargs: None if name == "pdftoppm" else original_which(name, *args, **kwargs)
    try:
        renderer = ingest_module.pdf_page_renderer()
        try:
            import pypdfium2  # noqa: F401
        except ImportError:
            assert renderer is None
        else:
            assert renderer == "pypdfium2"
            sample = Path(__file__).resolve().parents[1] / "test_materials" / "ocr" / "scan_english_basic.pdf"
            if sample.exists():
                with tempfile.TemporaryDirectory() as temp_dir:
                    pages = ingest_module.render_pdf_pages_for_ocr(sample, Path(temp_dir))
                    assert pages and all(page.suffix == ".png" and page.stat().st_size > 0 for page in pages)
    finally:
        ocr_module.shutil.which = original_which
    assert shutil.which is original_which  # the patch did not leak
    with tempfile.TemporaryDirectory() as temp_dir:
        tsv = Path(temp_dir) / "page.tsv"
        header = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext"
        tsv.write_text(
            "\n".join(
                [
                    header,
                    "1\t1\t0\t0\t0\t0\t0\t0\t100\t100\t-1\t",  # structural row: no text, no confidence
                    "5\t1\t1\t1\t1\t1\t0\t0\t10\t10\t90\tabcd",
                    "5\t1\t1\t1\t1\t2\t0\t0\t10\t10\t60\tab",
                ]
            ),
            encoding="utf-8",
        )
        assert abs(ingest_module.tsv_mean_confidence(tsv) - 80.0) < 1e-6  # weighted by characters: (90*4 + 60*2) / 6
        assert ingest_module.tsv_mean_confidence(Path(temp_dir) / "missing.tsv") == -1.0

def check_agent_loop() -> None:
    """The agent answers from cited evidence, searches again when asked, and never ends a last round searching."""
    import agentic_retrieval as agent

    corpus = {
        "开机流程": [{"chunk_id": "a", "file_name": "guide.md", "evidence": "开机流程：先自检，再加载配置"}],
        "BootUp": [{"chunk_id": "b", "file_name": "guide.md", "evidence": "BootUp() 依次调用 HardwareCheck() 和 LoadSettings()"}],
    }

    def retrieve(query_text: str, limit: int) -> list[dict]:
        return [dict(row) for row in corpus.get(query_text, [])]

    def scripted(*replies: dict) -> tuple:
        pending = list(replies)
        payloads: list[dict] = []

        def complete(messages: list[dict]) -> dict:
            payloads.append(json.loads(messages[-1]["content"]))
            return {"content": json.dumps(pending.pop(0), ensure_ascii=False), "usage": {"total_tokens": 10}}

        return complete, payloads

    complete, _ = scripted({"decision": "answer", "answer": "先自检再加载配置。\n- 自检 [#1]", "evidence_ids": [1]})
    outcome = agent.run_agent("开机流程", retrieve, complete)
    assert outcome.decision == "answer" and outcome.cited_ids == [1] and outcome.rounds == 1
    assert outcome.usage == {"total_tokens": 10}
    # A follow-up search adds evidence; the second round is the last one and is told so.
    complete, payloads = scripted(
        {"decision": "search", "queries": ["BootUp", "开机流程"]},
        {"decision": "answer", "answer": "开机时调用 BootUp()。\n- 依次自检和加载配置 [#2]", "evidence_ids": [2]},
    )
    outcome = agent.run_agent("开机流程", retrieve, complete)
    assert outcome.queries == ["BootUp"] and [item["id"] for item in outcome.evidence] == [1, 2]
    assert [payload["final_round"] for payload in payloads] == [False, True]
    # Still searching in the last round means the evidence is not there.
    complete, _ = scripted({"decision": "search", "queries": ["x"]}, {"decision": "search", "queries": ["y"]})
    assert agent.run_agent("开机流程", retrieve, complete).decision == "refuse"
    # Replies that cannot be trusted raise, so the caller answers locally instead.
    for bad in ("not json", json.dumps({"decision": "answer", "answer": "没有引用"}), json.dumps({"decision": "answer", "answer": "见 [#9]"})):
        try:
            agent.run_agent("开机流程", retrieve, lambda messages, bad=bad: {"content": bad})
        except agent.AgentError:
            pass
        else:
            raise AssertionError(bad)
    # An empty reply (DeepSeek's JSON mode occasionally returns one) or a server error gets one retry.
    complete, _ = scripted({"decision": "answer", "answer": "先自检。\n- 自检 [#1]"})
    replies = iter([{"content": ""}, None])
    flaky = lambda messages: next(replies) or complete(messages)  # noqa: E731
    assert agent.run_agent("开机流程", retrieve, flaky).decision == "answer"
    busy = iter([RuntimeError("HTTP 503: busy"), None])

    def overloaded(messages: list[dict]) -> dict:
        error = next(busy)
        if error:
            raise error
        return complete(messages)

    complete, _ = scripted({"decision": "refuse", "reason": "没有"})
    assert agent.run_agent("开机流程", retrieve, overloaded).decision == "refuse"
    for failure in ([{"content": ""}, {"content": ""}], [RuntimeError("HTTP 401: bad key")]):
        pending = iter(failure)

        def failing(messages: list[dict], pending=pending) -> dict:
            item = next(pending)
            if isinstance(item, Exception):
                raise item
            return item

        try:
            agent.run_agent("开机流程", retrieve, failing)
        except (agent.AgentError, RuntimeError):
            pass
        else:
            raise AssertionError(failure)
    assert agent.cited_ids("a [#3] b [#1, #3] c [#2]") == [3, 1, 2]
    assert agent.renumber_citations("a [#3] b [#1, #3] c [#2]", [3, 1, 2]) == "a [1] b [2][1] c [3]"
    # Bracketed data from the documents is never a citation.
    assert agent.cited_ids("门限 [10,20] 和 [-3,-7]，见 args[0] [#2]") == [2]
    assert agent.renumber_citations("门限 [10,20] [#2]", [2]) == "门限 [10,20] [1]"
    evidence: list[dict] = []
    agent.add_evidence(evidence, [{"evidence": "整段：甲乙丙"}, {"evidence": "甲乙丙"}, {"evidence": "丁"}])
    assert [item["text"] for item in evidence] == ["整段：甲乙丙", "丁"]  # contained evidence is shown once
    # Sources follow citation order and the answer is renumbered to match.
    outcome = agent.AgentOutcome(
        decision="answer",
        answer="结论 [#2]，另见 [#1]",
        reason="",
        evidence=[
            {"id": 1, "file": "a.md", "text": "甲", "row": {"file_name": "a.md", "chunk_id": "c1"}},
            {"id": 2, "file": "b.md", "text": "乙", "row": {"file_name": "b.md", "chunk_id": "c2"}},
        ],
        cited_ids=[2, 1],
        rounds=1,
        queries=[],
    )
    result = query_module.agent_answer_result("问", outcome, {}, {}, False)
    assert result["answer"] == "结论 [1]，另见 [2]" and [source["chunk_id"] for source in result["sources"]] == ["c2", "c1"]
    api_config = {"LKA_USE_API": "true", "LKA_API_KEY": "test", "LKA_API_MODEL": "m"}
    assert query_module.agent_enabled(api_config)
    assert not query_module.agent_enabled({**api_config, "LKA_AGENT_ENABLED": "false"})
    assert not query_module.agent_enabled({**api_config, "LKA_API_KEY": ""})


def check_unknown_names() -> None:
    """A name or name + version the index never mentions makes the gate refuse."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "knowledge.db"
        conn = sqlite3.connect(db_path)
        conn.executescript(
            """
            CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE chunks(chunk_id TEXT, text TEXT);
            INSERT INTO metadata VALUES ('indexed_at', 'first');
            INSERT INTO chunks VALUES ('a', 'Widget 2.0 接口列表：查询、上传'), ('b', 'LogLevel 级别：Info、Warn、Error'),
                                      ('c', '型号 AB123 的接口说明');
            """
        )
        conn.commit()
        assert query_module.unknown_names("Widget 3.0的接口有哪些", db_path) == ["Widget 3.0"]
        assert query_module.unknown_names("Widget3.0的接口有哪些", db_path) == ["Widget 3.0"]
        assert query_module.unknown_names("Widget 2.0的接口有哪些", db_path) == []
        assert query_module.unknown_names("怎么迁移到.NET 6", db_path) == [".NET 6"]
        assert query_module.unknown_names("LogLevel有没有FATAL级别", db_path) == ["FATAL"]
        assert query_module.unknown_names("AB123的接口有哪些", db_path) == []  # a model code is not word + version
        assert query_module.unknown_names("What is FATAL", db_path) == []  # English words are not names
        rows = [{"rerank_score": 0.7, "keyword_score": 1.0, "fts_score": 0.5, "text": "Widget 2.0 接口列表：查询、上传"}]
        assert query_module.local_result_is_relevant(rows, "Widget 3.0的接口有哪些")
        assert not query_module.local_result_is_relevant(rows, "Widget 3.0的接口有哪些", db_path)
        assert query_module.local_result_is_relevant(rows, "Widget 2.0的接口有哪些", db_path)
        # A rebuilt index is read again.
        conn.execute("INSERT INTO chunks VALUES ('d', 'Widget 3.0 新增接口：下载')")
        conn.execute("UPDATE metadata SET value = 'second' WHERE key = 'indexed_at'")
        conn.commit()
        conn.close()
        assert query_module.unknown_names("Widget 3.0的接口有哪些", db_path) == []


def check_query_terms() -> None:
    """Long questions keep their short terms, and FTS only receives terms it can match."""
    terms = query_module.candidate_terms("风扇测试的低中高转速门限分别是多少")
    assert {"风扇", "转速", "门限"} <= set(terms), terms
    assert not {"是多", "多少", "分别"} & set(terms), terms  # question words are not evidence
    assert query_module.candidate_terms("主板USB 4.0读写速度门限是多少")[0] == "usb"
    fts_terms = re.findall(r'"([^"]+)"', query_module.fts_query("测试中途重启后，已经通过的测试项还要重新测吗"))
    assert fts_terms and all(len(term) <= 4 for term in fts_terms), fts_terms  # index holds 2-4 char n-grams
    assert "重启" in fts_terms


def check_structured_xlsx() -> None:
    """Excel chunks: merged cells filled, every value titled, no empty rows or padding columns."""

    def cell(ref: str, value: str) -> str:
        return f'<c r="{ref}" t="inlineStr"><is><t>{value}</t></is></c>'

    table = (
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'
        f'<row r="1">{cell("A1", "#")}{cell("B1", "测试项目")}{cell("D1", "测试门限")}{cell("E1", "单位")}</row>'
        f'<row r="2">{cell("B2", "测试项")}{cell("C2", "测试子项")}</row>'
        f'<row r="3">{cell("A3", "1")}{cell("B3", "风扇测试")}{cell("C3", "低速")}{cell("D3", "1000±100")}{cell("E3", "rpm")}</row>'
        f'<row r="4">{cell("A4", "2")}{cell("C4", "高速")}{cell("D4", "3000±100")}{cell("E4", "rpm")}</row>'
        '<row r="5"></row>'
        f'<row r="6">{cell("Z6", "备注X")}</row>'
        f'<row r="7">{cell("A7", "3")}{cell("B7", "电压测试")}{cell("C7", "Voltage")}{cell("D7", "[1.0,2.0]")}{cell("E7", "V")}</row>'
        f'<row r="8">{cell("A8", "4")}{cell("C8", "Resolution")}{cell("D8", "1080P")}{cell("E8", "NA")}</row>'
        '</sheetData><mergeCells count="6"><mergeCell ref="A1:A2"/><mergeCell ref="B1:C1"/>'
        '<mergeCell ref="D1:D2"/><mergeCell ref="E1:E2"/><mergeCell ref="B3:B4"/><mergeCell ref="B7:B8"/></mergeCells></worksheet>'
    )
    cover = (
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'
        f'<row r="1">{cell("A1", "文件名称")}{cell("B1", "示例装备测试计划")}</row>'
        f'<row r="2">{cell("A2", "责任部门")}{cell("B2", "示例部门-测试组")}{cell("C2", "创建时间")}{cell("D2", "2025-11-18")}</row>'
        "</sheetData></worksheet>"
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        xlsx_path = Path(temp_dir) / "plan.xlsx"
        with zipfile.ZipFile(xlsx_path, "w") as archive:
            archive.writestr(
                "xl/workbook.xml",
                '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>'
                '<sheet name="门限" sheetId="1" r:id="rId1"/><sheet name="封面" sheetId="2" r:id="rId2"/></sheets></workbook>',
            )
            archive.writestr(
                "xl/_rels/workbook.xml.rels",
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                '<Relationship Id="rId1" Target="worksheets/sheet1.xml" Type="worksheet"/>'
                '<Relationship Id="rId2" Target="worksheets/sheet2.xml" Type="worksheet"/></Relationships>',
            )
            archive.writestr("xl/worksheets/sheet1.xml", table)
            archive.writestr("xl/worksheets/sheet2.xml", cover)
        records = ingest_module.read_xlsx_records(xlsx_path)
        # Rows under one merged item form one block; the overview lists every group's items.
        assert records.split("\n\n") == [
            "## 工作表 门限 / 风扇测试",
            "门限 | #：1 | 测试项目/测试项：风扇测试 | 测试项目/测试子项：低速 | 测试门限：1000±100 | 单位：rpm",
            "门限 | #：2 | 测试项目/测试项：风扇测试 | 测试项目/测试子项：高速 | 测试门限：3000±100 | 单位：rpm",
            "## 工作表 门限",
            "门限 | 备注X",
            "## 工作表 门限 / 电压测试",
            "门限 | #：3 | 测试项目/测试项：电压测试 | 测试项目/测试子项：Voltage | 测试门限：[1.0,2.0] | 单位：V",
            "门限 | #：4 | 测试项目/测试项：电压测试 | 测试项目/测试子项：Resolution | 测试门限：1080P | 单位：NA",
            "## 工作表 门限 概览",
            "门限 概览 | 风扇测试：\n低速 1000±100\n高速 3000±100",
            "门限 概览 | 电压测试：\nVoltage [1.0,2.0]\nResolution 1080P",
            "## 工作表 封面",
            "封面 | 文件名称 | 示例装备测试计划",
            "封面 | 责任部门 | 示例部门-测试组 | 创建时间 | 2025-11-18",
        ], records
        legacy, report = ingest_module.read_xlsx_with_report(xlsx_path)
        # documents.text keeps the old positional layout (padding columns included) for the table parsers.
        assert legacy.startswith("工作表 门限\n# | 测试项目 |") and legacy.splitlines()[-6].count(" | ") == 25, legacy
        assert report[ingest_module.CHUNK_TEXT_REPORT_KEY] == records

    # A list question prefers the overview chunk that holds every item over one row of the group.
    candidates = [
        {"chunk_id": "row", "text": "表 | 组一 | 测试项：A", "semantic_score": 0.70},
        {"chunk_id": "overview", "text": "表 概览 | 组一：\n测试项 A\n测试项 B", "semantic_score": 0.66},
    ]
    assert query_module.enumeration_or_list_question("SFT主板功能测试包含哪些测试项")
    assert [row["chunk_id"] for row in query_module.rerank(candidates, "组一包含哪些测试项", 2, {})] == ["overview", "row"]
    assert [row["chunk_id"] for row in query_module.rerank(candidates, "组一的测试门限", 2, {})] == ["row", "overview"]


def check_block_citations() -> None:
    """Top evidence is cited as a whole section or table, not as its two best-scoring sentences."""
    for question in ("FAI是什么的缩写", "FAI的全称是什么", "FAI是哪个词的简称", "缩写FAI"):
        assert query_module.abbreviation_query_term(question) == "FAI", question
    assert query_module.abbreviation_query_term("试运行的英文缩写是什么") is None

    section = (
        "return result; }\n---\n### 3.1 示例流程\n```\nRunDemo(mode=true)\n"
        "  ├── 1. 准备环境\n  │\n  ├── 2. 检查电源（若 CheckPower=true）\n  └── 3. 网络连接\n```"
    )
    cited = query_module.block_citation(section, "示例流程依次做了哪些事", 1500)
    assert cited.splitlines()[0] == "3.1 示例流程", cited  # lead-in from the previous section dropped
    assert "2. 检查电源（若 CheckPower=true）" in cited and "3. 网络连接" in cited
    assert "```" not in cited and "return result" not in cited and "│" not in cited.replace("├──", "")

    table = "| 对比项 | 方案 A | 方案 B |\n|---|---|---|\n| 接口协议 | HTTP JSON | 本地进程 |\n| 请求次数 | 1 次 | 3-5 次 |\n| | | |"
    cited = query_module.block_citation(table, "方案A和方案B有什么区别", 1500)
    assert cited.splitlines() == ["对比项 | 方案 A | 方案 B", "接口协议 | HTTP JSON | 本地进程", "请求次数 | 1 次 | 3-5 次"]

    long_rows = "\n".join(f"| {index} | 测试项{index} | " + "很长的测试方案描述" * 30 + " |" for index in range(1, 13))
    cited = query_module.block_citation(long_rows, "包含哪些测试项", 1500)
    assert all(f"测试项{index}" in cited for index in range(1, 13)) and len(cited) <= 1500  # rows shortened, none dropped

    steps = "## 十一、添加新测试项\n" + "".join(f"### 步骤 {index}：第{index}步\n" + "代码行内容\n" * 40 for index in range(1, 6))
    cited = query_module.block_citation(steps, "如何添加一个新测试项", 600)
    assert len(cited) <= 600 and cited.endswith("步骤 5：第5步"), cited[-80:]
    assert "本节其余部分：" in cited and "本节其余部分" not in query_module.block_citation(steps, "x", 300, outline=False)

    kept = query_module.local_translate_to_chinese("场景 6：服务连接失败\n1. 检查 **EnableRemote**\n2. 检查 ServerAddress")
    assert kept.splitlines() == ["场景 6：服务连接失败", "1. 检查 EnableRemote", "2. 检查 ServerAddress"]

    # A short quote already contained in the whole-section quote must not take a citation slot.
    deduped = query_module.grounded_citations(
        {
            "sources": [
                {"source_id": 1, "citation": "参数甲电压，[1.0,2.0]，V\n示例测试，参数乙电压，[3.0,4.0]，V\n参数丙电压，[5.0,6.0]，V"},
                {"source_id": 2, "citation": "示例测试，参数乙电压，[3.0,4.0]，V"},
                {"source_id": 3, "citation": "示例测试，Voltage，[7.0,8.0]，V"},
            ]
        }
    )
    assert [item["source_id"] for item in deduped] == [1, 3]


def check_embedding_cache() -> None:
    """Unchanged chunk texts are never re-embedded; vectors never leak across models."""
    with tempfile.TemporaryDirectory() as temp_dir:
        vector_dir = Path(temp_dir)
        cache = vector_store.EmbeddingCache("test|model", vector_dir)
        cache.put_many([("a", [0.5, 0.25]), ("b", [1.0, 0.0])])
        assert cache.get_many(["a", "missing"]) == {"a": [0.5, 0.25]}
        assert cache.prune(["a"]) == 1 and cache.count() == 1
        cache.close()
        other_model = vector_store.EmbeddingCache("other|model", vector_dir)
        assert other_model.get_many(["a"]) == {}
        other_model.close()

        calls: list[list[str]] = []

        def fake_embed(texts: list[str], config: dict | None = None, **kwargs: object) -> dict:
            calls.append(list(texts))
            return {"ok": True, "vectors": [[float(len(text)), 1.0] for text in texts]}

        original_embed = model_capabilities.embed_texts
        original_vector_dir = vector_store.VECTOR_DIR
        try:
            model_capabilities.embed_texts = fake_embed
            vector_store.VECTOR_DIR = vector_dir / "index"
            embedding = {"provider": "openai-compatible", "model": "fake", "base_url": "", "requested_dimensions": 2}
            first = ingest_module.embed_records_with_cache([{"text": "alpha"}, {"text": "beta"}, {"text": "alpha"}], embedding, {})
            assert first["ok"] and first["embedded_chunks"] == 3 and first["reused_chunks"] == 0
            assert calls == [["alpha", "beta"]], calls  # a repeated chunk text is embedded once
            second = ingest_module.embed_records_with_cache([{"text": "alpha"}, {"text": "gammas"}], embedding, {})
            assert calls[-1] == ["gammas"], calls  # only the new chunk reaches the model
            assert second["embedded_chunks"] == 1 and second["reused_chunks"] == 1
            assert second["vectors"] == [[5.0, 1.0], [6.0, 1.0]]
            third = ingest_module.embed_records_with_cache([{"text": "alpha"}, {"text": "gammas"}], embedding, {})
            assert len(calls) == 2 and third["reused_chunks"] == 2  # fully cached: no model call at all
            other = ingest_module.embed_records_with_cache([{"text": "alpha"}], {**embedding, "model": "other"}, {})
            assert calls[-1] == ["alpha"] and other["embedded_chunks"] == 1  # another model never reuses vectors

            if vector_store.faiss_dependency_status()["available"]:
                index_dir = vector_dir / "faiss"
                records = [
                    {"chunk_id": f"c{index}", "doc_id": "d", "file_name": "f.md", "file_path": "f.md", "file_type": "md", "chunk_index": index}
                    for index in range(2)
                ]
                vector_store.build_vector_index(records, [[1.0, 0.0], [0.0, 1.0]], {}, vector_dir=index_dir)
                first_load = vector_store.load_index(index_dir)
                assert vector_store.load_index(index_dir)[0] is first_load[0]  # served from memory
                assert vector_store.search_vector_index([1.0, 0.0], 1, vector_dir=index_dir)["rows"][0]["chunk_id"] == "c0"
                vector_store.build_vector_index(records, [[0.0, 1.0], [1.0, 0.0]], {}, vector_dir=index_dir)
                # The rewrite can land in the same timestamp tick with the same file sizes; give the files
                # a later time, as any real rebuild has, so the check does not depend on timing.
                for path in index_dir.iterdir():
                    stat = path.stat()
                    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
                assert vector_store.load_index(index_dir)[0] is not first_load[0]  # rewritten files reload
                assert vector_store.search_vector_index([1.0, 0.0], 1, vector_dir=index_dir)["rows"][0]["chunk_id"] == "c1"

                vector_store.VECTOR_DIR = index_dir
                conn = sqlite3.connect(":memory:")
                conn.execute("CREATE TABLE chunks(chunk_id TEXT, text TEXT)")
                conn.executemany("INSERT INTO chunks VALUES (?, ?)", [("c0", "zero"), ("c1", "one")])
                metadata = {"model": "seeded", "provider": "openai-compatible", "indexed_at": "t1", "chunk_count": 2}
                seeded_embedding = {**embedding, "model": "seeded"}
                assert ingest_module.seed_embedding_cache_from_index(conn, seeded_embedding, "t0", metadata) == 0  # stale index
                assert ingest_module.seed_embedding_cache_from_index(conn, seeded_embedding, "t1", metadata) == 2
                calls.clear()
                reused = ingest_module.embed_records_with_cache([{"text": "zero"}, {"text": "one"}], seeded_embedding, {})
                assert calls == [] and reused["vectors"] == [[0.0, 1.0], [1.0, 0.0]]
                conn.close()
        finally:
            model_capabilities.embed_texts = original_embed
            vector_store.VECTOR_DIR = original_vector_dir


if __name__ == "__main__":
    main()
