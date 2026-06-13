#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline regression checks for retrieval relevance guardrails."""

from __future__ import annotations

from pathlib import Path
import sqlite3
import tempfile
import zipfile

import query_knowledge_base as query_module
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
    assert web_query(question) == "量子纠缠"
    assert web_query("acwing是什么") == "acwing"
    assert web_query("雷军是谁") == "雷军"
    assert web_query("雷军是什么") == "雷军"
    polished = polish_internal_answer_text(
        "根据本地知识库，├── STPM.sln / STPM.exe.config | USB2.0/USB3.2 \\ ProgramData ![Image 1](x) [1]。"
    )
    assert "/" not in polished and "\\" not in polished and "|" not in polished
    assert "├" not in polished and "Image" not in polished
    assert "STPM.sln，STPM.exe.config" in polished
    assert "[1]" in polished
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
    assert query_module.cache_key("本地缓存", 5, False) == query_module.cache_key("本地缓存逻辑", 5, False)
    assert query_module.station_test_item_question("FFT站位测试")
    assert query_module.canonical_question_for_cache("FFT站位测试") == "station:fft:items"
    assert query_module.canonical_question_for_cache("FFT站位测试项") == "station:fft:items"
    assert query_module.abbreviation_query_term("PR") == "PR"
    assert query_module.abbreviation_query_term("PR是什么") == "PR"
    assert query_module.meaningful_query_terms("GW") == ["gw"]
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
    original_fetch_public_webpage_text = query_module.fetch_public_webpage_text
    try:
        query_module.fetch_public_webpage_text = lambda *args, **kwargs: (
            "<main><p>量子纠缠是量子力学中的一种现象，多个粒子的状态之间存在关联。</p></main>",
            "https://example.com/entanglement",
            "text/html",
        )
        researched = query_module.research_web_source("量子纠缠是什么", quantum_sources[0])
        assert researched["page_fetch_status"] == "ok"
        assert researched["content_source"] == "webpage"
        assert "多个粒子的状态之间存在关联" in researched["citation"]
    finally:
        query_module.fetch_public_webpage_text = original_fetch_public_webpage_text
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
            query_module.trigger_async_update = lambda *args, **kwargs: False
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
        finally:
            query_module.search = original_search
            query_module.trigger_async_update = original_trigger_async_update
            query_module.web_search = original_web_search
    print("retrieval guardrails: ok")


if __name__ == "__main__":
    main()
