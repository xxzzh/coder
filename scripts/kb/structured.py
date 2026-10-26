"""Structured answers read straight from tables: abbreviation dictionary, station test items, Excel fields and FFT lists."""

from __future__ import annotations

from difflib import SequenceMatcher
from html import unescape
import re
import sqlite3
from pathlib import Path
from typing import Any

from kb.terms import abbreviation_query_term, fft_manual_function_question, requested_station_name, station_question_includes_schemes, station_test_item_question


def dictionary_entry_rows(text: str, abbreviation: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for raw_line in text.splitlines():
        if "|" not in raw_line:
            continue
        columns = [structured_cell_text(column) for column in raw_line.split("|")]
        if len(columns) < 12 or columns[2].upper() != abbreviation.upper():
            continue
        english = columns[4]
        chinese = columns[11]
        if not english and not chinese:
            continue
        key = (columns[2].upper(), english.lower(), chinese)
        if key in seen:
            continue
        seen.add(key)
        records.append({"abbr": columns[2].upper(), "english": english, "chinese": chinese, "line": raw_line})
    return records


def structured_abbreviation_result(
    db_path: Path,
    question: str,
    metadata: dict[str, str] | None = None,
    refresh_scheduled: bool = False,
) -> dict[str, Any] | None:
    abbreviation = abbreviation_query_term(question)
    if not abbreviation:
        return None
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT doc_id, file_name, file_path, file_type, text
            FROM documents
            WHERE file_type = 'xlsx'
              AND text LIKE '%英文缩写%'
              AND text LIKE ?
            ORDER BY CASE WHEN file_name LIKE '%Dictionary%' OR file_name LIKE '%缩写%' THEN 0 ELSE 1 END, file_path
            """,
            (f"%| {abbreviation} |%",),
        ).fetchall()
    except sqlite3.DatabaseError:
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass

    best_row: sqlite3.Row | None = None
    best_records: list[dict[str, str]] = []
    for row in rows:
        records = dictionary_entry_rows(str(row["text"]), abbreviation)
        if len(records) > len(best_records):
            best_row = row
            best_records = records
    if not best_row or not best_records:
        return None

    lines = [
        f"- {record['english']}：{record['chinese']}" if record["english"] and record["chinese"]
        else f"- {record['english'] or record['chinese']}"
        for record in best_records
    ]
    answer = f"根据本地知识库，{abbreviation} 有以下含义：\n" + "\n".join(lines) + "\n[1]"
    excerpt = "\n".join(record["line"] for record in best_records)
    excerpt = structured_cell_text(excerpt, max_len=800)
    metadata = metadata or {}
    source = {
        "source_id": 1,
        "file": best_row["file_name"],
        "file_path": best_row["file_path"],
        "file_type": best_row["file_type"],
        "chunk_id": f"{best_row['doc_id']}:structured:abbreviation:{abbreviation.lower()}",
        "excerpt": excerpt,
        "citation": excerpt,
    }
    return {
        "answer": answer,
        "answer_mode": "local_structured_dictionary_summary",
        "source_type": "knowledge_base",
        "sources": [source],
        "citations": [{"source_id": 1, "file": best_row["file_name"], "chunk_id": source["chunk_id"], "quote": excerpt}],
        "need_web_search": False,
        "web_search_used": False,
        "index_updated_at": metadata.get("indexed_at"),
        "embedding_model": metadata.get("embedding_model"),
        "chunk_strategy": metadata.get("chunk_strategy"),
        "retrieval_mode": "local_structured_dictionary",
        "async_update_scheduled": refresh_scheduled,
        "cache_hit": False,
    }


def structured_cell_text(text: str, max_len: int | None = None) -> str:
    text = re.sub(r"\s+", " ", unescape(text)).strip()
    if max_len is not None and len(text) > max_len:
        text = text[: max_len - 1].rstrip() + "…"
    return text


def extract_station_table_items(text: str, station_name: str) -> list[str]:
    return [record["item"] for record in extract_station_table_records(text, station_name)]


def _station_record(parent: str, child: str, scheme: str = "") -> dict[str, str] | None:
    parent = structured_cell_text(parent)
    child = structured_cell_text(child)
    if not parent and not child:
        return None
    item = parent if not child or parent == child else f"{parent} / {child}" if parent else child
    return {"item": item, "scheme": structured_cell_text(scheme, max_len=1200)}


def _explicit_station_table_records(text: str, station_name: str) -> tuple[list[dict[str, str]], str]:
    candidates: list[tuple[list[dict[str, str]], str]] = []
    for section in re.split(r"(?=^工作表 )", text, flags=re.MULTILINE):
        records: list[dict[str, str]] = []
        current_parent = ""
        in_station = False
        for line in section.splitlines():
            if "|" not in line:
                continue
            columns = [column.strip() for column in line.split("|")]
            if not columns:
                continue
            if re.fullmatch(r"\d+", columns[0]):
                columns = columns[1:]
            elif not columns[0] and len(columns) >= 9:
                columns = columns[1:]
            else:
                continue
            if len(columns) < 3:
                continue
            row_station = columns[0]
            heading_match = re.search(r"工序名称[：:]\s*([A-Za-z][A-Za-z0-9_-]{1,30})", row_station, flags=re.IGNORECASE)
            if heading_match:
                row_station = heading_match.group(1)
            if row_station:
                if row_station.lower() == station_name.lower():
                    in_station = True
                    current_parent = ""
                elif in_station:
                    break
                else:
                    continue
            if not in_station:
                continue
            if columns[1]:
                current_parent = columns[1]
            record = _station_record(current_parent, columns[2], columns[7] if len(columns) > 7 else "")
            if record and not any(existing["item"] == record["item"] for existing in records):
                records.append(record)
        if records:
            candidates.append((records, section))
    return max(candidates, key=lambda candidate: len(candidate[0]), default=([], ""))


def _heading_station_table_records(text: str, station_name: str) -> tuple[list[dict[str, str]], str]:
    heading = re.compile(rf"工序名称[：:]\s*{re.escape(station_name)}(?:\s|$)", flags=re.IGNORECASE)
    match = heading.search(text)
    if not match:
        return [], ""
    end_candidates = [
        position
        for position in (
            text.find("\n工作表 ", match.end()),
            text.find("\n工序名称：", match.end()),
            text.find("\n工序名称:", match.end()),
        )
        if position >= 0
    ]
    section = text[match.start() : min(end_candidates) if end_candidates else len(text)]
    ignored = {
        "NA",
        "LC",
        "MES",
        "Acer",
        "厂商",
        "程序",
        "待导入",
        "程序&MES",
        "程序&人工",
        "STPM.exe",
    }
    blocks: list[str] = []
    current_lines: list[str] = []
    for line in section.splitlines():
        is_row_start = bool(re.match(r"^\s*(?:\d+\s*)?\|", line) or line.startswith("岗位资源："))
        if is_row_start and current_lines:
            blocks.append("\n".join(current_lines))
            current_lines = []
        if is_row_start or current_lines:
            current_lines.append(line)
    if current_lines:
        blocks.append("\n".join(current_lines))

    records: list[dict[str, str]] = []
    for block in blocks:
        columns = [column.strip() for column in block.split("|")]
        if columns and re.fullmatch(r"\d+", columns[0]):
            columns = columns[1:]
        labels: list[str] = []
        for column in columns[:4]:
            column = structured_cell_text(re.sub(r"^岗位资源：\s*", "", column))
            if (
                not column
                or column in ignored
                or column.lower().endswith((".exe", ".dll", ".bat"))
                or re.match(r"^\d+[.、]", column)
            ):
                continue
            if column not in labels:
                labels.append(column)
        label = " / ".join(labels)
        if not label:
            continue

        scheme = ""
        for column in columns[4:]:
            column = column.strip()
            if (
                not column
                or column in ignored
                or column.lower().endswith((".exe", ".dll", ".bat"))
                or column in {"Y", "N", "PASS", "FAIL", "Tracking", "CLOSE"}
            ):
                continue
            if len(column) >= 12 or "\n" in column or re.search(r"^\s*\d+[.、]", column):
                scheme = structured_cell_text(column, max_len=1200)
                break
        if not any(record["item"] == label for record in records):
            records.append({"item": label, "scheme": scheme})
    return records, section


def extract_station_table_records_with_excerpt(text: str, station_name: str) -> tuple[list[dict[str, str]], str]:
    records, excerpt = _explicit_station_table_records(text, station_name)
    if records:
        return records, excerpt
    return _heading_station_table_records(text, station_name)


def extract_station_table_records(text: str, station_name: str) -> list[dict[str, str]]:
    records, _ = extract_station_table_records_with_excerpt(text, station_name)
    return records


def structured_station_result(
    db_path: Path,
    question: str,
    metadata: dict[str, str] | None = None,
    refresh_scheduled: bool = False,
) -> dict[str, Any] | None:
    station_name = requested_station_name(question)
    if not station_name or not station_test_item_question(question):
        return None
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT doc_id, file_name, file_path, file_type, text FROM documents "
            "WHERE file_type = 'xlsx' AND text LIKE ? ORDER BY file_path",
            (f"%{station_name}%",),
        ).fetchall()
    except sqlite3.DatabaseError:
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass
    for row in rows:
        records, excerpt_text = extract_station_table_records_with_excerpt(str(row["text"]), station_name)
        if not records:
            continue
        include_schemes = station_question_includes_schemes(question)
        if include_schemes:
            lines = [
                f"- {record['item']}：{record['scheme'] or '资料中未填写测试方案'}"
                for record in records
            ]
            answer = (
                f"根据本地知识库，{station_name.upper()} 站位包含以下测试项及对应测试方案：\n\n"
                + "\n".join(lines)
                + "\n\n[1]"
            )
        else:
            answer = (
                f"根据本地知识库，{station_name.upper()} 站位包含以下测试项："
                + "；".join(record["item"] for record in records)
                + " [1]。"
            )
        excerpt = structured_cell_text(excerpt_text, max_len=600)
        source = {
            "source_id": 1,
            "file": row["file_name"],
            "file_path": row["file_path"],
            "file_type": row["file_type"],
            "chunk_id": f"{row['doc_id']}:structured:{station_name.lower()}",
            "excerpt": excerpt,
            "citation": excerpt,
        }
        metadata = metadata or {}
        return {
            "answer": answer,
            "answer_mode": "local_structured_table_summary",
            "source_type": "knowledge_base",
            "sources": [source],
            "citations": [{"source_id": 1, "file": row["file_name"], "chunk_id": source["chunk_id"], "quote": excerpt}],
            "need_web_search": False,
            "web_search_used": False,
            "index_updated_at": metadata.get("indexed_at"),
            "embedding_model": metadata.get("embedding_model"),
            "chunk_strategy": metadata.get("chunk_strategy"),
            "retrieval_mode": "local_structured_table",
            "async_update_scheduled": refresh_scheduled,
            "cache_hit": False,
        }
    return None


def normalize_excel_key(text: str) -> str:
    text = structured_cell_text(text).lower()
    return re.sub(r"[\s_\-—–:：；;，,。/\\()（）【】\[\]<>《》\"'`]+", "", text)


def n972_worksheet_sections(text: str) -> list[tuple[str, str]]:
    worksheet_re = re.compile("\u5de5\u4f5c\u8868 ([^\n]+)")
    matches = list(worksheet_re.finditer(text))
    sections: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections.append((match.group(1).strip(), text[start:end]))
    return sections


def n972_excel_field_intent(question: str) -> str | None:
    normalized = structured_cell_text(question)
    if re.search(r"测试子项|子项", normalized, flags=re.IGNORECASE):
        return "subitem"
    if re.search(r"测试工具|工具名称|使用什么.*工具|用什么.*工具", normalized, flags=re.IGNORECASE):
        return "tool"
    if re.search(r"治具|夹具|设备|辅料", normalized, flags=re.IGNORECASE):
        return "fixture"
    if re.search(r"门限|标准|规格|阈值", normalized, flags=re.IGNORECASE):
        return "threshold"
    if re.search(r"测试方案|方案|流程|步骤|怎么测|如何测", normalized, flags=re.IGNORECASE):
        return "scheme"
    if re.search(r"判定|卡关|判断方式|判定方式", normalized, flags=re.IGNORECASE):
        return "judgement"
    if re.search(r"是什么|多少|有哪些|什么", normalized, flags=re.IGNORECASE):
        return "spec"
    return None


def n972_value_is_useful(text: str) -> bool:
    value = structured_cell_text(text)
    if len(value) < 2:
        return False
    weak_values = {
        "NA",
        "N/A",
        "无",
        "なし",
        "TBD",
        "-",
        "--",
        "------",
        "OK",
        "Y",
        "N",
        "程序",
        "MES",
        "程序&MES",
        "程序/MES",
        "待导入",
        "已导入",
    }
    return value not in weak_values


def n972_record_from_cells(section_name: str, line: str) -> dict[str, str] | None:
    raw = [structured_cell_text(cell) for cell in line.split("|")]
    cells = raw + [""] * max(0, 12 - len(raw))
    if cells[0].isdigit() or (not cells[0] and not cells[1] and cells[2]):
        item = cells[2] or cells[1]
        subitem = cells[3]
        tool = cells[4]
        threshold = cells[7]
        scheme = cells[8]
        judgement = cells[9]
        fixture = cells[10]
    else:
        item = cells[1]
        subitem = cells[2]
        tool = cells[3]
        threshold = cells[4]
        scheme = cells[5]
        judgement = cells[6]
        fixture = cells[8]
    item = item or subitem
    if not item:
        return None
    if re.fullmatch(r"#|编号|序列|站位|工位|Category|sub-item|测试项|测试项目|测试子项|工具名称|工具信息|备注", item, re.I):
        return None
    return {
        "section": section_name,
        "item": item,
        "subitem": subitem,
        "tool": tool,
        "threshold": threshold,
        "scheme": scheme,
        "judgement": judgement,
        "fixture": fixture,
        "line": line,
    }


def n972_excel_records(text: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    spec_records: list[dict[str, str]] = []
    test_records: list[dict[str, str]] = []
    spec_field_re = re.compile(
        r"^(Main battery type|Quick Charge support|Adaptor Type|Connector Type|"
        r"WLAN|Bluetooth|HDMI|Type-C|Headphone|Power button|Battery indicator|"
        r"Camera indicator|Pixel- Front|Pixel- Rear|Codec|Microphone|Speakers|"
        r"Sensor|Keyboard|Touchpad Type|TouchPad)\b",
        re.I,
    )
    excluded_section = re.compile("修订|清单|check list|Test Flow|应答|门限评审", re.I)
    test_section = re.compile(
        r"SMT|DLTEST|FAT|FRT|FFT|SWDL|LCD|OOBE|Run In|功能测试|老化|售后备件|服务备件|门限表|测试门限",
        re.I,
    )
    for section_name, body in n972_worksheet_sections(text):
        lines = [line.strip() for line in body.splitlines() if "|" in line]
        if "feature list" in section_name.lower() or "产品规格" in section_name:
            for line in lines:
                columns = [structured_cell_text(cell) for cell in line.split("|")]
                non_empty = [cell for cell in columns if cell]
                if len(non_empty) < 2:
                    continue
                field = non_empty[0]
                if not spec_field_re.search(field):
                    continue
                value = next((cell for cell in non_empty[1:] if n972_value_is_useful(cell)), "")
                if value:
                    spec_records.append({"field": field, "value": value, "section": section_name, "line": line})
            continue
        if not test_section.search(section_name) or excluded_section.search(section_name):
            continue
        for line in lines:
            record = n972_record_from_cells(section_name, line)
            if record:
                test_records.append(record)
    return spec_records, test_records


def n972_best_record(question: str, records: list[dict[str, str]], field: str) -> dict[str, str] | None:
    matches = n972_best_records(question, records, field)
    return matches[0] if matches else None


def n972_best_records(question: str, records: list[dict[str, str]], field: str) -> list[dict[str, str]]:
    q_key = normalize_excel_key(question)
    scored: list[tuple[int, dict[str, str]]] = []
    for record in records:
        value = record.get(field, "")
        if not n972_value_is_useful(value):
            continue
        score = 0
        for label in (record.get("item", ""), record.get("subitem", "")):
            label_key = normalize_excel_key(label)
            if not label_key or len(label_key) < 2:
                continue
            if label_key in q_key:
                score = max(score, len(label_key) + 20)
                if structured_cell_text(label).lower() in structured_cell_text(question).lower():
                    score += 10
            else:
                ratio = SequenceMatcher(None, label_key, q_key).ratio()
                if ratio >= 0.72:
                    score = max(score, int(ratio * 20))
        section_key = normalize_excel_key(record.get("section", ""))
        if section_key and section_key in q_key:
            score += 5
        if score <= 0:
            continue
        scored.append((score, record))
    if not scored:
        return []
    best_score = max(score for score, _ in scored)
    selected: list[dict[str, str]] = []
    seen_values: set[str] = set()
    for score, record in sorted(scored, key=lambda item: item[0], reverse=True):
        if score < best_score:
            break
        value_key = normalize_excel_key(record.get(field, ""))
        if not value_key or value_key in seen_values:
            continue
        seen_values.add(value_key)
        selected.append(record)
        if len(selected) >= 12:
            break
    return selected


def structured_n972_excel_result(
    db_path: Path,
    question: str,
    metadata: dict[str, str] | None = None,
    refresh_scheduled: bool = False,
) -> dict[str, Any] | None:
    intent = n972_excel_field_intent(question)
    if not intent:
        return None
    if "N972" not in question.upper() and not re.search(r"\b(SWDL|FAT|FRT|FFT|SMT|OOBE)\b", question, re.I):
        return None
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT doc_id, file_name, file_path, file_type, text
            FROM documents
            WHERE file_name = 'N972-ACER_Test plan_V3.1.xlsx'
            LIMIT 1
            """
        ).fetchone()
    except sqlite3.DatabaseError:
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass
    if not row:
        return None

    spec_records, test_records = n972_excel_records(str(row["text"]))
    field_name = {
        "subitem": "测试子项",
        "tool": "测试工具",
        "threshold": "门限或标准",
        "scheme": "测试方案",
        "judgement": "判定方式",
        "fixture": "治具/设备/辅料",
        "spec": "规格",
    }[intent]

    if intent == "spec":
        q_key = normalize_excel_key(question)
        best_spec: tuple[int, dict[str, str]] | None = None
        for record in spec_records:
            field_key = normalize_excel_key(record["field"])
            if not field_key:
                continue
            score = len(field_key) + 20 if field_key in q_key else int(SequenceMatcher(None, field_key, q_key).ratio() * 20)
            if score > 12 and (best_spec is None or score > best_spec[0]):
                best_spec = (score, record)
        if not best_spec:
            return None
        record = best_spec[1]
        label = record["field"]
        value = record["value"]
        excerpt_line = record["line"]
    else:
        matched_records = n972_best_records(question, test_records, intent)
        if not matched_records:
            return None
        record = matched_records[0]
        label = record.get("subitem") or record.get("item") or "该测试项"
        values: list[str] = []
        for matched in matched_records:
            candidate = matched.get(intent, "")
            if n972_value_is_useful(candidate) and candidate not in values:
                values.append(candidate)
        value = "；".join(values)
        excerpt_line = f"{record.get('section', '')}: {record.get('line', '')}"

    if not n972_value_is_useful(value):
        return None
    excerpt = structured_cell_text(excerpt_line, max_len=1000)
    answer = f"根据本地知识库，N972 中 {label} 的{field_name}是：{value}。[1]"
    metadata = metadata or {}
    source = {
        "source_id": 1,
        "file": row["file_name"],
        "file_path": row["file_path"],
        "file_type": row["file_type"],
        "chunk_id": f"{row['doc_id']}:structured:n972-excel:{normalize_excel_key(label)[:48]}:{intent}",
        "excerpt": excerpt,
        "citation": excerpt,
    }
    return {
        "answer": answer,
        "answer_mode": "local_structured_excel_summary",
        "source_type": "knowledge_base",
        "sources": [source],
        "citations": [{"source_id": 1, "file": row["file_name"], "chunk_id": source["chunk_id"], "quote": excerpt}],
        "need_web_search": False,
        "web_search_used": False,
        "index_updated_at": metadata.get("indexed_at"),
        "embedding_model": metadata.get("embedding_model"),
        "chunk_strategy": metadata.get("chunk_strategy"),
        "retrieval_mode": "local_structured_excel",
        "async_update_scheduled": refresh_scheduled,
        "cache_hit": False,
    }


def extract_fft_manual_items(text: str) -> list[str]:
    records: list[str] = []
    current_parent = ""
    for raw_line in text.splitlines():
        if "|" not in raw_line:
            continue
        cells = [structured_cell_text(cell) for cell in raw_line.split("|")]
        if len(cells) < 4:
            continue
        if cells[0].isdigit():
            parent = cells[2] if not cells[1] or "人工功能测试" in cells[1] else cells[1]
            child = cells[3]
        elif not cells[0]:
            parent = cells[2] if len(cells) > 2 and not cells[1] else cells[1]
            child = cells[3] if len(cells) > 3 else ""
        else:
            continue
        if parent:
            current_parent = parent
        elif current_parent:
            parent = current_parent
        if not parent or not child:
            continue
        item = parent if parent == child else f"{parent} / {child}"
        if item not in records:
            records.append(item)
        if parent == "FFT 过站":
            break
    return records


def extract_fft_station_items(text: str) -> list[str]:
    records: list[str] = []
    current_category = ""
    current_item = ""
    for raw_line in text.splitlines():
        if "|" not in raw_line:
            continue
        cells = [structured_cell_text(cell) for cell in raw_line.split("|")]
        if not cells:
            continue
        if cells[0].isdigit():
            offset = 0
        elif cells[0] == "" and len(cells) > 3 and (cells[1] or cells[2] or cells[3]):
            offset = 0
        else:
            continue
        category = cells[offset + 1] if len(cells) > offset + 1 else ""
        item = cells[offset + 2] if len(cells) > offset + 2 else ""
        detail = cells[offset + 3] if len(cells) > offset + 3 else ""
        if category in {"测试项", "Category"} or item in {"测试项", "测试子项", "sub-item"} or detail in {"测试子项", "sub-item"}:
            continue
        if category:
            current_category = category
        else:
            category = current_category
        if item:
            current_item = item
        else:
            item = current_item
        if not item and detail:
            item, detail = detail, ""
        labels = [label for label in (category, item, detail) if label]
        deduped_labels: list[str] = []
        for label in labels:
            if not deduped_labels or deduped_labels[-1] != label:
                deduped_labels.append(label)
        if not labels:
            continue
        record = " / ".join(deduped_labels)
        if record not in records:
            records.append(record)
        if record.startswith("FFT 过站") or "FFT 过站 / FFT扫描过站" in record:
            break
    return records


def legacy_xlsx_region(
    db_path: Path,
    required: tuple[str, ...],
    start_terms: tuple[str, ...],
) -> tuple[dict[str, Any], str] | None:
    """A sheet region of the flattened workbook text in documents.text.

    Returns the first sheet holding every required string, from its first line containing
    every start term to the end of that sheet. Parsers that read cells by column position use
    this instead of chunks, whose layout is built for retrieval and may change.
    """
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            documents = conn.execute(
                "SELECT doc_id, file_name, file_path, file_type, text FROM documents "
                "WHERE file_type = 'xlsx' AND duplicate_of IS NULL ORDER BY file_path"
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return None
    for document in documents:
        for sheet in re.split(r"(?m)^(?=工作表 )", str(document["text"])):
            if not all(term in sheet for term in required):
                continue
            lines = sheet.splitlines()
            start = next((index for index, line in enumerate(lines) if all(term in line for term in start_terms)), None)
            if start is not None:
                return dict(document), "\n".join(lines[start:])
    return None


def structured_fft_station_result(
    db_path: Path,
    question: str,
    metadata: dict[str, str] | None = None,
    refresh_scheduled: bool = False,
) -> dict[str, Any] | None:
    station_name = requested_station_name(question)
    if not station_name or station_name.lower() != "fft" or not station_test_item_question(question):
        return None
    region = legacy_xlsx_region(
        db_path,
        required=("测试项目", "工具信息", "Test Coverage", "FFT 工具打开前"),
        start_terms=("测试项目", "工具信息"),
    )
    if region is None:
        return None
    document, context = region
    records = extract_fft_station_items(context)
    if len(records) < 10:
        return None
    lines = "\n".join(f"- {record}" for record in records)
    excerpt = structured_cell_text("\n".join(context.splitlines()[:3]), max_len=900)
    metadata = metadata or {}
    source = {
        "source_id": 1,
        "file": document["file_name"],
        "file_path": document["file_path"],
        "file_type": document["file_type"],
        "chunk_id": f"{document['doc_id']}:structured:station:fft",
        "excerpt": excerpt,
        "citation": excerpt,
    }
    return {
        "answer": f"根据本地知识库，FFT 站位测试项如下：\n{lines}\n[1]",
        "answer_mode": "local_structured_table_summary",
        "source_type": "knowledge_base",
        "sources": [source],
        "citations": [{"source_id": 1, "file": document["file_name"], "chunk_id": source["chunk_id"], "quote": excerpt}],
        "need_web_search": False,
        "web_search_used": False,
        "index_updated_at": metadata.get("indexed_at"),
        "embedding_model": metadata.get("embedding_model"),
        "chunk_strategy": metadata.get("chunk_strategy"),
        "retrieval_mode": "local_structured_table",
        "async_update_scheduled": refresh_scheduled,
        "cache_hit": False,
    }


def structured_fft_manual_function_answer(db_path: Path, question: str, sources: list[dict[str, Any]]) -> str | None:
    if not fft_manual_function_question(question):
        return None
    if not db_path.exists():
        return None
    region = legacy_xlsx_region(db_path, required=("人工功能测试", "IR CameraTest"), start_terms=("人工功能测试", "IR CameraTest"))
    if region is None:
        return None
    document, context = region
    records = extract_fft_manual_items(context)
    if len(records) < 6:
        return None
    source = next((item for item in sources if item.get("file") == document["file_name"]), sources[0] if sources else {"source_id": 1})
    source_id = int(source.get("source_id", 1))
    lines = "\n".join(f"- {record}" for record in records)
    return f"人工功能测试 / FFT测试包含以下测试项：\n{lines}[{source_id}]"


def structured_local_answer(db_path: Path, question: str, sources: list[dict[str, Any]]) -> str | None:
    fft_manual_answer = structured_fft_manual_function_answer(db_path, question, sources)
    if fft_manual_answer:
        return fft_manual_answer
    result = structured_station_result(db_path, question)
    if not result:
        return None
    source_id = next(
        (source["source_id"] for source in sources if source.get("file") == result["sources"][0]["file"]),
        1,
    )
    return str(result["answer"]).replace("[1]", f"[{source_id}]")
