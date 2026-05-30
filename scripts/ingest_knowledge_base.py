#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Import local documents and build a SQLite FTS5 knowledge-base index."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import zipfile
import zlib
from pathlib import Path
from xml.etree import ElementTree as ET


SUPPORTED = {".md", ".docx", ".pdf", ".xlsx"}
DB_PATH = Path("knowledge_base/index/knowledge.db")
PROCESSED_DIR = Path("knowledge_base/processed")


def read_plain_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def read_docx(path: Path) -> str:
    with zipfile.ZipFile(path) as zf:
        root = ET.fromstring(zf.read("word/document.xml"))

    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:p", ns):
        parts: list[str] = []
        for node in paragraph.iter():
            tag = node.tag.rsplit("}", 1)[-1]
            if tag == "t" and node.text:
                parts.append(node.text)
            elif tag == "tab":
                parts.append("\t")
            elif tag in {"br", "cr"}:
                parts.append("\n")
        line = "".join(parts).strip()
        if line:
            paragraphs.append(line)
    return "\n".join(paragraphs)


def _xlsx_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []

    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    strings: list[str] = []
    for item in root.findall(".//m:si", ns):
        strings.append("".join(node.text or "" for node in item.findall(".//m:t", ns)))
    return strings


def read_xlsx(path: Path) -> str:
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    lines: list[str] = []

    with zipfile.ZipFile(path) as zf:
        shared = _xlsx_shared_strings(zf)
        sheets = sorted(name for name in zf.namelist() if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name))

        for sheet_index, sheet in enumerate(sheets, start=1):
            root = ET.fromstring(zf.read(sheet))
            lines.append(f"工作表 sheet{sheet_index}")
            for row in root.findall(".//m:row", ns):
                cells: list[str] = []
                for cell in row.findall("m:c", ns):
                    cell_type = cell.attrib.get("t")
                    value = ""
                    if cell_type == "inlineStr":
                        value = "".join(node.text or "" for node in cell.findall(".//m:t", ns))
                    else:
                        value_node = cell.find("m:v", ns)
                        if value_node is not None:
                            value = value_node.text or ""
                            if cell_type == "s" and value.isdigit():
                                index = int(value)
                                if 0 <= index < len(shared):
                                    value = shared[index]
                    cells.append(value)
                if any(cell.strip() for cell in cells):
                    lines.append(" | ".join(cells))
    return "\n".join(lines)


def _read_pdf_with_library(path: Path) -> str | None:
    for module_name in ("pypdf", "PyPDF2"):
        try:
            module = __import__(module_name)
        except ImportError:
            continue
        reader = module.PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(page.strip() for page in pages if page.strip())
        if text.strip():
            return text
    return None


def _decode_pdf_literal(raw: bytes) -> str:
    raw = raw.replace(rb"\\(", b"(").replace(rb"\\)", b")").replace(rb"\\\\", b"\\")
    for encoding in ("utf-8", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="ignore")


def read_pdf(path: Path) -> str:
    text = _read_pdf_with_library(path)
    if text is not None:
        return text

    data = path.read_bytes()
    pieces: list[str] = []
    for match in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", data, flags=re.S):
        raw = match.group(1)
        try:
            raw = zlib.decompress(raw)
        except zlib.error:
            pass
        for text_match in re.finditer(rb"\((.*?)\)\s*Tj", raw, flags=re.S):
            pieces.append(_decode_pdf_literal(text_match.group(1)))
        for array_match in re.finditer(rb"\[(.*?)\]\s*TJ", raw, flags=re.S):
            for text_match in re.finditer(rb"\((.*?)\)", array_match.group(1), flags=re.S):
                pieces.append(_decode_pdf_literal(text_match.group(1)))
    return "\n".join(piece.strip() for piece in pieces if piece.strip())


def read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".md":
        return read_plain_text(path)
    if suffix == ".docx":
        return read_docx(path)
    if suffix == ".xlsx":
        return read_xlsx(path)
    if suffix == ".pdf":
        return read_pdf(path)
    raise ValueError(f"Unsupported file type: {path}")


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, size: int = 900, overlap: int = 120) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if not current:
            current = paragraph
        elif len(current) + len(paragraph) + 2 <= size:
            current = f"{current}\n\n{paragraph}"
        else:
            chunks.extend(_split_long_text(current, size, overlap))
            current = paragraph
    if current:
        chunks.extend(_split_long_text(current, size, overlap))
    return chunks


def _split_long_text(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        breakpoint = max(text.rfind("\n", start, end), text.rfind("。", start, end), text.rfind(".", start, end))
        if breakpoint > start + size // 2:
            end = breakpoint + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def searchable_text(text: str) -> str:
    chinese_terms: list[str] = []
    for block in re.findall(r"[\u4e00-\u9fff]+", text):
        for size in (2, 3, 4):
            if len(block) < size:
                continue
            chinese_terms.extend(block[index : index + size] for index in range(len(block) - size + 1))
    return f"{text}\n{' '.join(chinese_terms)}"


def init_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        DROP TABLE IF EXISTS documents;
        DROP TABLE IF EXISTS chunks;
        DROP TABLE IF EXISTS chunks_fts;

        CREATE TABLE documents (
            doc_id TEXT PRIMARY KEY,
            file_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_type TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            modified_at TEXT NOT NULL,
            text TEXT NOT NULL
        );

        CREATE TABLE chunks (
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
        );

        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            chunk_id UNINDEXED,
            doc_id UNINDEXED,
            file_name UNINDEXED,
            file_path UNINDEXED,
            file_type UNINDEXED,
            text,
            search_text,
            tokenize='unicode61'
        );
        """
    )
    return conn


def ingest(raw_dir: Path, db_path: Path = DB_PATH, processed_dir: Path = PROCESSED_DIR) -> dict[str, int]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    conn = init_db(db_path)
    doc_count = 0
    chunk_count = 0
    skipped_count = 0

    with (processed_dir / "documents.jsonl").open("w", encoding="utf-8") as docs_out, (
        processed_dir / "chunks.jsonl"
    ).open("w", encoding="utf-8") as chunks_out:
        for path in sorted(raw_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED:
                continue

            try:
                text = normalize_text(read_document(path))
            except Exception as exc:
                skipped_count += 1
                print(f"skip {path}: {exc}")
                continue
            if not text:
                skipped_count += 1
                continue

            doc_count += 1
            doc_id = f"doc_{doc_count:04d}"
            rel_path = str(path.relative_to(raw_dir)).replace("\\", "/")
            file_type = path.suffix.lower().lstrip(".")
            stat = path.stat()
            modified_at = f"{stat.st_mtime:.6f}"

            doc_record = {
                "doc_id": doc_id,
                "file_name": path.name,
                "file_path": rel_path,
                "file_type": file_type,
                "file_size": stat.st_size,
                "modified_at": modified_at,
                "text": text,
            }
            conn.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    doc_id,
                    path.name,
                    rel_path,
                    file_type,
                    stat.st_size,
                    modified_at,
                    text,
                ),
            )
            docs_out.write(json.dumps(doc_record, ensure_ascii=False) + "\n")

            for index, chunk in enumerate(chunk_text(text), start=1):
                chunk_count += 1
                chunk_id = f"{doc_id}_{index:04d}"
                conn.execute("INSERT INTO chunks VALUES (?, ?, ?, ?)", (chunk_id, doc_id, index, chunk))
                conn.execute(
                    "INSERT INTO chunks_fts VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (chunk_id, doc_id, path.name, rel_path, file_type, chunk, searchable_text(chunk)),
                )
                chunks_out.write(
                    json.dumps(
                        {
                            "chunk_id": chunk_id,
                            "doc_id": doc_id,
                            "chunk_index": index,
                            "file_name": path.name,
                            "file_path": rel_path,
                            "file_type": file_type,
                            "text": chunk,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    conn.commit()
    conn.close()
    return {"documents": doc_count, "chunks": chunk_count, "skipped": skipped_count}


def main() -> None:
    parser = argparse.ArgumentParser(description="Import local documents and build a SQLite FTS5 index.")
    parser.add_argument("raw_dir", nargs="?", default="knowledge_base/raw")
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--processed", default=str(PROCESSED_DIR))
    args = parser.parse_args()

    result = ingest(Path(args.raw_dir), Path(args.db), Path(args.processed))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
