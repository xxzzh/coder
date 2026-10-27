"""Excel workbooks: sheet grids, header detection, item records, and text in drawings and charts."""

from __future__ import annotations

import posixpath
import re
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from kb_ingest.ocr import ocr_image_bytes
from kb_ingest.reviews import extraction_report


def _xlsx_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []

    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    strings: list[str] = []
    for item in root.findall(".//m:si", ns):
        strings.append("".join(node.text or "" for node in item.findall(".//m:t", ns)))
    return strings


def _xlsx_column_index(cell_reference: str) -> int:
    match = re.match(r"([A-Z]+)", cell_reference.upper())
    if not match:
        return 1
    index = 0
    for letter in match.group(1):
        index = index * 26 + ord(letter) - ord("A") + 1
    return index


def _xlsx_sheets(zf: zipfile.ZipFile) -> list[tuple[str, str]]:
    main_ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    package_rel_ns = "http://schemas.openxmlformats.org/package/2006/relationships"
    workbook_path = "xl/workbook.xml"
    relationships_path = "xl/_rels/workbook.xml.rels"
    if workbook_path not in zf.namelist() or relationships_path not in zf.namelist():
        sheets = sorted(
            (name for name in zf.namelist() if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name)),
            key=lambda name: int(re.search(r"sheet(\d+)", name).group(1)),
        )
        return [(f"sheet{index}", sheet) for index, sheet in enumerate(sheets, start=1)]

    relationships_root = ET.fromstring(zf.read(relationships_path))
    targets = {
        relationship.attrib["Id"]: relationship.attrib["Target"]
        for relationship in relationships_root.findall(f"{{{package_rel_ns}}}Relationship")
    }
    workbook_root = ET.fromstring(zf.read(workbook_path))
    sheets: list[tuple[str, str]] = []
    for sheet in workbook_root.findall(f".//{{{main_ns}}}sheet"):
        relation_id = sheet.attrib.get(f"{{{rel_ns}}}id")
        target = targets.get(str(relation_id), "")
        if not target:
            continue
        normalized_target = target.lstrip("/")
        if not normalized_target.startswith("xl/"):
            normalized_target = f"xl/{normalized_target}"
        sheets.append((sheet.attrib.get("name", f"sheet{len(sheets) + 1}"), normalized_target))
    return sheets


XLSX_DRAWING_NS = {
    "xdr": "http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
}

OFFICE_RELATIONSHIP_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

PACKAGE_RELATIONSHIP_NS = "http://schemas.openxmlformats.org/package/2006/relationships"

CHART_POINTS_MAX = 60

# Package objects wrap an attached file; Excel localises their progId (Chinese Excel writes 包装程序外壳对象).
PACKAGE_PROG_IDS = {"package", "包装程序外壳对象"}


def _zip_part_relationships(zf: zipfile.ZipFile, part: str) -> list[tuple[str, str, str]]:
    """(relationship type, target part, relationship id) for the internal relationships of an OOXML part."""
    folder, name = posixpath.split(part)
    rels_path = posixpath.join(folder, "_rels", f"{name}.rels")
    if rels_path not in zf.namelist():
        return []
    relationships: list[tuple[str, str, str]] = []
    for relationship in ET.fromstring(zf.read(rels_path)).findall(f"{{{PACKAGE_RELATIONSHIP_NS}}}Relationship"):
        if relationship.attrib.get("TargetMode") == "External":
            continue
        target = relationship.attrib.get("Target", "")
        resolved = target.lstrip("/") if target.startswith("/") else posixpath.normpath(posixpath.join(folder, target))
        relationships.append((relationship.attrib.get("Type", "").rsplit("/", 1)[-1], resolved, relationship.attrib.get("Id", "")))
    return relationships


def _drawingml_text(element: ET.Element | None) -> str:
    """The paragraphs of a DrawingML text body (shape, title, SmartArt node), joined by spaces."""
    if element is None:
        return ""
    paragraphs = []
    for paragraph in element.iter(f"{{{XLSX_DRAWING_NS['a']}}}p"):
        line = "".join(node.text or "" for node in paragraph.iter(f"{{{XLSX_DRAWING_NS['a']}}}t")).strip()
        if line:
            paragraphs.append(line)
    return " ".join(paragraphs)


def _chart_values(node: ET.Element | None) -> list[str]:
    """Cached labels or numbers of a chart series part (name, categories, values), in point order."""
    if node is None:
        return []
    points = node.findall(".//c:pt", XLSX_DRAWING_NS)
    if points:
        ordered = sorted(points, key=lambda point: int(point.attrib.get("idx", "0") or 0))
        return [str(point.findtext("c:v", default="", namespaces=XLSX_DRAWING_NS)).strip() for point in ordered]
    text = _drawingml_text(node) or str(node.findtext(".//c:v", default="", namespaces=XLSX_DRAWING_NS)).strip()
    return [text] if text else []


def _xlsx_chart_text(root: ET.Element) -> str:
    """Title, axis titles and each series with its data points, as one line of text."""
    chart = root.find("c:chart", XLSX_DRAWING_NS)
    if chart is None:
        return ""

    def title_of(node: ET.Element | None) -> str:
        title = node.find("c:title", XLSX_DRAWING_NS) if node is not None else None
        return " ".join(_chart_values(title)) if title is not None else ""

    pieces: list[str] = []
    if title_of(chart):
        pieces.append(f"标题：{title_of(chart)}")
    for axis_tag, label in (("catAx", "横轴"), ("dateAx", "横轴"), ("valAx", "数值轴")):
        for axis in chart.iter(f"{{{XLSX_DRAWING_NS['c']}}}{axis_tag}"):
            if title_of(axis):
                pieces.append(f"{label}：{title_of(axis)}")
    for index, series in enumerate(chart.iter(f"{{{XLSX_DRAWING_NS['c']}}}ser"), start=1):
        name = " ".join(_chart_values(series.find("c:tx", XLSX_DRAWING_NS))) or f"系列{index}"
        categories = _chart_values(series.find("c:cat", XLSX_DRAWING_NS)) or _chart_values(series.find("c:xVal", XLSX_DRAWING_NS))
        values = _chart_values(series.find("c:val", XLSX_DRAWING_NS)) or _chart_values(series.find("c:yVal", XLSX_DRAWING_NS))
        if categories and values:
            points = [f"{category}={value}" for category, value in zip(categories, values)]
        else:
            points = categories or values
        pieces.append(f"{name}：{'，'.join(points[:CHART_POINTS_MAX])}" if points else name)
    return "；".join(pieces)


def _xlsx_drawing_contents(zf: zipfile.ZipFile) -> dict[str, list[str]]:
    """Text a workbook keeps outside its cells, tagged with the worksheet it sits on.

    texts: text boxes and shapes, charts (title, axes, series with data points) and SmartArt.
    unreadable: drawing or chart parts that could not be parsed.
    embedded: embedded objects (OLE); their contents are not extracted.
    Pictures are not covered here: they go through OCR.
    """
    names = set(zf.namelist())
    sheets = _xlsx_sheets(zf)
    drawing_sheets: dict[str, str] = {}
    for sheet_name, sheet_part in sheets:
        for relationship_type, target, _ in _zip_part_relationships(zf, sheet_part):
            if relationship_type == "drawing":
                drawing_sheets[target] = sheet_name
    texts: list[str] = []
    unreadable: list[str] = []
    for part in sorted(name for name in names if re.fullmatch(r"xl/drawings/drawing\d+\.xml", name)):
        sheet = drawing_sheets.get(part)
        prefix = f"[工作表 {sheet} " if sheet else "["
        try:
            root = ET.fromstring(zf.read(part))
            relationships = {rid: (kind, target) for kind, target, rid in _zip_part_relationships(zf, part)}
        except ET.ParseError:
            unreadable.append(part)
            continue
        seen: set[str] = set()
        # A shape can appear twice (mc:AlternateContent keeps a fallback copy).
        for shape in root.iter(f"{{{XLSX_DRAWING_NS['xdr']}}}sp"):
            shape_text = _drawingml_text(shape)
            if shape_text and shape_text not in seen:
                seen.add(shape_text)
                texts.append(f"{prefix}文本框] {shape_text}")
        for kind, target in relationships.values():
            if kind == "chart":
                try:
                    chart_text = _xlsx_chart_text(ET.fromstring(zf.read(target))) if target in names else ""
                except ET.ParseError:
                    chart_text = ""
                if chart_text:
                    texts.append(f"{prefix}图表] {chart_text}")
                else:
                    unreadable.append(target)
            elif kind == "diagramData" and target in names:
                try:
                    diagram_text = _drawingml_text(ET.fromstring(zf.read(target)))
                except ET.ParseError:
                    unreadable.append(target)
                    continue
                if diagram_text:
                    texts.append(f"{prefix}SmartArt] {diagram_text}")
            elif kind == "chartEx":
                unreadable.append(target)  # newer chart types (waterfall, treemap ...) are not parsed
    embedded: list[str] = []
    if any(name.startswith("xl/embeddings/") for name in names):
        seen_objects: set[tuple[str, str]] = set()
        for sheet_name, sheet_part in sheets:
            if sheet_part not in names:
                continue
            for match in re.finditer(rb"<oleObject\b[^>]*>", zf.read(sheet_part)):
                tag = match.group(0).decode("utf-8", errors="ignore")
                prog_match = re.search(r'progId="([^"]*)"', tag)
                id_match = re.search(r'r:id="([^"]*)"', tag)
                prog_id = prog_match.group(1) if prog_match else ""
                relationship_id = id_match.group(1) if id_match else match.group(0).decode("latin-1")
                if (sheet_part, relationship_id) in seen_objects:
                    continue  # mc:AlternateContent repeats the object tag
                seen_objects.add((sheet_part, relationship_id))
                kind = "attached file" if prog_id.strip().lower() in PACKAGE_PROG_IDS else (prog_id or "object")
                embedded.append(f"{kind} on sheet {sheet_name}")
        if not embedded:
            embedded = [f"object {posixpath.basename(name)}" for name in sorted(names) if name.startswith("xl/embeddings/")]
    return {"texts": texts, "unreadable": unreadable, "embedded": embedded}


def read_xlsx(path: Path) -> str:
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    lines: list[str] = []

    with zipfile.ZipFile(path) as zf:
        shared = _xlsx_shared_strings(zf)
        sheets = _xlsx_sheets(zf)

        for sheet_name, sheet in sheets:
            root = ET.fromstring(zf.read(sheet))
            lines.append(f"工作表 {sheet_name}")
            for row in root.findall(".//m:row", ns):
                cells: dict[int, str] = {}
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
                    cells[_xlsx_column_index(cell.attrib.get("r", "A1"))] = re.sub(r"\s*\r?\n\s*", " ", value).strip()
                if any(cell.strip() for cell in cells.values()):
                    lines.append(" | ".join(cells.get(index, "") for index in range(1, max(cells) + 1)))
    return "\n".join(lines)


XLSX_HEADER_SCAN_ROWS = 8

XLSX_HEADER_CELL_MAX_CHARS = 16

XLSX_GROUP_HEADING_MAX_CHARS = 80

XLSX_OVERVIEW_ITEM_MAX_CHARS = 40

XLSX_OVERVIEW_DESCRIPTION_CHARS = 60

# documents.text keeps the legacy flattened workbook text that the table parsers read by column
# position; chunks are built from the structured rows passed under this report key.
CHUNK_TEXT_REPORT_KEY = "chunk_text"


def _xlsx_cell_value(cell: ET.Element, ns: dict[str, str], shared: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        value = "".join(node.text or "" for node in cell.findall(".//m:t", ns))
    else:
        value_node = cell.find("m:v", ns)
        value = value_node.text or "" if value_node is not None else ""
        if cell_type == "s" and value.isdigit() and 0 <= int(value) < len(shared):
            value = shared[int(value)]
    return re.sub(r"\s*\r?\n\s*", " ", value).strip()


def _xlsx_ref(reference: str) -> tuple[int, int]:
    match = re.fullmatch(r"([A-Z]+)(\d+)", reference.upper())
    if not match:
        return 0, 0
    return int(match.group(2)), _xlsx_column_index(match.group(1))


def _xlsx_sheet_grid(
    root: ET.Element, ns: dict[str, str], shared: list[str]
) -> tuple[dict[int, dict[int, str]], set[tuple[int, int]], list[tuple[int, int, int, int]]]:
    """Cell values by row/column with merged ranges filled from their top-left cell.

    Returns the grid, the set of cells whose value was inherited from a merge, and the
    merge ranges as (first_row, first_col, last_row, last_col).
    """
    grid: dict[int, dict[int, str]] = {}
    for row_number, row in enumerate(root.findall(".//m:row", ns), start=1):
        row_index = int(row.attrib.get("r", row_number))
        for cell in row.findall("m:c", ns):
            value = _xlsx_cell_value(cell, ns, shared)
            if value:
                _, column = _xlsx_ref(cell.attrib.get("r", f"A{row_index}"))
                grid.setdefault(row_index, {})[column or 1] = value
    merges: list[tuple[int, int, int, int]] = []
    for merge in root.findall(".//m:mergeCells/m:mergeCell", ns):
        first, _, last = merge.attrib.get("ref", "").partition(":")
        (r1, c1), (r2, c2) = _xlsx_ref(first), _xlsx_ref(last or first)
        if r1 and c1 and r2 and c2:
            merges.append((r1, c1, r2, c2))
    inherited: set[tuple[int, int]] = set()
    for r1, c1, r2, c2 in merges:
        value = grid.get(r1, {}).get(c1)
        if not value:
            continue
        for row_index in range(r1, r2 + 1):
            for column in range(c1, c2 + 1):
                if (row_index, column) != (r1, c1):
                    grid.setdefault(row_index, {})[column] = value
                    inherited.add((row_index, column))
    return grid, inherited, merges


def _xlsx_header_score(cells: dict[int, str]) -> int:
    """Count of short, digit-free text cells: what column titles look like."""
    return sum(
        1
        for value in dict.fromkeys(cells.values())
        if 2 <= len(value) <= XLSX_HEADER_CELL_MAX_CHARS and not re.search(r"\d", value)
    )


def _xlsx_header_labels(
    grid: dict[int, dict[int, str]],
    inherited: set[tuple[int, int]],
    merges: list[tuple[int, int, int, int]],
) -> tuple[dict[int, str], int | None]:
    """Column labels and the last header row, or ({}, None) when the sheet has no table header."""
    rows = sorted(grid)
    candidates = rows[:XLSX_HEADER_SCAN_ROWS]
    # Score only a row's own cells: vertical merges copy the top header's titles into the row below.
    scores = {
        row: _xlsx_header_score({column: value for column, value in grid[row].items() if (row, column) not in inherited})
        for row in candidates
    }
    best = max(scores.values(), default=0)
    if best < 3:
        return {}, None
    header = next(row for row in candidates if scores[row] >= max(3, round(best * 0.7)))
    if len(rows) - rows.index(header) - 1 < 2:
        # Header rows are not emitted as data, so a "header" with almost nothing under it
        # (a key-value cover sheet, say) would make its content disappear.
        return {}, None
    labels = dict(grid[header])
    last_header = header
    following = rows[rows.index(header) + 1] if rows.index(header) + 1 < len(rows) else None
    own_following = (
        [value for column, value in grid[following].items() if (following, column) not in inherited] if following is not None else []
    )
    if own_following and max(map(len, own_following)) <= XLSX_HEADER_CELL_MAX_CHARS:
        # A second header row only when the whole row is short titles and it subdivides a
        # horizontally merged header cell, e.g. 「测试项目」 over 「测试项 | 测试子项」.
        subdivides = any(
            r1 <= header == r2
            and c2 > c1
            and _xlsx_header_score(
                {
                    column: value
                    for column, value in grid[following].items()
                    if c1 <= column <= c2 and (following, column) not in inherited
                }
            )
            >= 2
            for r1, c1, r2, c2 in merges
        )
        if subdivides:
            for column, value in grid[following].items():
                labels[column] = "/".join(dict.fromkeys(part for part in (labels.get(column, ""), value) if part))
            last_header = following
    return labels, last_header


def _xlsx_item_columns(
    labels: dict[int, str],
    data_rows: list[int],
    grid: dict[int, dict[int, str]],
    inherited: set[tuple[int, int]],
    vertical: set[tuple[int, int]],
) -> list[int]:
    """The two titled columns that name a row's item (e.g. 测试项 + 测试子项, or 测试子项 + 测试门限).

    Skipped: mostly-numeric columns (#, 序号), the merged grouping columns (站位, 工位) and long
    description columns (测试方案).
    """
    chosen: list[int] = []
    for column in sorted(labels):
        own = [grid[row][column] for row in data_rows if column in grid[row] and (row, column) not in inherited]
        present = [row for row in data_rows if column in grid[row]]
        if not own or not present:
            continue
        if sum(bool(re.fullmatch(r"[\d.\s]+", value)) for value in own) > len(own) / 2:
            continue
        if sum((row, column) in vertical for row in present) > len(present) / 2:
            continue
        if sorted(len(value) for value in own)[len(own) // 2] > XLSX_OVERVIEW_DESCRIPTION_CHARS:
            continue
        if any(labels[column] == labels[other] for other in chosen):
            continue  # a header merged across several columns
        chosen.append(column)
        if len(chosen) == 2:
            break
    return chosen


def read_xlsx_records(path: Path) -> str:
    """Excel as self-describing rows for chunking: one line per data row, each value with its column title.

    Merged cells are filled, so a station or item that spans many rows appears on every row;
    empty rows, rows holding only inherited values and empty columns are dropped. Rows that share
    vertically merged cells (one station, one test item) form a block headed by that context, so a
    chunk holds one coherent group; sheets organised that way also get an overview block listing
    every group's items, for "which items does X include" questions.
    """
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    blocks: list[str] = []
    with zipfile.ZipFile(path) as zf:
        shared = _xlsx_shared_strings(zf)
        for sheet_name, sheet in _xlsx_sheets(zf):
            root = ET.fromstring(zf.read(sheet))
            grid, inherited, merges = _xlsx_sheet_grid(root, ns, shared)
            if not grid:
                continue
            labels, last_header = _xlsx_header_labels(grid, inherited, merges)
            title = sheet_name.strip()
            vertical = {
                (row, column)
                for r1, c1, r2, c2 in merges
                if r2 > r1
                for row in range(r1, r2 + 1)
                for column in range(c1, c2 + 1)
            }
            data_rows = [
                row
                for row in sorted(grid)
                if (last_header is None or row > last_header) and any((row, column) not in inherited for column in grid[row])
            ]
            item_columns = _xlsx_item_columns(labels, data_rows, grid, inherited, vertical) if labels else []
            # Groups are hierarchical from the left (station > test item); merged cells further right
            # (owner, a shared procedure text) must not split or name a group.
            key_limit = min(item_columns) if item_columns else None

            groups: list[tuple[tuple[str, ...], list[str], list[str]]] = []
            seen: set[str] = set()
            for row_index in data_rows:
                cells = grid[row_index]
                parts: list[str] = []
                for column, value in sorted(cells.items()):
                    label = labels.get(column, "")
                    part = f"{label}：{value}" if label and label != value else value
                    if part not in parts:  # a horizontal merge repeats one value across its columns
                        parts.append(part)
                line = f"{title} | " + " | ".join(parts)
                if line in seen:
                    continue
                seen.add(line)
                key = tuple(
                    dict.fromkeys(
                        value
                        for column, value in sorted(cells.items())
                        if (row_index, column) in vertical and (key_limit is None or column < key_limit)
                    )
                )
                item_values = dict.fromkeys(cells[column] for column in item_columns if column in cells)
                item = " ".join(
                    value if len(value) <= XLSX_OVERVIEW_ITEM_MAX_CHARS else value[: XLSX_OVERVIEW_ITEM_MAX_CHARS - 1] + "…"
                    for value in item_values
                )
                if groups and groups[-1][0] == key:
                    groups[-1][1].append(line)
                    groups[-1][2].append(item)
                else:
                    groups.append((key, [line], [item]))

            ungrouped_open = False
            for key, lines, _ in groups:
                if key:
                    context = " / ".join(key)
                    if len(context) > XLSX_GROUP_HEADING_MAX_CHARS:
                        context = context[: XLSX_GROUP_HEADING_MAX_CHARS - 1] + "…"
                    blocks.append(f"## 工作表 {title} / {context}\n\n" + "\n\n".join(lines))
                    ungrouped_open = False
                elif ungrouped_open:
                    blocks[-1] += "\n\n" + "\n\n".join(lines)
                else:
                    blocks.append(f"## 工作表 {title}\n\n" + "\n\n".join(lines))
                    ungrouped_open = True

            # One block per group: the group line, then one item per line, so a cited overview lists
            # items as separate lines instead of one long row that gets shortened.
            overview = [
                f"{title} 概览 | {' / '.join(key)}：\n" + "\n".join(dict.fromkeys(item for item in items if item))
                for key, _, items in groups
                if key and any(items)
            ]
            if overview and len(data_rows) >= 5:
                blocks.append(f"## 工作表 {title} 概览\n\n" + "\n\n".join(overview))
    return "\n\n".join(blocks)


def read_xlsx_with_report(path: Path) -> tuple[str, dict[str, Any]]:
    report = extraction_report(path, "xlsx")
    text = read_xlsx(path)
    try:
        structured = read_xlsx_records(path)
    except (ET.ParseError, KeyError, zipfile.BadZipFile):
        structured = ""
    image_texts: list[str] = []
    drawing = {"texts": [], "unreadable": [], "embedded": []}
    media_files: list[str] = []
    try:
        with zipfile.ZipFile(path) as zf:
            media_files = [name for name in zf.namelist() if name.startswith("xl/media/")]
            for media_file in media_files:
                media_text = ocr_image_bytes(media_file, zf.read(media_file))
                if media_text:
                    image_texts.append(f"[embedded image {Path(media_file).name}]\n{media_text}")
            drawing = _xlsx_drawing_contents(zf)
    except (zipfile.BadZipFile, ET.ParseError, KeyError):
        pass
    extra_texts = [*drawing["texts"], *image_texts]
    if extra_texts:
        text = "\n\n".join(part for part in [text, *extra_texts] if part).strip()
    if drawing["texts"]:
        report["method"].append("drawing_text")
    if image_texts:
        report["method"].append("embedded_image_ocr")
    if structured:
        # Carried to the ingest loop, which chunks it and removes it before the report is written.
        report[CHUNK_TEXT_REPORT_KEY] = "\n\n".join(part for part in [structured, *extra_texts] if part).strip()
        report["method"].append("structured_rows")
    # Review is needed only for content that could not be extracted: text boxes, charts and SmartArt
    # are read above and pictures go through OCR.
    not_extracted: list[str] = []
    if drawing["unreadable"]:
        not_extracted.append(f"{len(drawing['unreadable'])} chart/drawing part(s) could not be read")
    if drawing["embedded"]:
        not_extracted.append(f"{len(drawing['embedded'])} embedded object(s) not extracted ({'; '.join(drawing['embedded'])})")
    if media_files and not image_texts:
        not_extracted.append("embedded image(s) returned no OCR text")
    if not_extracted:
        report["warnings"].append("XLSX content not extracted: " + ", ".join(not_extracted) + ".")
        report["requires_review"] = True
    report["text_chars"] = len(text)
    return text, report
