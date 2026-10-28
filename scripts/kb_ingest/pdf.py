"""PDF text: pypdf, pdfplumber and pdftotext readers, with OCR for scanned pages."""

from __future__ import annotations

import hashlib
import io
import re
import shutil
import subprocess
import tempfile
import zlib
from pathlib import Path
from typing import Any

from kb_ingest.ocr import ocr_pdf
from kb_ingest.reviews import extraction_report


MIN_EXTRACTED_TEXT_CHARS = 80

# A page whose text layer is shorter than this is a scanned page and is read by OCR.
MIN_PAGE_TEXT_CHARS = 20

# Pictures smaller than this (in pixels) are icons and logos: a page with only these is not read by OCR.
MIN_OCR_PICTURE_PIXELS = 300 * 150

# Documents up to this many pages get OCR on every page; longer ones only where it adds something.
OCR_ALL_PAGES_UP_TO = 30

# A line counts as already in the text layer when this share of its character pairs is there.
LINE_KNOWN_SHARE = 0.6


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


def _read_pdf_with_pdfplumber(path: Path) -> str | None:
    try:
        import pdfplumber  # type: ignore
    except ImportError:
        return None

    pieces: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pieces.append(f"[page {page_index}]\n{text.strip()}")
            try:
                tables = page.extract_tables() or []
            except Exception:
                tables = []
            for table_index, table in enumerate(tables, start=1):
                rows = [
                    " | ".join((cell or "").strip() for cell in row)
                    for row in table
                    if row and any((cell or "").strip() for cell in row)
                ]
                if rows:
                    pieces.append(f"[page {page_index} table {table_index}]\n" + "\n".join(rows))
    text = "\n\n".join(pieces).strip()
    return text or None


def pdf_likely_has_images(path: Path) -> bool:
    data = path.read_bytes()
    return b"/Subtype" in data and b"/Image" in data


def find_pdftotext() -> str | None:
    found = shutil.which("pdftotext")
    if found:
        return found
    pdftoppm = shutil.which("pdftoppm")
    if pdftoppm:
        for suffix in (".cmd", ".exe", ""):
            candidate = Path(pdftoppm).with_name(f"pdftotext{suffix}")
            if candidate.exists():
                return str(candidate)
    return None


def _read_pdf_with_pdftotext(path: Path) -> str | None:
    pdftotext = find_pdftotext()
    if not pdftotext:
        return None
    with tempfile.TemporaryDirectory(prefix="lka_pdftotext_") as temp:
        output_path = Path(temp) / "out.txt"
        proc = subprocess.run(
            [pdftotext, "-layout", "-enc", "UTF-8", str(path), str(output_path)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if proc.returncode != 0 or not output_path.exists():
            return None
        text = output_path.read_text(encoding="utf-8", errors="ignore").strip()
    return text or None


def _decode_pdf_literal(raw: bytes) -> str:
    raw = raw.replace(rb"\\(", b"(").replace(rb"\\)", b")").replace(rb"\\\\", b"\\")
    for encoding in ("utf-8", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="ignore")


def read_pdf(path: Path) -> str:
    text = _read_pdf_with_pdfplumber(path)
    if text is not None:
        return text

    text = _read_pdf_with_library(path)
    if text is not None:
        return text

    text = _read_pdf_with_pdftotext(path)
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


def pdf_library_installed() -> bool:
    for module_name in ("pdfplumber", "pypdf", "PyPDF2"):
        try:
            __import__(module_name)
        except ImportError:
            continue
        return True
    return False


def squashed(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.lower())


def char_pairs(text: str) -> set[str]:
    return {text[index : index + 2] for index in range(len(text) - 1)}


def lines_not_in(text: str, known: str) -> str:
    """The lines of `text` that `known` does not already say. OCR misreads a character here and
    there, so a line counts as known when most of its character pairs occur in `known`."""
    known_compact = squashed(known)
    if not known_compact:
        return text.strip()
    known_pairs = char_pairs(known_compact)
    kept: list[str] = []
    for line in text.splitlines():
        compact = squashed(line)
        if not compact:
            continue
        pairs = char_pairs(compact)
        if len(compact) < 4 or not pairs:
            new = compact not in known_compact
        else:
            new = len(pairs & known_pairs) < LINE_KNOWN_SHARE * len(pairs)
        if new:
            kept.append(line.strip())
    return "\n".join(kept)


def text_by_page(text: str) -> dict[int, str]:
    """pdfplumber's text split at its "[page 3]" and "[page 3 table 1]" markers."""
    pages: dict[int, list[str]] = {}
    parts = re.split(r"^\[page (\d+)(?: table \d+)?\]$", text, flags=re.M)
    for number, body in zip(parts[1::2], parts[2::2]):
        pages.setdefault(int(number), []).append(body)
    return {number: "\n".join(bodies) for number, bodies in pages.items()}


def pdf_pages_for_ocr(path: Path) -> tuple[list[int], dict[int, str]] | None:
    """The pages to read by OCR (1-based) and each page's text layer. A short document gets OCR
    on every page: its tables and screenshots are often drawn without a text layer (an error code
    table, say). A long one only on pages without a text layer (scanned) and pages with a larger
    picture, so a 400-page book with a few figures does not take hours. None when pypdf or Pillow
    is missing and the pages cannot be told apart."""
    try:
        from PIL import Image  # type: ignore
        from pypdf import PdfReader  # type: ignore
    except ImportError:
        return None
    try:
        pages = list(PdfReader(str(path)).pages)
    except Exception:
        return None
    layer: dict[int, str] = {}
    pictures: dict[int, list[bytes]] = {}
    for number, page in enumerate(pages, start=1):
        try:
            layer[number] = page.extract_text() or ""
        except Exception:
            layer[number] = ""
        if len(pages) <= OCR_ALL_PAGES_UP_TO:
            continue
        try:
            images = list(page.images)
        except Exception:
            continue
        for image in images:
            try:
                width, height = Image.open(io.BytesIO(image.data)).size
            except Exception:
                continue
            if width * height >= MIN_OCR_PICTURE_PIXELS:
                pictures.setdefault(number, []).append(hashlib.sha1(image.data).digest())
    if len(pages) <= OCR_ALL_PAGES_UP_TO:
        return list(layer), layer
    # A picture repeated on many pages is a logo or a page banner, not content.
    repeats: dict[bytes, int] = {}
    for digests in pictures.values():
        for digest in set(digests):
            repeats[digest] = repeats.get(digest, 0) + 1
    selected = [
        number
        for number in layer
        if len(layer[number].strip()) < MIN_PAGE_TEXT_CHARS
        or any(repeats[digest] <= 3 for digest in pictures.get(number, []))
    ]
    return selected, layer


def new_ocr_text(ocr_text: str, layer: dict[int, str], plumber_pages: dict[int, str]) -> str:
    """Keep from each OCR page only the lines its text layer does not already have."""
    parts = re.split(r"^\[ocr page (\d+)\]$", ocr_text, flags=re.M)
    pieces: list[str] = []
    for number, body in zip(parts[1::2], parts[2::2]):
        page = int(number)
        new = lines_not_in(body, layer.get(page, "") + "\n" + plumber_pages.get(page, ""))
        if new:
            pieces.append(f"[ocr page {page}]\n{new}")
    return "\n\n".join(pieces)


def read_pdf_with_report(path: Path) -> tuple[str, dict[str, Any]]:
    report = extraction_report(path, "pdf")

    # Each reader returns the same pages again, and repeated text crowds out other evidence in
    # search: pdfplumber (which also reads tables) comes first, and the other readers only add the
    # lines it missed.
    text = ""
    plumber_pages: dict[int, str] = {}
    for method, reader in (
        ("pdfplumber", _read_pdf_with_pdfplumber),
        ("pypdf_or_pypdf2", _read_pdf_with_library),
        ("pdftotext", _read_pdf_with_pdftotext),
    ):
        other = (reader(path) or "").strip()
        if not other:
            continue
        if method == "pdfplumber":
            plumber_pages = text_by_page(other)
        extra = lines_not_in(other, text) if text else other
        if extra:
            text = f"{text}\n\n{extra}" if text else extra
            report["method"].append(method)
    # Text literals pulled from the raw streams, only when no PDF library is installed: on a
    # scanned PDF this pattern search runs through every image stream and takes many minutes.
    if not text and not pdf_library_installed():
        text = read_pdf(path).strip()
        if text:
            report["method"].append("pdf_stream_literals")

    has_images = pdf_likely_has_images(path)
    targets = pdf_pages_for_ocr(path)
    if targets is None:
        needs_ocr = has_images or len(text) < MIN_EXTRACTED_TEXT_CHARS
        ocr_text, ocr_warnings = ocr_pdf(path) if needs_ocr else ("", [])
    else:
        pages, layer = targets
        needs_ocr = bool(pages)
        report["ocr_pages"] = len(pages)
        ocr_text, ocr_warnings = ocr_pdf(path, pages) if pages else ("", [])
        ocr_text = new_ocr_text(ocr_text, layer, plumber_pages)
    report["warnings"].extend(ocr_warnings)
    if ocr_text:
        text = "\n\n".join(part for part in [text, ocr_text] if part).strip()
        report["method"].append("ocr_tesseract")
    elif needs_ocr and any(warning.startswith("OCR skipped") for warning in ocr_warnings) and (has_images or len(text) < MIN_EXTRACTED_TEXT_CHARS):
        report["warnings"].append("PDF appears to contain images; OCR text was not available.")
        report["requires_review"] = True

    if not text:
        report["warnings"].append("No text could be extracted from PDF.")
        report["requires_review"] = True
    report["text_chars"] = len(text)
    return text, report
