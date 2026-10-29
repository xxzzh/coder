"""Tesseract OCR: locating the engine and language data, rendering PDF pages, choosing the surest page-segmentation mode."""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from kb_ingest import progress
from kb_ingest.files import PROJECT_ROOT, portable_error_message


PROJECT_TESSDATA_DIR = PROJECT_ROOT / "tools" / "tessdata"

OCR_LANGUAGES = "eng+chi_sim"

OCR_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def find_tesseract() -> str | None:
    found = shutil.which("tesseract")
    if found:
        return found
    for candidate in (
        Path("C:/Program Files/Tesseract-OCR/tesseract.exe"),
        Path("C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"),
    ):
        if candidate.exists():
            return str(candidate)
    return None


def find_tessdata_dir() -> Path | None:
    if PROJECT_TESSDATA_DIR.exists():
        return PROJECT_TESSDATA_DIR
    for candidate in (
        Path("C:/Program Files/Tesseract-OCR/tessdata"),
        Path("C:/Program Files (x86)/Tesseract-OCR/tessdata"),
    ):
        if candidate.exists():
            return candidate
    return None


def available_ocr_languages() -> str:
    tessdata_dir = find_tessdata_dir()
    if not tessdata_dir:
        return "eng"
    desired = OCR_LANGUAGES.split("+")
    available = [lang for lang in desired if (tessdata_dir / f"{lang}.traineddata").exists()]
    return "+".join(available or ["eng"])


OCR_RENDER_DPI = 300


def pdf_page_renderer() -> str | None:
    """How scanned PDF pages become images for OCR: Poppler's pdftoppm when it is on PATH,
    otherwise pypdfium2, which pdfplumber installs, so no separate download is needed."""
    if shutil.which("pdftoppm"):
        return "pdftoppm"
    try:
        import pypdfium2  # noqa: F401
    except ImportError:
        return None
    return "pypdfium2"


def render_pdf_pages_for_ocr(path: Path, temp_dir: Path, pages: list[int] | None = None) -> list[Path]:
    """Page images named page-0003.png by page number; `pages` (1-based) limits which are rendered."""
    renderer = pdf_page_renderer()
    if renderer == "pdftoppm":
        if pages is None:
            output_prefix = temp_dir / "page"
            subprocess.run(
                [str(shutil.which("pdftoppm")), "-r", str(OCR_RENDER_DPI), "-png", str(path), str(output_prefix)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return sorted(temp_dir.glob("page-*.png"))
        for number in pages:
            subprocess.run(
                [
                    str(shutil.which("pdftoppm")), "-r", str(OCR_RENDER_DPI), "-png", "-f", str(number), "-l", str(number),
                    "-singlefile", str(path), str(temp_dir / f"page-{number:04d}"),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        return sorted(temp_dir.glob("page-*.png"))
    if renderer == "pypdfium2":
        import pypdfium2

        images: list[Path] = []
        document = pypdfium2.PdfDocument(str(path))
        try:
            for index in range(len(document)) if pages is None else [number - 1 for number in pages if 0 < number <= len(document)]:
                page = document[index]
                try:
                    image_path = temp_dir / f"page-{index + 1:04d}.png"
                    page.render(scale=OCR_RENDER_DPI / 72).to_pil().save(image_path)
                    images.append(image_path)
                finally:
                    page.close()
        finally:
            document.close()
        return images
    return []


def tsv_mean_confidence(tsv_path: Path) -> float:
    """Tesseract's word confidence (0-100) averaged over recognised characters; -1 when unknown."""
    if not tsv_path.exists():
        return -1.0
    total = weight = 0.0
    for line in tsv_path.read_text(encoding="utf-8", errors="ignore").splitlines()[1:]:
        fields = line.split("\t")
        if len(fields) < 12 or not fields[11].strip():
            continue
        try:
            confidence = float(fields[10])
        except ValueError:
            continue
        if confidence < 0:
            continue
        total += confidence * len(fields[11].strip())
        weight += len(fields[11].strip())
    return total / weight if weight else -1.0


def ocr_image(path: Path) -> str:
    tesseract = find_tesseract()
    if not tesseract:
        return ""
    tessdata_dir = find_tessdata_dir()
    best_text, best_confidence = "", -2.0
    with tempfile.TemporaryDirectory(prefix="lka_ocr_modes_") as temp:
        for psm in ("6", "11", "4", "3"):
            output_base = Path(temp) / f"psm{psm}"
            # One run writes both the text and a TSV with per-word confidence. The settings are
            # passed with -c: the named "tsv" config is not found when --tessdata-dir points at
            # the project's own language data.
            command = [
                tesseract, str(path), str(output_base), "-l", available_ocr_languages(), "--psm", psm,
                "-c", "tessedit_create_txt=1", "-c", "tessedit_create_tsv=1",
            ]
            if tessdata_dir:
                command.extend(["--tessdata-dir", str(tessdata_dir)])
            proc = subprocess.run(command, check=False, capture_output=True)
            text_path = output_base.with_name(output_base.name + ".txt")
            if proc.returncode != 0 or not text_path.exists():
                continue
            text = text_path.read_text(encoding="utf-8", errors="ignore").strip()
            confidence = tsv_mean_confidence(output_base.with_name(output_base.name + ".tsv"))
            # Keep the mode Tesseract is surest of. Text length is misleading: sparse-text mode (11)
            # splits lines into fragments and looks longer while dropping words. A tie keeps the
            # earlier mode.
            if text and confidence > best_confidence:
                best_text, best_confidence = text, confidence
    return close_cjk_gaps(best_text)


# Tesseract's Chinese output puts a space between characters ("过 去 两 年"), which breaks phrase
# search and reads badly when quoted; spaces next to Latin words and numbers stay.
_CJK = r"　-〿㐀-䶿一-鿿＀-￯"
_CJK_GAP = re.compile(rf"(?<=[{_CJK}])[ \t]+(?=[{_CJK}])")


def close_cjk_gaps(text: str) -> str:
    return _CJK_GAP.sub("", text)


def ocr_image_bytes(name: str, data: bytes) -> str:
    suffix = Path(name).suffix.lower()
    if suffix not in OCR_IMAGE_EXTENSIONS:
        return ""
    with tempfile.TemporaryDirectory(prefix="lka_image_ocr_") as temp:
        image_path = Path(temp) / f"image{suffix}"
        image_path.write_bytes(data)
        return ocr_image(image_path)


def ocr_pdf(path: Path, pages: list[int] | None = None) -> tuple[str, list[str]]:
    """OCR text of the PDF's pages, or only of `pages` (1-based), each marked with its page number."""
    warnings: list[str] = []
    if not find_tesseract():
        warnings.append("OCR skipped: tesseract executable is not installed or not on PATH.")
        return "", warnings
    if not pdf_page_renderer():
        warnings.append("OCR skipped: no PDF page renderer; install the Python package pypdfium2 or put Poppler's pdftoppm on PATH.")
        return "", warnings

    pieces: list[str] = []
    with tempfile.TemporaryDirectory(prefix="lka_ocr_") as temp:
        try:
            images = render_pdf_pages_for_ocr(path, Path(temp), pages)
        except Exception as exc:
            warnings.append(f"OCR render failed: {portable_error_message(exc)}")
            return "", warnings
        for done, image_path in enumerate(images):
            progress.report("ocr", done=done, total=len(images))
            page_index = int(re.sub(r"\D", "", image_path.stem) or 0)
            page_text = ocr_image(image_path)
            if page_text:
                pieces.append(f"[ocr page {page_index}]\n{page_text}")
            else:
                warnings.append(f"OCR returned no text for page {page_index}.")
    return "\n\n".join(pieces).strip(), warnings
