"""Word documents: .docx text and images, .doc conversion through Word."""

from __future__ import annotations

import base64
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from kb_ingest.ocr import ocr_image_bytes
from kb_ingest.reviews import extraction_report


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


def find_winword() -> str | None:
    found = shutil.which("WINWORD.EXE")
    if found:
        return found
    candidates: list[Path] = []
    for office_version in ("16", "15", "14"):
        candidates.extend(
            (
                Path(f"C:/Program Files/Microsoft Office/root/Office{office_version}/WINWORD.EXE"),
                Path(f"C:/Program Files (x86)/Microsoft Office/root/Office{office_version}/WINWORD.EXE"),
                Path(f"C:/Program Files/Microsoft Office/Office{office_version}/WINWORD.EXE"),
                Path(f"C:/Program Files (x86)/Microsoft Office/Office{office_version}/WINWORD.EXE"),
            )
        )
    if os.name == "nt":
        try:
            import winreg

            for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
                for office_version in ("16.0", "15.0", "14.0"):
                    for view in (winreg.KEY_WOW64_64KEY, winreg.KEY_WOW64_32KEY):
                        try:
                            key = winreg.OpenKey(
                                hive,
                                rf"SOFTWARE\Microsoft\Office\{office_version}\Common\InstallRoot",
                                0,
                                winreg.KEY_READ | view,
                            )
                            install_root, _ = winreg.QueryValueEx(key, "Path")
                            candidates.append(Path(install_root) / "WINWORD.EXE")
                            winreg.CloseKey(key)
                        except OSError:
                            continue
        except ImportError:
            pass
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def convert_doc_to_docx(path: Path, target: Path) -> None:
    if os.name != "nt":
        raise RuntimeError("旧版 .doc 提取目前需要在 Windows 上运行。")
    powershell = shutil.which("powershell") or shutil.which("powershell.exe")
    if not powershell:
        raise RuntimeError("旧版 .doc 提取需要 Windows PowerShell。")
    script = r"""
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.UTF8Encoding]::new()
$source = [IO.Path]::GetFullPath($env:LKA_DOC_SOURCE)
$target = [IO.Path]::GetFullPath($env:LKA_DOC_TARGET)
$word = $null
$document = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    try { $word.AutomationSecurity = 3 } catch {}
    $document = $word.Documents.Open($source, $false, $true)
    $document.SaveAs2($target, 16)
} finally {
    if ($document) {
        $document.Close(0)
        [Runtime.InteropServices.Marshal]::ReleaseComObject($document) | Out-Null
    }
    if ($word) {
        $word.Quit()
        [Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
"""
    encoded_script = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    env = os.environ.copy()
    env["LKA_DOC_SOURCE"] = str(path.resolve())
    env["LKA_DOC_TARGET"] = str(target.resolve())
    completed = subprocess.run(
        [powershell, "-NoProfile", "-STA", "-EncodedCommand", encoded_script],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        env=env,
    )
    if completed.returncode != 0 or not target.exists():
        detail = completed.stderr.strip() or completed.stdout.strip() or f"exit code {completed.returncode}"
        raise RuntimeError(f"Microsoft Word 转换 .doc 失败：{detail}")


def read_doc_with_report(path: Path) -> tuple[str, dict[str, Any]]:
    report = extraction_report(path, "doc")
    with tempfile.TemporaryDirectory(prefix="lka_doc_") as temp:
        converted = Path(temp) / f"{path.stem}.docx"
        convert_doc_to_docx(path, converted)
        text, converted_report = read_docx_with_report(converted)
    report["method"] = ["word_com_doc_to_docx", *converted_report["method"]]
    report["warnings"].extend(converted_report["warnings"])
    report["requires_review"] = converted_report["requires_review"]
    report["text_chars"] = len(text)
    return text, report


def read_docx_with_report(path: Path) -> tuple[str, dict[str, Any]]:
    report = extraction_report(path, "docx")
    text = read_docx(path)
    image_texts: list[str] = []
    try:
        with zipfile.ZipFile(path) as zf:
            media_files = [name for name in zf.namelist() if name.startswith("word/media/")]
            for media_file in media_files:
                media_text = ocr_image_bytes(media_file, zf.read(media_file))
                if media_text:
                    image_texts.append(f"[embedded image {Path(media_file).name}]\n{media_text}")
    except zipfile.BadZipFile:
        media_files = []
    if media_files:
        if image_texts:
            text = "\n\n".join(part for part in [text, *image_texts] if part).strip()
            report["method"].append("embedded_image_ocr")
        else:
            report["warnings"].append(
                f"DOCX contains {len(media_files)} embedded image(s), but OCR returned no image text."
            )
            report["requires_review"] = True
    report["text_chars"] = len(text)
    return text, report
