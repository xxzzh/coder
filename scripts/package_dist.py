"""Build the user-facing local-knowledge package under dist/."""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = ROOT / "dist"
PACKAGE_DIR = DIST_DIR / "local-knowledge"
ZIP_PATH = DIST_DIR / "local-knowledge.zip"

ROOT_FILES = [
    ".gitignore",
    "AGENTS.md",
    "audit-local.bat",
    "install_kb_phase1.py",
    "README.md",
    "requirements.txt",
    "package-dist.bat",
    "run-agent.bat",
    "start.bat",
]

ROOT_DIRS = [
    ".codex",
    "scripts",
    "test_materials",
    "tools",
]

RAW_SUBDIRS = [
    "excel",
    "markdown",
    "pdf",
    "word",
]

EXCLUDED_DIR_NAMES = {"__pycache__"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}
EXCLUDED_FILE_NAMES: set[str] = set()


def ensure_under_root(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT):
        raise RuntimeError(f"Refusing to operate outside project root: {resolved}")
    return resolved


def ignore_generated(_: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        path = Path(name)
        if name in EXCLUDED_DIR_NAMES or name in EXCLUDED_FILE_NAMES or path.suffix in EXCLUDED_SUFFIXES:
            ignored.add(name)
    return ignored


def copy_file(relative_path: str) -> None:
    source = ROOT / relative_path
    if not source.exists():
        raise FileNotFoundError(source)
    target = PACKAGE_DIR / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def copy_dir(relative_path: str) -> None:
    source = ROOT / relative_path
    if not source.exists():
        raise FileNotFoundError(source)
    target = PACKAGE_DIR / relative_path
    shutil.copytree(source, target, ignore=ignore_generated)


def build() -> None:
    ensure_under_root(DIST_DIR).mkdir(exist_ok=True)
    package_dir = ensure_under_root(PACKAGE_DIR)
    zip_path = ensure_under_root(ZIP_PATH)

    if package_dir.exists():
        shutil.rmtree(package_dir)
    if zip_path.exists():
        zip_path.unlink()

    package_dir.mkdir(parents=True)
    for relative_path in ROOT_FILES:
        copy_file(relative_path)
    for relative_path in ROOT_DIRS:
        copy_dir(relative_path)

    raw_dir = package_dir / "knowledge_base" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for subdir in RAW_SUBDIRS:
        (raw_dir / subdir).mkdir(parents=True, exist_ok=True)

    shutil.make_archive(str(zip_path.with_suffix("")), "zip", DIST_DIR, PACKAGE_DIR.name)
    print(f"Built {PACKAGE_DIR}")
    print(f"Built {ZIP_PATH}")


if __name__ == "__main__":
    build()
