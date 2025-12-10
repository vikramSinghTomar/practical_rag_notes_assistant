from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from pypdf import PdfReader
from rich.console import Console
from rich.progress import Progress

from .config import AppConfig, load_config

console = Console()


@dataclass
class DocumentChunk:
    text: str
    source: str
    page: int | None
    chunk_id: int


def load_markdown(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n\n".join(pages)


def chunk_text(
    text: str,
    source: str,
    page: int | None,
    chunk_size: int = 800,
    chunk_overlap: int = 200,
) -> List[DocumentChunk]:
    chunks: List[DocumentChunk] = []
    cleaned = text.replace("\r\n", "\n").strip()
    if not cleaned:
        return chunks

    start = 0
    chunk_id = 0
    while start < len(cleaned):
        end = start + chunk_size
        chunk_text_str = cleaned[start:end]
        chunk_text_str = textwrap.dedent(chunk_text_str).strip()
        if chunk_text_str:
            chunks.append(
                DocumentChunk(
                    text=chunk_text_str,
                    source=source,
                    page=page,
                    chunk_id=chunk_id,
                )
            )
            chunk_id += 1
        start += chunk_size - chunk_overlap
    return chunks


def iter_files(data_dir: Path) -> Iterable[Path]:
    for path in data_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".md", ".pdf"}:
            yield path


def build_chunks(cfg: AppConfig | None = None) -> List[DocumentChunk]:
    if cfg is None:
        cfg = load_config()

    data_dir = cfg.data_dir_resolved
    if not data_dir.exists():
        console.print(f"[red]Data directory not found:[/red] {data_dir}")
        return []

    files = list(iter_files(data_dir))
    if not files:
        console.print(f"[yellow]No .md or .pdf files found in {data_dir}[/yellow]")
        return []

    chunks: List[DocumentChunk] = []

    console.print(f"[green]Loading documents from:[/green] {data_dir}")
    with Progress() as progress:
        task = progress.add_task("Reading & chunking documents...", total=len(files))
        for path in files:
            progress.update(task, description=f"Processing {path.name}")
            try:
                if path.suffix.lower() == ".md":
                    text = load_markdown(path)
                    chunks.extend(
                        chunk_text(
                            text=text, source=str(path), page=None, chunk_size=800, chunk_overlap=200
                        )
                    )
                elif path.suffix.lower() == ".pdf":
                    reader = PdfReader(str(path))
                    for i, page in enumerate(reader.pages):
                        try:
                            page_text = page.extract_text() or ""
                        except Exception:
                            page_text = ""
                        if not page_text.strip():
                            continue
                        chunks.extend(
                            chunk_text(
                                text=page_text,
                                source=str(path),
                                page=i + 1,
                                chunk_size=800,
                                chunk_overlap=200,
                            )
                        )
            except Exception as exc:  # pragma: no cover - defensive
                console.print(f"[red]Failed to process {path}: {exc}[/red]")
            finally:
                progress.update(task, advance=1)

    console.print(f"[green]Created {len(chunks)} text chunks.[/green]")
    return chunks


__all__ = ["DocumentChunk", "build_chunks"]


