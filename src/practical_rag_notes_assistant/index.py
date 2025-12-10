from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from rich.console import Console
from sentence_transformers import SentenceTransformer

from .config import AppConfig, load_config
from .ingest import DocumentChunk, build_chunks

console = Console()


def _load_or_create_model(name: str) -> SentenceTransformer:
    console.print(f"[green]Loading embedding model:[/green] {name}")
    return SentenceTransformer(name)


def build_index(cfg: AppConfig | None = None) -> None:
    if cfg is None:
        cfg = load_config()

    chunks = build_chunks(cfg)
    if not chunks:
        console.print("[yellow]No chunks created. Aborting index build.[/yellow]")
        return

    model = _load_or_create_model(cfg.embedding_model_name)

    texts = [c.text for c in chunks]
    console.print(f"[green]Encoding {len(texts)} chunks...[/green]")
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    index_dir = cfg.index_dir_resolved
    index_dir.mkdir(parents=True, exist_ok=True)

    faiss_path = index_dir / "faiss.index"
    meta_path = index_dir / "chunks.pkl"
    config_snapshot_path = index_dir / "config_snapshot.json"

    console.print(f"[green]Saving index to:[/green] {faiss_path}")
    faiss.write_index(index, str(faiss_path))

    console.print(f"[green]Saving chunk metadata to:[/green] {meta_path}")
    with meta_path.open("wb") as f:
        pickle.dump(chunks, f)

    with config_snapshot_path.open("w", encoding="utf-8") as f:
        json.dump(cfg.model_dump(), f, indent=2, default=str)

    console.print("[bold green]Index build complete.[/bold green]")


def load_index(cfg: AppConfig | None = None) -> Tuple[faiss.IndexFlatIP, List[DocumentChunk], SentenceTransformer]:
    if cfg is None:
        cfg = load_config()

    index_dir = cfg.index_dir_resolved
    faiss_path = index_dir / "faiss.index"
    meta_path = index_dir / "chunks.pkl"

    if not faiss_path.exists() or not meta_path.exists():
        raise SystemExit(
            f"Index not found in {index_dir}. Run 'python -m practical_rag_notes_assistant.cli ingest' first."
        )

    console.print(f"[green]Loading FAISS index from:[/green] {faiss_path}")
    index = faiss.read_index(str(faiss_path))

    console.print(f"[green]Loading chunk metadata from:[/green] {meta_path}")
    with meta_path.open("rb") as f:
        chunks: List[DocumentChunk] = pickle.load(f)

    model = _load_or_create_model(cfg.embedding_model_name)

    return index, chunks, model


def search(
    query: str,
    cfg: AppConfig | None = None,
    top_k: int | None = None,
) -> List[Tuple[float, DocumentChunk]]:
    if cfg is None:
        cfg = load_config()

    index, chunks, model = load_index(cfg)
    if top_k is None:
        top_k = cfg.top_k

    console.print(f"[green]Searching for:[/green] {query!r}")

    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, top_k)

    results: List[Tuple[float, DocumentChunk]] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        results.append((float(score), chunks[int(idx)]))

    return results


__all__ = ["build_index", "load_index", "search"]


