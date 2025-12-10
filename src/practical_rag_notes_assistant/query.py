from __future__ import annotations

import os
from textwrap import shorten
from typing import List

from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

from .config import AppConfig, load_config
from .index import search

console = Console()


def _build_prompt(question: str, contexts: List[str]) -> str:
    header = (
        "You are a helpful assistant answering questions based only on the provided context.\n"
        "If the answer is not clearly present, say you are not sure rather than guessing.\n\n"
    )
    joined_contexts = "\n\n---\n\n".join(contexts)
    prompt = f"{header}Context:\n{joined_contexts}\n\nUser question: {question}\nAnswer:"
    return prompt


def answer_question(question: str, cfg: AppConfig | None = None) -> None:
    # Ensure .env is loaded even if config loading is bypassed elsewhere.
    load_dotenv()

    if cfg is None:
        cfg = load_config()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set. Put it in a .env file or environment variable.")

    results = search(question, cfg=cfg)
    if not results:
        console.print("[yellow]No results found. Did you build the index and add documents?[/yellow]")
        return

    # Build context from top chunks, staying within a character budget.
    contexts: List[str] = []
    total_chars = 0
    for score, chunk in results:
        snippet = chunk.text.strip()
        if not snippet:
            continue
        if total_chars + len(snippet) > cfg.max_context_chars:
            break
        contexts.append(snippet)
        total_chars += len(snippet)

    prompt = _build_prompt(question, contexts)

    client = OpenAI()

    console.print("[green]Calling OpenAI...[/green]")
    response = client.chat.completions.create(
        model=cfg.openai_model,
        messages=[
            {"role": "system", "content": "You are a careful assistant for question answering over notes."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    answer = response.choices[0].message.content or ""

    console.rule("[bold green]Answer[/bold green]")
    console.print(answer.strip())

    console.rule("[bold blue]Retrieved Chunks[/bold blue]")
    for score, chunk in results:
        preview = shorten(chunk.text.replace("\n", " "), width=180, placeholder="...")
        meta = f"{Path(chunk.source).name}"
        if chunk.page is not None:
            meta += f" (p.{chunk.page})"
        console.print(
            Panel(
                preview,
                title=f"{meta}",
                subtitle=f"score={score:.3f}",
                expand=False,
            )
        )


__all__ = ["answer_question"]


