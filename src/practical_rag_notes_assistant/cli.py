from __future__ import annotations

import argparse

from rich.console import Console

from .config import load_config
from .index import build_index
from .query import answer_question

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Practical RAG Notes Assistant - build an index and ask questions over your notes."
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest documents and build the FAISS index.",
    )
    ingest_parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to a config YAML file (default: config.yaml).",
    )

    ask_parser = subparsers.add_parser(
        "ask",
        help="Ask a question using the existing index.",
    )
    ask_parser.add_argument("question", type=str, help="Question to ask over your notes.")
    ask_parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to a config YAML file (default: config.yaml).",
    )

    args = parser.parse_args()

    cfg = load_config()

    if args.command == "ingest":
        console.print("[bold green]Building index...[/bold green]")
        build_index(cfg)
    elif args.command == "ask":
        answer_question(args.question, cfg)
    else:  # pragma: no cover - defensive
        parser.print_help()


if __name__ == "__main__":
    main()


