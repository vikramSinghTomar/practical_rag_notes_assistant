from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError


class AppConfig(BaseModel):
    data_dir: Path = Field(default=Path("data"))
    index_dir: Path = Field(default=Path("index"))
    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2"
    )
    openai_model: str = Field(default="gpt-4o-mini")
    top_k: int = Field(default=5, ge=1)
    max_context_chars: int = Field(default=6000, ge=1000)

    @property
    def data_dir_resolved(self) -> Path:
        return self.data_dir.resolve()

    @property
    def index_dir_resolved(self) -> Path:
        return self.index_dir.resolve()


def load_config(path: Optional[Path] = None) -> AppConfig:
    """
    Load configuration from a YAML file.

    If `path` is None, looks for `config.yaml` in the current working directory.
    Also loads environment variables from a `.env` file if present.
    """
    load_dotenv()

    if path is None:
        path = Path("config.yaml")

    if not path.exists():
        # Fall back to defaults if no config file is present.
        return AppConfig()

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    try:
        cfg = AppConfig(**raw)
    except ValidationError as e:
        raise SystemExit(f"Invalid configuration in {path}:\n{e}") from e

    # Ensure directories exist
    cfg.data_dir_resolved.mkdir(parents=True, exist_ok=True)
    cfg.index_dir_resolved.mkdir(parents=True, exist_ok=True)
    return cfg


__all__ = ["AppConfig", "load_config"]


