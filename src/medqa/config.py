"""Centralized configuration management using Pydantic settings."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
CONFIG_DIR = ROOT_DIR / "configs"


class LLMConfig(BaseModel):
    provider: str = "groq"
    intent_model: str = "llama-3.3-70b-versatile"
    generation_model: str = "llama-3.1-8b-instant"
    symptom_model: str = "llama-3.1-8b-instant"
    temperature: float = 0.0
    max_retries: int = 3
    retry_delay: float = 0.5
    rate_limit_delay: float = 0.15


class EmbeddingsConfig(BaseModel):
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int = 384
    batch_size: int = 128


class RetrievalConfig(BaseModel):
    top_k: int = 5
    index_type: str = "flat_l2"


class DataConfig(BaseModel):
    corpus_path: str = "data/corpus.json"
    labels_path: str = "data/labels.json"


class EvaluationConfig(BaseModel):
    sample_size: int = 30
    random_state: int = 42
    test_size: float = 0.3


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:8501", "http://localhost:3000"]
    )


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "json"


class Settings(BaseSettings):
    groq_api_key: SecretStr = Field(default=...)
    medqa_env: str = "development"
    log_level: str = "INFO"

    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    model_config = {"env_file": str(ROOT_DIR / ".env"), "extra": "ignore"}


def _load_yaml_config(env: str = "base") -> dict:
    path = CONFIG_DIR / f"{env}.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


@lru_cache
def get_settings() -> Settings:
    """Load settings from YAML config overlaid with environment variables."""
    env = os.getenv("MEDQA_ENV", "base")
    yaml_cfg = _load_yaml_config("base")
    if env != "base":
        overlay = _load_yaml_config(env)
        yaml_cfg = {**yaml_cfg, **overlay}

    nested = {}
    for key in ("llm", "embeddings", "retrieval", "data", "evaluation", "api", "logging"):
        if key in yaml_cfg:
            nested[key] = yaml_cfg[key]

    return Settings(**nested)
