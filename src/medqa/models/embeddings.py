"""Embedding model manager with lazy loading."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from medqa.config import Settings, get_settings
from medqa.log import get_logger

logger = get_logger(__name__)


class EmbeddingModel:
    """Manages the sentence-transformer embedding model."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            cfg = self._settings.embeddings
            logger.info("Loading embedding model: %s", cfg.model_name)
            self._model = SentenceTransformer(cfg.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        return self._settings.embeddings.dimension

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int | None = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        bs = batch_size or self._settings.embeddings.batch_size
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
            batch_size=bs,
        )
        return embeddings.astype(np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode([text])
