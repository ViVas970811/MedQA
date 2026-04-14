"""FAISS-backed vector store for semantic retrieval."""

from __future__ import annotations

import numpy as np
import faiss

from medqa.log import get_logger
from medqa.models.schemas import RetrievalResult

logger = get_logger(__name__)


class VectorStore:
    """FAISS vector index for efficient similarity search."""

    def __init__(self) -> None:
        self._index: faiss.IndexFlatL2 | None = None
        self._texts: list[str] = []

    @property
    def size(self) -> int:
        return self._index.ntotal if self._index else 0

    @property
    def is_built(self) -> bool:
        return self._index is not None and self._index.ntotal > 0

    def build(self, texts: list[str], embeddings: np.ndarray) -> None:
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")

        dimension = embeddings.shape[1]
        self._index = faiss.IndexFlatL2(dimension)
        self._index.add(embeddings.astype(np.float32))
        self._texts = list(texts)
        logger.info("Built FAISS index: %d vectors, dim=%d", self.size, dimension)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[RetrievalResult]:
        if not self.is_built:
            raise RuntimeError("Vector store not built. Call build() first.")

        query_embedding = query_embedding.astype(np.float32)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        k = min(k, self.size)
        distances, indices = self._index.search(query_embedding, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self._texts):
                results.append(RetrievalResult(
                    question=self._texts[idx],
                    score=float(dist),
                ))
        return results
