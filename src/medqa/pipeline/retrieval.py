"""Semantic retrieval pipeline stage."""

from __future__ import annotations

from medqa.config import Settings, get_settings
from medqa.data.vectorstore import VectorStore
from medqa.log import get_logger
from medqa.models.embeddings import EmbeddingModel
from medqa.models.schemas import RetrievalResult

logger = get_logger(__name__)


class Retriever:
    """Retrieves semantically similar medical questions from the corpus."""

    def __init__(
        self,
        embedder: EmbeddingModel | None = None,
        store: VectorStore | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._embedder = embedder or EmbeddingModel(self._settings)
        self._store = store or VectorStore()
        self._corpus_loaded = False

    @property
    def index_size(self) -> int:
        return self._store.size

    @property
    def is_ready(self) -> bool:
        return self._store.is_built

    def build_index(self, corpus_texts: list[str]) -> None:
        logger.info("Building index for %d documents...", len(corpus_texts))
        embeddings = self._embedder.encode(corpus_texts, show_progress=True)
        self._store.build(corpus_texts, embeddings)
        self._corpus_loaded = True
        logger.info("Index ready: %d vectors", self._store.size)

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievalResult]:
        if not self._store.is_built:
            raise RuntimeError("Index not built. Call build_index() first.")

        k = k or self._settings.retrieval.top_k
        query_emb = self._embedder.encode_query(query)
        results = self._store.search(query_emb, k=k)
        logger.info("Retrieved %d results for query", len(results))
        return results
