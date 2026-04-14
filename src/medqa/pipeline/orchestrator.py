"""Pipeline orchestrator that coordinates all stages."""

from __future__ import annotations

import time

from medqa.config import Settings, get_settings
from medqa.data.loader import DataLoader
from medqa.log import get_logger
from medqa.models.embeddings import EmbeddingModel
from medqa.models.llm import LLMClient
from medqa.models.schemas import PipelineResponse, PipelineRequest
from medqa.pipeline.generation import AnswerGenerator
from medqa.pipeline.intent import IntentClassifier
from medqa.pipeline.retrieval import Retriever
from medqa.pipeline.symptoms import SymptomExtractor

logger = get_logger(__name__)


class MedQAPipeline:
    """End-to-end medical question answering pipeline.

    Orchestrates four stages:
        1. Intent classification
        2. Symptom extraction
        3. Semantic retrieval (RAG)
        4. Answer generation
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._llm = LLMClient(self._settings)
        self._embedder = EmbeddingModel(self._settings)

        self.intent_classifier = IntentClassifier(self._llm, self._settings)
        self.symptom_extractor = SymptomExtractor(self._llm, self._settings)
        self.retriever = Retriever(self._embedder, settings=self._settings)
        self.answer_generator = AnswerGenerator(self._llm, self._settings)

        self._initialized = False

    @property
    def is_ready(self) -> bool:
        return self._initialized and self.retriever.is_ready

    def initialize(self) -> None:
        """Load corpus and build the retrieval index."""
        if self._initialized:
            return

        logger.info("Initializing MedQA pipeline...")
        loader = DataLoader(self._settings)
        corpus = loader.load_corpus()
        self.retriever.build_index(corpus)
        self._initialized = True
        logger.info("Pipeline ready (index size: %d)", self.retriever.index_size)

    def run(self, request: PipelineRequest) -> PipelineResponse:
        """Execute the full pipeline on a question."""
        if not self.is_ready:
            self.initialize()

        start = time.perf_counter()
        question = request.question.strip()

        intent_result = self.intent_classifier.classify(question)
        symptoms = self.symptom_extractor.extract(question)
        retrieved = self.retriever.retrieve(question, k=request.top_k)
        answer = self.answer_generator.generate(question, retrieved)

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "Pipeline complete: intent=%s, time=%.0fms",
            intent_result.intent,
            elapsed_ms,
        )

        return PipelineResponse(
            question=question,
            intent=intent_result,
            symptoms=symptoms,
            retrieved_questions=retrieved,
            answer=answer,
            processing_time_ms=round(elapsed_ms, 1),
        )
