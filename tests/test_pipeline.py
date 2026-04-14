"""Tests for pipeline components (offline, no LLM calls)."""

import numpy as np
import pytest

from medqa.data.vectorstore import VectorStore
from medqa.evaluation.metrics import exact_match, token_f1
from medqa.models.embeddings import EmbeddingModel
from medqa.models.llm import parse_json_response


class TestParseJsonResponse:
    def test_clean_json(self):
        result = parse_json_response('{"intent": "treatment_question"}')
        assert result["intent"] == "treatment_question"

    def test_json_with_markdown(self):
        raw = '```json\n{"intent": "symptom_question"}\n```'
        result = parse_json_response(raw)
        assert result["intent"] == "symptom_question"

    def test_json_with_surrounding_text(self):
        raw = 'Here is the result:\n{"intent": "causality_question"}\nDone.'
        result = parse_json_response(raw)
        assert result["intent"] == "causality_question"

    def test_invalid_json(self):
        result = parse_json_response("not json at all")
        assert result == {}


class TestVectorStore:
    def test_build_and_search(self, sample_corpus):
        store = VectorStore()
        dim = 8
        embeddings = np.random.randn(len(sample_corpus), dim).astype(np.float32)
        store.build(sample_corpus, embeddings)

        assert store.is_built
        assert store.size == len(sample_corpus)

        query = np.random.randn(1, dim).astype(np.float32)
        results = store.search(query, k=3)
        assert len(results) == 3
        assert all(r.question in sample_corpus for r in results)

    def test_search_before_build_raises(self):
        store = VectorStore()
        with pytest.raises(RuntimeError):
            store.search(np.zeros((1, 8), dtype=np.float32))

    def test_invalid_embeddings_shape(self):
        store = VectorStore()
        with pytest.raises(ValueError):
            store.build(["test"], np.zeros(8, dtype=np.float32))


class TestMetrics:
    def test_exact_match_identical(self):
        assert exact_match("chest pain", "chest pain") is True

    def test_exact_match_case_insensitive(self):
        assert exact_match("Chest Pain", "chest pain") is True

    def test_exact_match_different(self):
        assert exact_match("chest pain", "headache") is False

    def test_token_f1_perfect(self):
        assert token_f1("chest pain", "chest pain") == pytest.approx(1.0, abs=0.01)

    def test_token_f1_partial(self):
        score = token_f1("chest pain severe", "chest pain")
        assert 0.5 < score < 1.0

    def test_token_f1_no_overlap(self):
        assert token_f1("chest pain", "headache nausea") == pytest.approx(0.0, abs=0.01)

    def test_token_f1_both_empty(self):
        assert token_f1("", "") == 1.0
