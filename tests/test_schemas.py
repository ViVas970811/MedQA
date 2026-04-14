"""Tests for Pydantic data models."""

import pytest
from medqa.models.schemas import (
    BODY_GROUP_MEMBERS,
    INTENT_MERGE_MAP,
    IntentCategory,
    IntentResult,
    PipelineRequest,
    RetrievalResult,
    SymptomExtraction,
)


class TestIntentCategory:
    def test_all_merge_targets_are_valid(self):
        valid = {e.value for e in IntentCategory}
        for target in INTENT_MERGE_MAP.values():
            assert target in valid, f"{target} is not a valid IntentCategory"

    def test_merge_map_covers_original_labels(self):
        originals = [
            "symptom_centric_query", "severity_question",
            "disease_general_health_information", "prognosis_inquiry",
            "treatment_question", "causality_question",
            "transmission_question", "diagnosis_decision_question",
        ]
        for orig in originals:
            assert orig in INTENT_MERGE_MAP


class TestSymptomExtraction:
    def test_defaults(self):
        s = SymptomExtraction()
        assert s.symptom == ""
        assert s.body_location_group == "other"

    def test_custom(self):
        s = SymptomExtraction(
            symptom="chest pain",
            body_location="chest",
            body_location_group="chest",
        )
        assert s.symptom == "chest pain"


class TestBodyGroups:
    def test_all_groups_nonempty(self):
        for group, members in BODY_GROUP_MEMBERS.items():
            assert len(members) > 0, f"Group {group} is empty"

    def test_no_duplicate_members(self):
        all_members = []
        for members in BODY_GROUP_MEMBERS.values():
            all_members.extend(members)
        assert len(all_members) == len(set(all_members)), "Duplicate body part members found"


class TestPipelineRequest:
    def test_valid(self):
        req = PipelineRequest(question="What causes headaches?")
        assert req.top_k == 5

    def test_custom_k(self):
        req = PipelineRequest(question="Test", top_k=10)
        assert req.top_k == 10

    def test_min_length(self):
        with pytest.raises(Exception):
            PipelineRequest(question="ab")


class TestRetrievalResult:
    def test_default_score(self):
        r = RetrievalResult(question="test")
        assert r.score == 0.0
