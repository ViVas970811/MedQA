"""Pydantic data models for the MedQA pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Intent categories
# ---------------------------------------------------------------------------

class IntentCategory(str, Enum):
    TREATMENT = "treatment_question"
    SYMPTOM = "symptom_question"
    DISEASE_INFO = "disease_information"
    PROGNOSIS = "prognosis_question"
    CAUSALITY = "causality_question"
    TRANSMISSION = "transmission_question"
    DIAGNOSIS = "diagnosis_decision_question"
    SEVERITY = "severity_question"
    ACTIVITY = "activity_question"
    ABILITY = "ability_question"
    UNKNOWN = "unknown"


INTENT_MERGE_MAP: dict[str, str] = {
    "symptom_centric_query": IntentCategory.SYMPTOM.value,
    "severity_question": IntentCategory.SYMPTOM.value,
    "disease_general_health_information": IntentCategory.DISEASE_INFO.value,
    "prognosis_inquiry": IntentCategory.PROGNOSIS.value,
    "treatment_question": IntentCategory.TREATMENT.value,
    "causality_question": IntentCategory.CAUSALITY.value,
    "transmission_question": IntentCategory.TRANSMISSION.value,
    "diagnosis_decision_question": IntentCategory.DIAGNOSIS.value,
    "activity_question": IntentCategory.ACTIVITY.value,
    "ability_question": IntentCategory.ABILITY.value,
}


# ---------------------------------------------------------------------------
# Body location groups
# ---------------------------------------------------------------------------

class BodyGroup(str, Enum):
    HEAD = "head"
    NECK_THROAT = "neck_throat"
    CHEST = "chest"
    ABDOMEN = "abdomen"
    BACK_SPINE = "back_spine"
    UPPER_LIMB = "upper_limb"
    SKIN_HAIR = "skin_hair"
    CIRCULATORY = "circulatory_blood"
    GENERAL = "general_body"
    OTHER = "other"


BODY_GROUP_MEMBERS: dict[str, list[str]] = {
    "head": [
        "brain", "head", "face", "eye", "eyes", "lips", "mouth",
        "nose", "gums", "tongue", "teeth", "salivary_glands", "psychological",
    ],
    "neck_throat": ["throat"],
    "chest": ["chest", "lungs", "heart"],
    "abdomen": [
        "abdomen", "bladder", "kidney", "rectum", "prostate", "uterus",
        "urinary system", "urinary_tract", "reproductive system",
        "testicle", "genital", "penis",
    ],
    "back_spine": ["back", "spine", "hips", "joints"],
    "upper_limb": ["hand"],
    "skin_hair": ["skin", "hair", "body hair", "tissue"],
    "circulatory_blood": ["blood", "systemic", "nervous system", "nervous_system"],
    "general_body": ["body"],
}


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class SymptomExtraction(BaseModel):
    symptom: str = ""
    body_location: str = ""
    duration: str = ""
    trigger: str = ""
    body_location_group: str = "other"


class IntentResult(BaseModel):
    intent: str
    confidence: Optional[float] = None


class RetrievalResult(BaseModel):
    question: str
    score: float = 0.0


class PipelineRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)


class PipelineResponse(BaseModel):
    question: str
    intent: IntentResult
    symptoms: SymptomExtraction
    retrieved_questions: list[RetrievalResult]
    answer: str
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str
    index_size: int = 0


class EvaluationResult(BaseModel):
    model_name: str
    accuracy: float
    report: dict
    sample_size: int
