"""Data loading utilities for corpus and labeled datasets."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from medqa.config import ROOT_DIR, Settings, get_settings
from medqa.log import get_logger
from medqa.models.schemas import INTENT_MERGE_MAP

logger = get_logger(__name__)


class DataLoader:
    """Loads and preprocesses MedQA datasets."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def _resolve(self, relative: str) -> Path:
        return ROOT_DIR / relative

    def load_corpus(self) -> list[str]:
        path = self._resolve(self._settings.data.corpus_path)
        logger.info("Loading corpus from %s", path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        texts = [item["question"] for item in data]
        logger.info("Loaded %d corpus questions", len(texts))
        return texts

    def load_labels(self) -> pd.DataFrame:
        path = self._resolve(self._settings.data.labels_path)
        logger.info("Loading labels from %s", path)
        df = pd.read_json(path)
        df["intent_merged"] = df["intent"].map(
            lambda x: INTENT_MERGE_MAP.get(x, x)
        )
        logger.info("Loaded %d labeled questions (%d unique intents)",
                     len(df), df["intent_merged"].nunique())
        return df

    def load_gold_symptoms(self) -> list[dict]:
        return [
            {"question": "Are dry lips a symptom of anything?", "symptom": "dry lips", "body_location": "lips", "duration": "", "trigger": ""},
            {"question": "Are floaters in eye serious?", "symptom": "floaters in eye", "body_location": "eye", "duration": "", "trigger": ""},
            {"question": "Are red veins in eyes serious?", "symptom": "red veins in eyes", "body_location": "eyes", "duration": "", "trigger": ""},
            {"question": "At what age is occasional shortness of breath normal?", "symptom": "shortness of breath", "body_location": "chest", "duration": "occasional", "trigger": ""},
            {"question": "At what age is rectal bleeding common?", "symptom": "rectal bleeding", "body_location": "rectum", "duration": "", "trigger": ""},
            {"question": "Can high blood pressure cause blue lips?", "symptom": "blue lips", "body_location": "lips", "duration": "", "trigger": "high blood pressure"},
            {"question": "Can dizziness be serious?", "symptom": "dizziness", "body_location": "brain", "duration": "", "trigger": ""},
            {"question": "Can dry eye syndrome be fixed?", "symptom": "dry eye syndrome", "body_location": "eye", "duration": "", "trigger": ""},
            {"question": "Can rectal bleeding be serious?", "symptom": "rectal bleeding", "body_location": "rectum", "duration": "", "trigger": ""},
            {"question": "Can runny nose be a symptom of Covid?", "symptom": "runny nose", "body_location": "nose", "duration": "", "trigger": "Covid infection"},
        ]
