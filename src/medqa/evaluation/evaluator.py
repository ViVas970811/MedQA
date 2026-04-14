"""End-to-end evaluation runner for the MedQA pipeline."""

from __future__ import annotations

import time

import pandas as pd
from sklearn.metrics import classification_report

from medqa.config import Settings, get_settings
from medqa.data.loader import DataLoader
from medqa.evaluation.baselines import BaselineEvaluator
from medqa.evaluation.metrics import compute_symptom_metrics
from medqa.log import get_logger
from medqa.models.llm import LLMClient
from medqa.models.schemas import INTENT_MERGE_MAP
from medqa.pipeline.intent import IntentClassifier
from medqa.pipeline.symptoms import SymptomExtractor

logger = get_logger(__name__)


class PipelineEvaluator:
    """Evaluates individual pipeline stages and baselines."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._loader = DataLoader(self._settings)

    def evaluate_intent(self, sample_size: int | None = None) -> dict:
        """Evaluate LLM intent classification against labeled data."""
        cfg = self._settings.evaluation
        sample_size = sample_size or cfg.sample_size

        df = self._loader.load_labels()
        df_eval = df.sample(sample_size, random_state=cfg.random_state).reset_index(drop=True)

        classifier = IntentClassifier(settings=self._settings)
        preds = []

        logger.info("Evaluating intent classification on %d samples...", sample_size)
        for question in df_eval["question"]:
            result = classifier.classify(question)
            raw = result.intent
            merged = INTENT_MERGE_MAP.get(raw, raw)
            preds.append(merged)
            time.sleep(self._settings.llm.rate_limit_delay)

        df_eval["pred_intent"] = preds
        labels = sorted(df_eval["intent_merged"].unique())

        report = classification_report(
            df_eval["intent_merged"],
            df_eval["pred_intent"],
            labels=labels,
            output_dict=True,
            zero_division=0,
        )

        accuracy = report.get("accuracy", 0.0)
        logger.info("LLM intent accuracy: %.2f%%", accuracy * 100)

        return {
            "name": "LLM (Llama-3.3-70B)",
            "accuracy": accuracy,
            "report": report,
            "sample_size": sample_size,
        }

    def evaluate_symptoms(self) -> dict:
        """Evaluate symptom extraction against gold-standard data."""
        gold_data = self._loader.load_gold_symptoms()
        gold_df = pd.DataFrame(gold_data)

        extractor = SymptomExtractor(settings=self._settings)
        preds = []
        for item in gold_data:
            result = extractor.extract(item["question"])
            preds.append(result.model_dump())

        pred_df = pd.DataFrame(preds).add_prefix("pred_")
        metrics = compute_symptom_metrics(gold_df, pred_df)

        logger.info("Symptom extraction metrics: %s", metrics)
        return metrics

    def evaluate_baselines(self) -> list[dict]:
        """Run all baseline classifiers."""
        df = self._loader.load_labels()
        evaluator = BaselineEvaluator(
            df,
            test_size=self._settings.evaluation.test_size,
            random_state=self._settings.evaluation.random_state,
        )
        return evaluator.run_all()
