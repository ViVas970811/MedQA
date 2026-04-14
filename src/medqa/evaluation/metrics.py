"""Evaluation metrics for the MedQA pipeline."""

from __future__ import annotations

import pandas as pd


def exact_match(gold: str, pred: str) -> bool:
    gold = "" if pd.isna(gold) else str(gold)
    pred = "" if pd.isna(pred) else str(pred)
    return gold.strip().lower() == pred.strip().lower()


def token_f1(gold: str, pred: str) -> float:
    gold = "" if pd.isna(gold) else str(gold)
    pred = "" if pd.isna(pred) else str(pred)

    gold_tokens = set(gold.lower().split())
    pred_tokens = set(pred.lower().split())

    if not gold_tokens and not pred_tokens:
        return 1.0
    if not gold_tokens or not pred_tokens:
        return 0.0

    tp = len(gold_tokens & pred_tokens)
    fp = len(pred_tokens - gold_tokens)
    fn = len(gold_tokens - pred_tokens)

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    return 2 * precision * recall / (precision + recall + 1e-9)


def compute_symptom_metrics(
    gold_df: pd.DataFrame,
    pred_df: pd.DataFrame,
) -> dict:
    """Compute exact-match accuracy and token F1 for symptom extraction."""
    fields = ["symptom", "body_location", "duration", "trigger"]

    accuracy = {}
    f1_scores = {}

    for field in fields:
        gold_col = gold_df[field] if field in gold_df.columns else pd.Series([""] * len(gold_df))
        pred_col = pred_df.get(f"pred_{field}", pred_df.get(field, pd.Series([""] * len(pred_df))))

        acc = sum(exact_match(g, p) for g, p in zip(gold_col, pred_col)) / max(len(gold_col), 1)
        f1 = sum(token_f1(g, p) for g, p in zip(gold_col, pred_col)) / max(len(gold_col), 1)

        accuracy[f"{field}_accuracy"] = round(acc, 4)
        f1_scores[f"{field}_f1"] = round(f1, 4)

    return {"accuracy": accuracy, "token_f1": f1_scores}
