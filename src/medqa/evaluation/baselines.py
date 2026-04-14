"""Baseline intent classifiers for comparison."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from medqa.log import get_logger

logger = get_logger(__name__)


def rule_based_classify(question: str) -> str:
    """Rule-based intent classifier using keyword matching."""
    q = question.lower()

    treatment_kw = [
        "treat", "treatment", "cure", "medication", "medicine",
        "antibiotic", "painkiller", "therapy", "home remedy", "manage",
    ]
    symptom_kw = [
        "symptom", "symptoms", "pain", "hurt", "fever", "swelling",
        "bleeding", "rash", "headache", "nausea", "cough",
        "itch", "itching", "burning", "sore",
    ]
    diagnosis_kw = ["do i have", "diagnosed", "diagnosis", "what is wrong", "should i see a doctor"]
    prognosis_kw = ["prognosis", "outlook", "recover", "recovery", "long term", "permanent", "fatal"]
    transmission_kw = ["contagious", "spread", "transmitted", "catch it", "passed on"]

    if any(k in q for k in treatment_kw):
        return "treatment_question"
    if any(k in q for k in symptom_kw):
        return "symptom_centric_query"
    if any(k in q for k in diagnosis_kw):
        return "diagnosis_decision_question"
    if any(k in q for k in prognosis_kw):
        return "prognosis_inquiry"
    if any(k in q for k in transmission_kw):
        return "transmission_question"
    if q.startswith("why ") or "cause of" in q:
        return "causality_question"
    return "disease_general_health_information"


class BaselineEvaluator:
    """Runs all baseline classifiers and collects results."""

    def __init__(self, df: pd.DataFrame, test_size: float = 0.3, random_state: int = 42):
        self.df = df
        self.test_size = test_size
        self.random_state = random_state

    def run_rule_based(self) -> dict:
        logger.info("Running rule-based baseline...")
        preds = self.df["question"].apply(rule_based_classify)
        report = classification_report(
            self.df["intent"], preds, output_dict=True, zero_division=0,
        )
        accuracy = report.get("accuracy", 0.0)
        logger.info("Rule-based accuracy: %.2f%%", accuracy * 100)
        return {"name": "Rule-Based", "accuracy": accuracy, "report": report}

    def _build_tfidf(self):
        X = self.df["question"]
        y = self.df["intent"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state,
        )
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        return vectorizer, X_train_vec, X_test_vec, y_train, y_test

    def run_random_forest(self) -> dict:
        logger.info("Running Random Forest baseline...")
        _, X_train, X_test, y_train, y_test = self._build_tfidf()
        clf = RandomForestClassifier(
            n_estimators=300, random_state=self.random_state, n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        report = classification_report(y_test, preds, output_dict=True, zero_division=0)
        accuracy = report.get("accuracy", 0.0)
        logger.info("Random Forest accuracy: %.2f%%", accuracy * 100)
        return {"name": "TF-IDF + Random Forest", "accuracy": accuracy, "report": report}

    def run_logistic_regression(self) -> dict:
        logger.info("Running Logistic Regression baseline...")
        _, X_train, X_test, y_train, y_test = self._build_tfidf()
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        report = classification_report(y_test, preds, output_dict=True, zero_division=0)
        accuracy = report.get("accuracy", 0.0)
        logger.info("Logistic Regression accuracy: %.2f%%", accuracy * 100)
        return {"name": "TF-IDF + Logistic Regression", "accuracy": accuracy, "report": report}

    def run_all(self) -> list[dict]:
        return [
            self.run_rule_based(),
            self.run_random_forest(),
            self.run_logistic_regression(),
        ]
