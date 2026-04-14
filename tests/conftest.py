"""Shared test fixtures."""

import sys
from pathlib import Path

import pytest

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@pytest.fixture
def sample_questions():
    return [
        "Why does my chest feel tight when I wake up?",
        "How is strep throat treated?",
        "Can high blood pressure cause blue lips?",
        "Are floaters in the eye serious?",
        "Is occasional shortness of breath normal?",
    ]


@pytest.fixture
def sample_corpus():
    return [
        "What causes chest pain?",
        "How to treat strep throat?",
        "What are the symptoms of high blood pressure?",
        "When are eye floaters dangerous?",
        "What causes shortness of breath?",
        "How is bronchitis treated?",
        "What are the signs of a heart attack?",
        "Can stress cause headaches?",
        "How to relieve back pain?",
        "What causes dizziness?",
    ]
