"""Intent classification pipeline stage."""

from __future__ import annotations

from medqa.config import Settings, get_settings
from medqa.log import get_logger
from medqa.models.llm import LLMClient
from medqa.models.schemas import INTENT_MERGE_MAP, IntentCategory, IntentResult

logger = get_logger(__name__)

INTENT_LABELS = sorted(set(INTENT_MERGE_MAP.values()))


def _build_prompt(question: str) -> str:
    label_list = ", ".join(INTENT_LABELS)
    return (
        "You are an expert medical question classifier.\n\n"
        "Classify the user's question into EXACTLY ONE of the following intent categories:\n\n"
        f"{label_list}\n\n"
        "Return ONLY valid JSON in this format:\n"
        '{\n  "intent": "one_of_the_categories"\n}\n\n'
        f'User question: "{question}"'
    )


class IntentClassifier:
    """Zero-shot LLM-based intent classification."""

    def __init__(self, llm: LLMClient | None = None, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._llm = llm or LLMClient(self._settings)

    def classify(self, question: str) -> IntentResult:
        prompt = _build_prompt(question)
        try:
            result = self._llm.complete_json(
                prompt,
                model=self._settings.llm.intent_model,
            )
            raw_intent = result.get("intent", "unknown")
            merged = INTENT_MERGE_MAP.get(raw_intent, raw_intent)

            valid = {e.value for e in IntentCategory}
            if merged not in valid:
                merged = IntentCategory.UNKNOWN.value

            logger.info("Intent classified: %s -> %s", raw_intent, merged)
            return IntentResult(intent=merged)

        except Exception as exc:
            logger.error("Intent classification failed: %s", exc)
            return IntentResult(intent=IntentCategory.UNKNOWN.value)
