"""Answer generation pipeline stage."""

from __future__ import annotations

from medqa.config import Settings, get_settings
from medqa.log import get_logger
from medqa.models.llm import LLMClient
from medqa.models.schemas import RetrievalResult

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful medical question answering assistant. "
    "You answer strictly medical-related queries. For non-medical questions, "
    'respond with "I can only answer medical-related queries." '
    "Always include a disclaimer that users should consult a healthcare professional."
)


def _build_prompt(question: str, retrieved: list[RetrievalResult]) -> str:
    context = "\n".join(f"- {r.question}" for r in retrieved)
    return (
        "Use the retrieved similar questions below to provide a medically safe "
        "and helpful answer. If context is insufficient, say more information is needed.\n\n"
        f"Retrieved similar questions:\n{context}\n\n"
        f"User question: {question}\n\n"
        "Provide your best answer:"
    )


class AnswerGenerator:
    """Generates answers using retrieved context and an LLM."""

    def __init__(self, llm: LLMClient | None = None, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._llm = llm or LLMClient(self._settings)

    def generate(self, question: str, retrieved: list[RetrievalResult]) -> str:
        prompt = _build_prompt(question, retrieved)
        try:
            answer = self._llm.complete(
                prompt,
                model=self._settings.llm.generation_model,
                system=SYSTEM_PROMPT,
            )
            logger.info("Generated answer (%d chars)", len(answer))
            return answer
        except Exception as exc:
            logger.error("Answer generation failed: %s", exc)
            return "An error occurred while generating the answer. Please try again."
