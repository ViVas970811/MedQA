"""Symptom extraction pipeline stage."""

from __future__ import annotations

from medqa.config import Settings, get_settings
from medqa.log import get_logger
from medqa.models.llm import LLMClient
from medqa.models.schemas import BODY_GROUP_MEMBERS, SymptomExtraction

logger = get_logger(__name__)


def _map_body_location(loc: str) -> str:
    if not loc:
        return "other"
    loc = loc.lower().strip()
    for group, members in BODY_GROUP_MEMBERS.items():
        if loc in members:
            return group
    return "other"


def _build_prompt(question: str) -> str:
    return (
        "You are a medical symptom extraction system.\n"
        "Extract the following fields from the user's question:\n\n"
        '- "symptom": short phrase describing what the user is experiencing\n'
        '- "body_location": the explicit body part mentioned (use lowercase)\n'
        '- "duration": when or how long the symptom occurs (if present)\n'
        '- "trigger": what causes, worsens, or relates to the symptom (if present)\n\n'
        "Return ONLY valid JSON. Do NOT add commentary.\n\n"
        "Examples:\n\n"
        'Q: "Why do my eyes feel dry in the morning?"\n'
        'A: {"symptom": "dry eyes", "body_location": "eyes", "duration": "in the morning", "trigger": ""}\n\n'
        'Q: "I get knee pain when running."\n'
        'A: {"symptom": "knee pain", "body_location": "knee", "duration": "", "trigger": "running"}\n\n'
        'Q: "My chest feels tight when I wake up."\n'
        'A: {"symptom": "chest tightness", "body_location": "chest", "duration": "when I wake up", "trigger": ""}\n\n'
        "Now extract for this question:\n\n"
        f'Q: "{question}"\nA:'
    )


class SymptomExtractor:
    """Few-shot LLM-based symptom extraction."""

    def __init__(self, llm: LLMClient | None = None, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._llm = llm or LLMClient(self._settings)

    def extract(self, question: str) -> SymptomExtraction:
        prompt = _build_prompt(question)
        try:
            data = self._llm.complete_json(
                prompt,
                model=self._settings.llm.symptom_model,
            )
            body_loc = data.get("body_location", "")
            group = _map_body_location(body_loc)

            result = SymptomExtraction(
                symptom=data.get("symptom", ""),
                body_location=body_loc,
                duration=data.get("duration", ""),
                trigger=data.get("trigger", ""),
                body_location_group=group,
            )
            logger.info("Extracted symptom: %s (%s)", result.symptom, result.body_location_group)
            return result

        except Exception as exc:
            logger.error("Symptom extraction failed: %s", exc)
            return SymptomExtraction()
