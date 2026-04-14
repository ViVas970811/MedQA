"""Run the full evaluation suite from the command line."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medqa.config import get_settings
from medqa.evaluation.evaluator import PipelineEvaluator
from medqa.log import get_logger

logger = get_logger("medqa.evaluate")


def main() -> None:
    settings = get_settings()
    evaluator = PipelineEvaluator(settings)

    print("=" * 60)
    print("MedQA Evaluation Suite")
    print("=" * 60)

    # Baselines
    print("\n--- Baseline Classifiers ---\n")
    baselines = evaluator.evaluate_baselines()
    for b in baselines:
        print(f"  {b['name']:.<40} {b['accuracy'] * 100:.1f}%")

    # Symptom extraction
    print("\n--- Symptom Extraction ---\n")
    symptom_metrics = evaluator.evaluate_symptoms()
    for category, values in symptom_metrics.items():
        print(f"  {category}:")
        for k, v in values.items():
            print(f"    {k:.<35} {v:.4f}")

    print("\n" + "=" * 60)
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
