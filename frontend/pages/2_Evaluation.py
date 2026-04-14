"""MedQA — Model evaluation and benchmarks."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import streamlit as st
import pandas as pd

st.set_page_config(page_title="MedQA — Evaluation", layout="wide")

css_path = Path(__file__).parent.parent / "static" / "styles.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("#### Evaluation")
    st.markdown(
        '<p style="color:#64748b; font-size:0.78rem; margin-top:-0.5rem;">'
        "Benchmark results across models</p>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<p class="overline">Benchmarks</p>', unsafe_allow_html=True)
st.title("Model Evaluation")
st.markdown(
    '<p class="subtitle">'
    "Side-by-side comparison of intent classification approaches "
    "and symptom extraction quality."
    "</p>",
    unsafe_allow_html=True,
)

st.markdown("")

tab_intent, tab_symptoms = st.tabs(["Intent Classification", "Symptom Extraction"])

# =====================================================================
# INTENT CLASSIFICATION
# =====================================================================
with tab_intent:

    # ------------------------------------------------------------------
    # Benchmark comparison (always visible)
    # ------------------------------------------------------------------
    st.markdown("#### Model Comparison")
    st.markdown(
        '<p style="color:#64748b; font-size:0.85rem; margin-top:-0.5rem;">'
        "148 labeled questions &middot; 7 intent categories &middot; 70/30 split</p>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    benchmarks = [
        ("Rule-Based (keywords)", 0.35, "#ef4444"),
        ("TF-IDF + Random Forest", 0.49, "#f59e0b"),
        ("TF-IDF + Logistic Regression", 0.49, "#f59e0b"),
        ("LLM &mdash; Llama-3.3-70B (zero-shot)", 0.77, "#10b981"),
    ]

    rows_html = ""
    for name, acc, color in benchmarks:
        pct = f"{acc * 100:.0f}%"
        bar_width = f"{acc * 100:.0f}%"
        rows_html += (
            f'<div class="bench-row">'
            f'<span class="bench-name">{name}</span>'
            f'<div class="bench-bar-bg">'
            f'<div class="bench-bar" style="width:{bar_width}; background:{color};"></div>'
            f"</div>"
            f'<span class="bench-value" style="color:{color};">{pct}</span>'
            f"</div>"
        )

    st.markdown(
        f"""<div style="background:var(--surface); border:1px solid var(--border);
             border-radius:var(--r); overflow:hidden; box-shadow:var(--shadow-xs);">
            {rows_html}
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown("")

    # ------------------------------------------------------------------
    # Run live baselines (opt-in)
    # ------------------------------------------------------------------
    st.markdown("#### Detailed Reports")
    st.markdown(
        '<p style="color:#64748b; font-size:0.85rem; margin-top:-0.5rem;">'
        "Run the baseline classifiers to view per-class precision, recall, and F1.</p>",
        unsafe_allow_html=True,
    )

    @st.cache_data(show_spinner="Evaluating classifiers...")
    def run_baselines():
        from medqa.data.loader import DataLoader
        from medqa.evaluation.baselines import BaselineEvaluator

        loader = DataLoader()
        df = loader.load_labels()
        return BaselineEvaluator(df).run_all()

    if st.button("Run baseline evaluation"):
        results = run_baselines()

        for r in results:
            if "report" not in r:
                continue
            with st.expander(f"{r['name']}  —  {r['accuracy'] * 100:.1f}% accuracy"):
                report = r["report"]
                rows = []
                for cls, m in report.items():
                    if isinstance(m, dict) and "precision" in m:
                        rows.append({
                            "Class": cls.replace("_", " ").title(),
                            "Precision": f"{m['precision']:.2f}",
                            "Recall": f"{m['recall']:.2f}",
                            "F1": f"{m['f1-score']:.2f}",
                            "Support": int(m.get("support", 0)),
                        })
                if rows:
                    st.dataframe(
                        pd.DataFrame(rows),
                        use_container_width=True,
                        hide_index=True,
                    )

    st.markdown("---")

    # ------------------------------------------------------------------
    # Methodology
    # ------------------------------------------------------------------
    st.markdown("#### Methodology")

    mcol1, mcol2 = st.columns(2)
    with mcol1:
        st.markdown(
            """<div class="info-block">
                <h4>Data</h4>
                <p style="font-size:0.84rem; color:#475569; line-height:1.7; margin:0;">
                    148 medical questions, manually labeled with one of 7 intent
                    categories. Split 70/30 (stratified) for training and test.
                    Traditional baselines trained on TF-IDF bigram features
                    (max 5,000 features, min document frequency 2).
                </p>
            </div>""",
            unsafe_allow_html=True,
        )

    with mcol2:
        st.markdown(
            """<div class="info-block">
                <h4>LLM Evaluation</h4>
                <p style="font-size:0.84rem; color:#475569; line-height:1.7; margin:0;">
                    The LLM classifier (Llama-3.3-70B via Groq) uses zero-shot
                    prompting with no training data. Evaluated on a 30-question
                    random sample from the labeled set (seed 42). Original 18
                    labels merged into 10 categories to reduce sparsity.
                </p>
            </div>""",
            unsafe_allow_html=True,
        )


# =====================================================================
# SYMPTOM EXTRACTION
# =====================================================================
with tab_symptoms:

    st.markdown("#### Extraction Quality")
    st.markdown(
        '<p style="color:#64748b; font-size:0.85rem; margin-top:-0.5rem;">'
        "Evaluated against 10 gold-standard annotated questions using Llama-3.1-8B.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    # Metrics table
    metrics_df = pd.DataFrame([
        {"Field": "Symptom", "Exact Match": "90%", "Token F1": "0.93"},
        {"Field": "Body Location", "Exact Match": "80%", "Token F1": "0.85"},
        {"Field": "Duration", "Exact Match": "80%", "Token F1": "0.80"},
        {"Field": "Trigger", "Exact Match": "70%", "Token F1": "0.75"},
    ])

    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    st.markdown("#### Gold-Standard Annotations")
    st.markdown(
        '<p style="color:#64748b; font-size:0.85rem; margin-top:-0.5rem;">'
        "Subset of the manually annotated evaluation data.</p>",
        unsafe_allow_html=True,
    )

    gold_df = pd.DataFrame([
        {"Question": "Are dry lips a symptom of anything?", "Symptom": "dry lips", "Location": "lips", "Duration": "—", "Trigger": "—"},
        {"Question": "Can high blood pressure cause blue lips?", "Symptom": "blue lips", "Location": "lips", "Duration": "—", "Trigger": "high blood pressure"},
        {"Question": "At what age is shortness of breath normal?", "Symptom": "shortness of breath", "Location": "chest", "Duration": "occasional", "Trigger": "—"},
        {"Question": "Can runny nose be a symptom of Covid?", "Symptom": "runny nose", "Location": "nose", "Duration": "—", "Trigger": "Covid infection"},
        {"Question": "Can dizziness be serious?", "Symptom": "dizziness", "Location": "brain", "Duration": "—", "Trigger": "—"},
    ])

    st.dataframe(gold_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    st.markdown("#### Anatomical Region Mapping")

    from medqa.models.schemas import BODY_GROUP_MEMBERS

    group_rows = []
    for group, members in BODY_GROUP_MEMBERS.items():
        group_rows.append({
            "Region": group.replace("_", " ").title(),
            "Mapped Terms": ", ".join(members),
        })

    st.dataframe(pd.DataFrame(group_rows), use_container_width=True, hide_index=True)
