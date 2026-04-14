"""MedQA — Analyze a medical question."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import streamlit as st

st.set_page_config(page_title="MedQA — Analyze", layout="wide")

css_path = Path(__file__).parent.parent / "static" / "styles.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Pipeline (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Warming up models and index...")
def load_pipeline():
    from medqa.config import get_settings
    from medqa.pipeline.orchestrator import MedQAPipeline

    pipeline = MedQAPipeline(get_settings())
    pipeline.initialize()
    return pipeline


# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("#### Settings")
    st.markdown("---")
    top_k = st.slider("Evidence passages to retrieve", 1, 15, 5)
    st.markdown("---")
    st.markdown(
        '<p style="font-size:0.78rem; color:#64748b;">Try one of these:</p>',
        unsafe_allow_html=True,
    )
    samples = [
        "Why does my chest feel tight when I wake up?",
        "Can high blood pressure cause blue lips?",
        "How is strep throat treated in children?",
        "Is occasional shortness of breath normal?",
        "Are floaters in the eye serious?",
    ]
    for s in samples:
        if st.button(s, key=f"s_{hash(s)}", use_container_width=True):
            st.session_state["q"] = s

# ---------------------------------------------------------------------------
# Main — input
# ---------------------------------------------------------------------------
st.markdown('<p class="overline">Analyze</p>', unsafe_allow_html=True)
st.title("Ask a medical question")
st.markdown(
    '<p class="subtitle">'
    "Type a question below. The pipeline will classify its intent, extract "
    "clinical attributes, retrieve evidence, and generate a response."
    "</p>",
    unsafe_allow_html=True,
)

st.markdown("")

question = st.text_area(
    "Question",
    value=st.session_state.get("q", ""),
    height=90,
    placeholder="e.g., Why do my eyes feel dry in the morning?",
    label_visibility="collapsed",
)

bcol, _ = st.columns([1, 5])
with bcol:
    run = st.button("Run analysis", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
if run and question.strip():
    pipeline = load_pipeline()
    from medqa.models.schemas import PipelineRequest

    with st.spinner(""):
        result = pipeline.run(PipelineRequest(question=question.strip(), top_k=top_k))

    st.markdown("---")

    # Timing
    st.markdown(
        f'<p style="color:#94a3b8; font-size:0.76rem; text-align:right; margin-bottom:0.5rem;">'
        f"Completed in {result.processing_time_ms:,.0f} ms</p>",
        unsafe_allow_html=True,
    )

    # ── Intent ──────────────────────────────────────────────
    intent_display = result.intent.intent.replace("_", " ").title()
    st.markdown(
        f"""<div class="result-section">
            <div class="result-header">
                <span class="result-icon">&bull;</span>
                <span class="result-title">Intent</span>
            </div>
            <span class="intent-pill">{intent_display}</span>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Symptoms ────────────────────────────────────────────
    sym = result.symptoms
    sym_fields = [
        ("Symptom", sym.symptom),
        ("Body Location", sym.body_location),
        ("Duration", sym.duration),
        ("Trigger", sym.trigger),
    ]

    sym_items = ""
    for label, val in sym_fields:
        display = val if val else "&mdash;"
        sym_items += (
            f'<div class="sym-item">'
            f'<div class="sym-label">{label}</div>'
            f'<div class="sym-value">{display}</div>'
            f"</div>"
        )

    group = sym.body_location_group.replace("_", " ").title()
    st.markdown(
        f"""<div class="result-section">
            <div class="result-header">
                <span class="result-icon">&bull;</span>
                <span class="result-title">Extracted Attributes</span>
                <span class="result-subtitle">Region: {group}</span>
            </div>
            <div class="sym-grid">{sym_items}</div>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Retrieved Evidence ──────────────────────────────────
    rq_items = ""
    for i, rq in enumerate(result.retrieved_questions, 1):
        rq_items += (
            f'<div class="rq-item">'
            f'<span class="rq-rank">{i}</span>'
            f"<span>{rq.question}</span>"
            f'<span class="rq-score">{rq.score:.2f}</span>'
            f"</div>"
        )

    st.markdown(
        f"""<div class="result-section">
            <div class="result-header">
                <span class="result-icon">&bull;</span>
                <span class="result-title">Retrieved Evidence</span>
                <span class="result-subtitle">{len(result.retrieved_questions)} passages</span>
            </div>
            {rq_items}
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Answer ──────────────────────────────────────────────
    st.markdown(
        f"""<div class="result-section" style="border:none; box-shadow:none; padding:0;">
            <div class="result-header" style="margin-bottom:0.6rem;">
                <span class="result-icon">&bull;</span>
                <span class="result-title">Response</span>
            </div>
            <div class="answer-container">{result.answer}</div>
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """<div class="disclaimer">
            <span class="disclaimer-icon">&#x26A0;</span>
            <span>AI-generated response for informational purposes only.
            Not a substitute for professional medical advice.</span>
        </div>""",
        unsafe_allow_html=True,
    )

elif run:
    st.warning("Enter a question to continue.")
