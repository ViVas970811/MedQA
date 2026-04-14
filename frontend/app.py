"""MedQA — Home."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import streamlit as st

st.set_page_config(page_title="MedQA", layout="wide", initial_sidebar_state="expanded")

css_path = Path(__file__).parent / "static" / "styles.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("#### MedQA")
    st.markdown(
        '<p style="color:#64748b; font-size:0.78rem; margin-top:-0.75rem;">'
        "Clinical Intelligence Platform</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown(
        '<p style="font-size:0.82rem; color:#94a3b8;">'
        "Navigate using the pages above to analyze questions, "
        "view model benchmarks, or explore the dataset.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown(
        '<p style="color:#475569; font-size:0.72rem;">'
        "v1.0 &nbsp;&middot;&nbsp; Llama 3 on Groq &nbsp;&middot;&nbsp; FAISS</p>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown('<p class="overline">Clinical NLP</p>', unsafe_allow_html=True)
st.title("MedQA")
st.markdown(
    '<p class="subtitle">'
    "Understand any medical question in seconds. This system decomposes queries "
    "into structured clinical attributes &mdash; intent, symptoms, body region "
    "&mdash; then retrieves evidence and generates a grounded response."
    "</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Pipeline flow (horizontal visualization)
# ---------------------------------------------------------------------------
st.markdown("")
st.markdown(
    """<div class="flow-container">
        <div class="flow-step">
            <div class="flow-num">1</div>
            <div class="flow-title">Classify Intent</div>
            <div class="flow-model">Llama-3.3-70B</div>
        </div>
        <div class="flow-step">
            <div class="flow-num">2</div>
            <div class="flow-title">Extract Symptoms</div>
            <div class="flow-model">Llama-3.1-8B</div>
        </div>
        <div class="flow-step">
            <div class="flow-num">3</div>
            <div class="flow-title">Retrieve Evidence</div>
            <div class="flow-model">MiniLM + FAISS</div>
        </div>
        <div class="flow-step">
            <div class="flow-num">4</div>
            <div class="flow-title">Generate Answer</div>
            <div class="flow-model">Llama-3.1-8B</div>
        </div>
    </div>""",
    unsafe_allow_html=True,
)

st.markdown("---")

# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------
st.markdown("### What it does")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """<div class="feature-card">
            <span class="feature-icon">&oplus;</span>
            <div class="feature-title">Intent Recognition</div>
            <p class="feature-desc">
                Classifies medical questions into 10 clinical intent categories
                &mdash; treatment, symptom, diagnosis, prognosis, and more &mdash;
                using zero-shot LLM inference.
            </p>
        </div>""",
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """<div class="feature-card">
            <span class="feature-icon">&Xi;</span>
            <div class="feature-title">Structured Extraction</div>
            <p class="feature-desc">
                Pulls out symptom name, body location, duration, and trigger
                from free-text questions, then maps to 9 standardized
                anatomical groups.
            </p>
        </div>""",
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """<div class="feature-card">
            <span class="feature-icon">&Delta;</span>
            <div class="feature-title">Evidence-Grounded Answers</div>
            <p class="feature-desc">
                Retrieves the most semantically similar questions from a
                3,173-question medical corpus, then uses them as context
                to generate a safe, cited response.
            </p>
        </div>""",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ---------------------------------------------------------------------------
# Design Principles
# ---------------------------------------------------------------------------
st.markdown("### Design principles")

pcol1, pcol2 = st.columns(2)

with pcol1:
    st.markdown(
        """<div class="info-block">
            <h4>Transparent by default</h4>
            <p style="font-size:0.86rem; color:#475569; line-height:1.6; margin:0;">
                Every intermediate step &mdash; the classified intent, extracted
                symptoms, retrieved sources &mdash; is exposed alongside the
                final answer. No black boxes.
            </p>
        </div>""",
        unsafe_allow_html=True,
    )

with pcol2:
    st.markdown(
        """<div class="info-block">
            <h4>Modular evaluation</h4>
            <p style="font-size:0.86rem; color:#475569; line-height:1.6; margin:0;">
                Each pipeline stage can be benchmarked independently. Intent
                classification is tested against rule-based and ML baselines;
                symptom extraction against gold-standard annotations.
            </p>
        </div>""",
        unsafe_allow_html=True,
    )

st.markdown("")

# Disclaimer
st.markdown(
    """<div class="disclaimer">
        <span class="disclaimer-icon">&#x26A0;</span>
        <span>For informational purposes only. This system does not provide
        medical advice, diagnosis, or treatment. Always consult a qualified
        healthcare professional.</span>
    </div>""",
    unsafe_allow_html=True,
)
