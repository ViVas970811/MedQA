"""MedQA — Dataset explorer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import streamlit as st
import pandas as pd

st.set_page_config(page_title="MedQA — Explorer", layout="wide")

css_path = Path(__file__).parent.parent / "static" / "styles.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


@st.cache_data
def load_data():
    from medqa.data.loader import DataLoader

    loader = DataLoader()
    return loader.load_corpus(), loader.load_labels()


corpus, labels_df = load_data()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("#### Explorer")
    st.markdown(
        '<p style="color:#64748b; font-size:0.78rem; margin-top:-0.5rem;">'
        "Browse and search the datasets</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.metric("Corpus", f"{len(corpus):,}")
    st.metric("Labeled", f"{len(labels_df):,}")
    st.metric("Intent Classes", labels_df["intent_merged"].nunique())

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<p class="overline">Dataset</p>', unsafe_allow_html=True)
st.title("Explorer")
st.markdown(
    '<p class="subtitle">'
    "Search the medical question corpus, filter labeled data by intent, "
    "and inspect dataset distributions."
    "</p>",
    unsafe_allow_html=True,
)

st.markdown("")

tab_corpus, tab_labels, tab_dist = st.tabs(["Corpus", "Labeled Data", "Distributions"])

# =====================================================================
# CORPUS
# =====================================================================
with tab_corpus:
    st.markdown("")
    search = st.text_input(
        "Search",
        placeholder="Filter by keyword (e.g. 'headache', 'diabetes')...",
        label_visibility="collapsed",
    )

    filtered = corpus
    if search.strip():
        kw = search.strip().lower()
        filtered = [q for q in corpus if kw in q.lower()]

    st.markdown(
        f'<p style="font-size:0.8rem; color:#94a3b8; margin-bottom:0.25rem;">'
        f"Showing {len(filtered):,} of {len(corpus):,} questions"
        f"{f' matching &ldquo;{search.strip()}&rdquo;' if search.strip() else ''}"
        f"</p>",
        unsafe_allow_html=True,
    )

    PAGE_SIZE = 30
    total_pages = max(1, -(-len(filtered) // PAGE_SIZE))

    if "corpus_page" not in st.session_state:
        st.session_state.corpus_page = 1
    # Reset page on new search
    if search.strip():
        st.session_state.corpus_page = 1

    page = st.session_state.corpus_page
    start = (page - 1) * PAGE_SIZE
    page_items = filtered[start : start + PAGE_SIZE]

    items_html = ""
    for i, q in enumerate(page_items, start + 1):
        items_html += (
            f'<div class="corpus-item">'
            f'<span class="corpus-num">{i}</span>'
            f"<span>{q}</span>"
            f"</div>"
        )

    st.markdown(
        f'<div style="background:var(--surface); border:1px solid var(--border); '
        f'border-radius:var(--r); padding:0.5rem 1.25rem; box-shadow:var(--shadow-xs);">'
        f"{items_html}</div>",
        unsafe_allow_html=True,
    )

    # Pagination
    pcol1, pcol2, pcol3 = st.columns([1, 3, 1])
    with pcol1:
        if st.button("Previous", disabled=(page <= 1), use_container_width=True, key="prev"):
            st.session_state.corpus_page = max(1, page - 1)
            st.rerun()
    with pcol2:
        st.markdown(
            f'<p style="text-align:center; color:#94a3b8; font-size:0.82rem; margin-top:0.5rem;">'
            f"Page {page} of {total_pages}</p>",
            unsafe_allow_html=True,
        )
    with pcol3:
        if st.button("Next", disabled=(page >= total_pages), use_container_width=True, key="next"):
            st.session_state.corpus_page = min(total_pages, page + 1)
            st.rerun()


# =====================================================================
# LABELED DATA
# =====================================================================
with tab_labels:
    st.markdown("")

    intent_filter = st.multiselect(
        "Filter by intent",
        options=sorted(labels_df["intent_merged"].unique()),
    )

    view = labels_df.copy()
    if intent_filter:
        view = view[view["intent_merged"].isin(intent_filter)]

    cols = ["question", "intent_merged", "symptoms", "urgency"]
    available = [c for c in cols if c in view.columns]

    st.dataframe(
        view[available].rename(columns={
            "question": "Question",
            "intent_merged": "Intent",
            "symptoms": "Symptoms",
            "urgency": "Urgency",
        }),
        use_container_width=True,
        hide_index=True,
        height=520,
    )

    st.markdown(
        f'<p style="font-size:0.78rem; color:#94a3b8; margin-top:0.5rem;">'
        f"{len(view)} of {len(labels_df)} questions shown</p>",
        unsafe_allow_html=True,
    )


# =====================================================================
# DISTRIBUTIONS
# =====================================================================
with tab_dist:
    st.markdown("")

    st.markdown("#### Intent Categories")
    intent_counts = labels_df["intent_merged"].value_counts()
    st.bar_chart(
        pd.DataFrame({"Count": intent_counts.values}, index=intent_counts.index),
        color="#0f766e",
        height=380,
    )

    st.markdown("---")

    dcol1, dcol2 = st.columns(2)

    with dcol1:
        if "urgency" in labels_df.columns:
            st.markdown("#### Urgency Levels")
            urgency_counts = labels_df["urgency"].value_counts()
            st.bar_chart(
                pd.DataFrame({"Count": urgency_counts.values}, index=urgency_counts.index),
                color="#06b6d4",
                height=280,
            )

    with dcol2:
        st.markdown("#### Question Length (words)")
        lengths = pd.Series([len(q.split()) for q in corpus])
        st.bar_chart(
            lengths.value_counts().sort_index().rename("Count"),
            color="#0f766e",
            height=280,
        )
