"""Torvalds Digital Clone — Streamlit web UI.

All heavy I/O routes through DigitalCloneFlow (single-leader) or
compare_leaders (dual-leader). No direct LiteLLM / FAISS / Cohere imports.
"""

from __future__ import annotations

import multiprocessing
import os
from pathlib import Path
from typing import Union

os.environ.setdefault("OMP_NUM_THREADS", "1")
multiprocessing.set_start_method("spawn", force=True)

import streamlit as st

from src.config import load_config
from src.flow import DigitalCloneFlow
from src.flow import compare_leaders as _compare_leaders
from src.schemas import EvaluationResult, FallbackResponse, LeaderComparison, StyledResponse
from src.style.profile_builder import load_profile

# ---------------------------------------------------------------------------
# section: page_config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Torvalds Digital Clone",
    page_icon="🐧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# section: sidebar_visualizations
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Portfolio Charts")
    charts_dir = Path("results/charts")
    if charts_dir.exists():
        chart_files = sorted(charts_dir.glob("*.png"))
        if chart_files:
            for chart_path in chart_files:
                label = chart_path.stem.replace("-", " ").replace("_", " ").title()
                st.image(str(chart_path), caption=label, use_column_width=True)
        else:
            st.caption("No charts yet — run `cli evaluate` to generate.")
    else:
        st.caption("No charts directory found.")

# ---------------------------------------------------------------------------
# section: query_input
# ---------------------------------------------------------------------------

st.title("Torvalds Digital Clone")
st.markdown(
    "Ask a question and get a response styled as a Linux kernel leader. "
    "Uses the full pipeline: retrieve → rerank → style → evaluate → route."
)

col_input, col_leader = st.columns([3, 1])
with col_input:
    query_text = st.text_input(
        "Question",
        placeholder="e.g. What is the right way to handle memory allocation in kernel drivers?",
        label_visibility="collapsed",
    )
with col_leader:
    leader_choice = st.selectbox(
        "Leader",
        options=["Torvalds", "Kroah-Hartman", "Compare Both"],
        label_visibility="collapsed",
    )

run_button = st.button("Ask", type="primary", disabled=not query_text.strip())

# ---------------------------------------------------------------------------
# section: render_score_breakdown
# ---------------------------------------------------------------------------

def render_score_breakdown(ev: EvaluationResult) -> None:
    with st.expander("Score breakdown", expanded=True):
        col1, col2, col3 = st.columns(3)
        col1.metric("Style", f"{ev.style_score:.2f}", help="Cosine similarity to leader profile vector")
        col2.metric(
            "Groundedness",
            f"{ev.groundedness_score:.2f}",
            help="HHEM entailment score against retrieved chunks (ADR-020)",
        )
        col3.metric("Confidence", f"{ev.confidence_score:.2f}", help="Reranker + keyword + hedge heuristic")
        st.caption(f"Explanation: {ev.explanation}")


# ---------------------------------------------------------------------------
# section: render_fallback_card
# ---------------------------------------------------------------------------

def render_fallback_card(fb: FallbackResponse) -> None:
    st.warning(fb.acknowledgment)

    if fb.suggested_redirections:
        st.markdown("**Suggested redirections:**")
        for r in fb.suggested_redirections:
            st.markdown(f"- {r}")

    if fb.unstyled_response:
        with st.expander("Unstyled response"):
            st.markdown(fb.unstyled_response)

    st.markdown(f"**Book a call instead:** [{fb.calendar_link}]({fb.calendar_link})")

    if fb.available_slots:
        st.markdown("**Available slots:**")
        for slot in fb.available_slots:
            st.markdown(f"- {slot}")


# ---------------------------------------------------------------------------
# section: render_response_card
# ---------------------------------------------------------------------------

def render_response_card(resp: StyledResponse) -> None:
    st.markdown(resp.response)

    if resp.citations:
        with st.expander(f"Citations ({len(resp.citations)})"):
            for cit in resp.citations:
                st.markdown(
                    f"- **{cit.source_topic}** — _{cit.text_snippet[:120]}…_ "
                    f"(relevance: {cit.relevance_score:.2f})"
                )

    render_score_breakdown(resp.evaluation)


# ---------------------------------------------------------------------------
# section: render_single
# ---------------------------------------------------------------------------

def render_single(output: Union[StyledResponse, FallbackResponse, None], leader: str) -> None:
    if output is None:
        st.error("No output returned from pipeline.")
        return
    st.subheader(f"{leader} responds")
    if isinstance(output, FallbackResponse):
        render_fallback_card(output)
    else:
        render_response_card(output)


# ---------------------------------------------------------------------------
# section: render_compare
# ---------------------------------------------------------------------------

def render_compare(result: LeaderComparison) -> None:
    st.subheader("Side-by-side comparison")
    col_t, col_g = st.columns(2)
    with col_t:
        st.markdown("**Linus Torvalds**")
        if isinstance(result.torvalds, FallbackResponse):
            render_fallback_card(result.torvalds)
        else:
            render_response_card(result.torvalds)
    with col_g:
        st.markdown("**Greg Kroah-Hartman**")
        if isinstance(result.kroah_hartman, FallbackResponse):
            render_fallback_card(result.kroah_hartman)
        else:
            render_response_card(result.kroah_hartman)


# ---------------------------------------------------------------------------
# section: dispatch
# ---------------------------------------------------------------------------

if run_button and query_text.strip():
    if leader_choice == "Compare Both":
        with st.spinner("Running dual-leader comparison (retrieve once, style twice)…"):
            result = _compare_leaders(query_text.strip())
        render_compare(result)
    else:
        display_name = "Linus Torvalds" if leader_choice == "Torvalds" else "Greg Kroah-Hartman"
        config_key = "torvalds" if leader_choice == "Torvalds" else "kroah_hartman"
        with st.spinner(f"Querying as {display_name}…"):
            _config = load_config()
            _profile = load_profile(Path(_config.leaders[config_key].profile_path))
            flow = DigitalCloneFlow()
            flow.kickoff(inputs={"query": query_text.strip(), "leader": display_name, "style_profile": _profile})
            output = flow.state.styled_response or flow.state.fallback_response
        render_single(output, display_name)

# ---------------------------------------------------------------------------
# section: footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "Torvalds Digital Clone — P6 Portfolio Project. "
    "All responses are AI-generated; scores reflect automated quality signals, not human judgment."
)
