"""Eval harness for Day 12 P6 v2 end-to-end measurement.

run_leader_pair() mirrors the ADR-005 shared-retrieval pattern from
compare_leaders() but holds both DigitalCloneFlow instances so it can
capture routing_decision, timings, and full response text that
compare_leaders() discards (C1 resolution).

run_measurement() implements the C4 run design (one full pass + two
in-domain re-runs + reactive OOD recheck). Results are written
incrementally to results/evaluation_day12.json after every query pair.

Category classification:
    IN_DOMAIN_CATEGORIES and OOD_CATEGORIES map query category to eval axis.
    A category in neither set raises ValueError — fail-loud, not silent-miscolumn (C2).

Field naming for fallback records:
    clone_response_text — what the EvaluatorAgent scored (CloneAgent's raw output).
    delivered_text      — what the user receives (FallbackAgent acknowledgment or
                          StyledResponse.response on the deliver path).
    chunk_contents      — the five retrieved chunks with rank, score, and full text,
                          persisted so post-run span analysis needs no extra LLM calls.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

from src.components.retriever import Retriever
from src.config import load_config
from src.flow import DigitalCloneFlow
from src.schemas import StyledResponse
from src.style.profile_builder import load_profile

logger = logging.getLogger(__name__)

_LEADERS = ("Linus Torvalds", "Greg Kroah-Hartman")
_LEADER_KEYS = ("torvalds", "kroah_hartman")

IN_DOMAIN_CATEGORIES: frozenset[str] = frozenset({
    "statistical_learning_ml",
    "data_mining",
    "numerical_methods",
    "programming_fundamentals",
})
OOD_CATEGORIES: frozenset[str] = frozenset({
    "systems_absent_from_corpus",
    "off_topic_technical",
})


def classify_category(category: str) -> str:
    """Return 'in_domain' or 'ood'; raise ValueError on unknown category."""
    if category in IN_DOMAIN_CATEGORIES:
        return "in_domain"
    if category in OOD_CATEGORIES:
        return "ood"
    raise ValueError(
        f"Unknown category: {category!r}. "
        f"Not in IN_DOMAIN_CATEGORIES={set(IN_DOMAIN_CATEGORIES)} "
        f"or OOD_CATEGORIES={set(OOD_CATEGORIES)}."
    )


def _extract_leader_record(flow: DigitalCloneFlow, leader_key: str, leader_name: str) -> dict:
    """Pull per-leader result from flow state after kickoff completes."""
    state = flow.state
    rd = state.routing_decision

    decision = rd.decision if rd else "unknown"
    trigger_category = rd.trigger_category if rd else None
    trigger_reason = rd.trigger_reason if rd else None
    routing_reasoning = rd.reasoning if rd else ""

    is_deliver = isinstance(state.styled_response, StyledResponse)
    output_type = "StyledResponse" if is_deliver else "FallbackResponse"

    # What the user receives — deliver path gets styled response, fallback path gets acknowledgment
    if is_deliver and state.styled_response:
        delivered_text = state.styled_response.response
    elif state.fallback_response:
        delivered_text = state.fallback_response.acknowledgment
    else:
        delivered_text = None

    ev = state.evaluation

    # Chunk contents for post-run span analysis — no extra LLM calls needed
    chunk_contents = [
        {
            "rank": r.rank,
            "score": round(r.score, 4),
            "source_topic": r.chunk.source_topic,
            "content": r.chunk.content,
        }
        for r in state.chunks
    ]

    return {
        "leader": leader_name,
        "leader_key": leader_key,
        "decision": decision,
        "trigger_category": trigger_category,
        "trigger_reason": trigger_reason,
        "routing_reasoning": routing_reasoning,
        # CloneAgent's raw output — what the EvaluatorAgent scored for groundedness
        "clone_response_text": state.response_text,
        # What was delivered to the user (FallbackAgent acknowledgment on fallback path)
        "delivered_text": delivered_text,
        "output_type": output_type,
        "style_score": ev.style_score if ev else None,
        "groundedness_score": ev.groundedness_score if ev else None,
        "confidence_score": ev.confidence_score if ev else None,
        "flags": ev.flags if ev else [],
        "chunk_contents": chunk_contents,
        "timings": dict(flow.timings),
    }


def run_leader_pair(query: str) -> dict:
    """Run both leaders on one query, sharing retrieved chunks (ADR-005).

    Asserts exactly one Retriever.run() call across the pair — the same gate
    as tests/integration/test_compare_leaders.py::test_compare_leaders_retriever_called_once.
    Torvalds flow retrieves; KH flow receives pre-populated chunks and early-exits
    the retrieve step.

    Returns per-leader records with timing, routing decision (decision,
    trigger_category, trigger_reason, routing_reasoning), evaluation scores,
    output type, clone_response_text (scored), delivered_text, and chunk_contents (C5).
    """
    config = load_config()

    _call_count: list[int] = []
    _original_run = Retriever.run

    def _counting_run(self: Retriever, q: str) -> list:  # type: ignore[override]
        _call_count.append(1)
        return _original_run(self, q)

    with patch.object(Retriever, "run", _counting_run):
        profile_t = load_profile(Path(config.leaders["torvalds"].profile_path))
        flow_t = DigitalCloneFlow()
        flow_t.kickoff(inputs={
            "query": query,
            "leader": _LEADERS[0],
            "style_profile": profile_t,
        })
        shared_chunks = list(flow_t.state.chunks)

        profile_kh = load_profile(Path(config.leaders["kroah_hartman"].profile_path))
        flow_kh = DigitalCloneFlow()
        flow_kh.kickoff(inputs={
            "query": query,
            "leader": _LEADERS[1],
            "style_profile": profile_kh,
            "chunks": shared_chunks,
        })

    retriever_call_count = len(_call_count)
    assert retriever_call_count == 1, (
        f"ONE-RETRIEVAL ASSERTION FAILED: Retriever.run() called {retriever_call_count} times "
        f"(expected 1). KH retrieve step did not early-exit on pre-populated chunks — "
        f"ADR-005 shared-retrieval guarantee is broken."
    )

    t_record = _extract_leader_record(flow_t, "torvalds", _LEADERS[0])
    kh_record = _extract_leader_record(flow_kh, "kroah_hartman", _LEADERS[1])

    return {
        "query": query,
        "retriever_call_count": retriever_call_count,
        "torvalds": t_record,
        "kroah_hartman": kh_record,
    }


def run_measurement(
    path: str | Path = "data/eval/queries.json",
    output: str | Path = "results/evaluation_day12.json",
) -> dict:
    """C4 run design: one full pass + two in-domain re-runs + reactive OOD recheck.

    Pass 1 (full): all 20 queries × both leaders = 40 records.
    Passes 2-3 (in-domain only): 14 in-domain queries × both leaders = 28 records each.
    Reactive OOD recheck: if any OOD record delivers in pass 1, re-run that
    specific query twice before classifying it a hallucination.

    Results are written incrementally after every query pair so a mid-run failure
    loses at most one pair.
    """
    path = Path(path)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(path) as f:
        queries = json.load(f)

    for q in queries:
        q["axis"] = classify_category(q["category"])

    in_domain = [q for q in queries if q["axis"] == "in_domain"]
    ood = [q for q in queries if q["axis"] == "ood"]

    logger.info(
        "Loaded %d queries: %d in-domain, %d OOD",
        len(queries), len(in_domain), len(ood),
    )

    all_records: list[dict] = []
    ood_delivers: list[str] = []  # query_ids where any leader delivered on OOD

    def _run_and_append(q: dict, pass_num: int | str, idx: int, total: int) -> None:
        t0 = perf_counter()
        logger.info(
            "Pass %s [%d/%d] %s (%s, expected=%s)",
            pass_num, idx, total, q["id"], q["axis"], q["expected_behavior"],
        )
        result = run_leader_pair(q["query"])
        elapsed = round(perf_counter() - t0, 1)

        record = {
            "pass": pass_num,
            "query_id": q["id"],
            "query": q["query"],
            "category": q["category"],
            "axis": q["axis"],
            "expected_behavior": q["expected_behavior"],
            "retriever_call_count": result["retriever_call_count"],
            "torvalds": result["torvalds"],
            "kroah_hartman": result["kroah_hartman"],
            "pair_elapsed_s": elapsed,
        }
        all_records.append(record)

        # Track OOD delivers for reactive recheck
        if pass_num == 1 and q["axis"] == "ood":
            for lk in ("torvalds", "kroah_hartman"):
                if result[lk]["decision"] == "deliver":
                    if q["id"] not in ood_delivers:
                        ood_delivers.append(q["id"])
                    logger.warning(
                        "OOD DELIVER DETECTED: %s / %s — will trigger reactive recheck",
                        q["id"], lk,
                    )

        # Write incrementally — survives a mid-run crash
        with open(output, "w") as f:
            json.dump(all_records, f, indent=2, default=str)

        logger.info(
            "  T=%s KH=%s  gs_T=%.3f gs_KH=%.3f  (%.1fs)",
            result["torvalds"]["decision"],
            result["kroah_hartman"]["decision"],
            result["torvalds"]["groundedness_score"] or 0,
            result["kroah_hartman"]["groundedness_score"] or 0,
            elapsed,
        )

    # Pass 1: all 20 queries
    logger.info("=== PASS 1: all %d queries ===", len(queries))
    for i, q in enumerate(queries, 1):
        _run_and_append(q, 1, i, len(queries))

    # Reactive OOD recheck
    if ood_delivers:
        logger.warning(
            "=== REACTIVE OOD RECHECK: %d query-ids with unexpected delivers ===",
            len(ood_delivers),
        )
        ood_by_id = {q["id"]: q for q in ood}
        for qid in ood_delivers:
            q = ood_by_id[qid]
            for recheck_num in (1, 2):
                _run_and_append(q, f"ood_recheck_{recheck_num}", recheck_num, 2)
    else:
        logger.info("No OOD delivers in pass 1 — reactive recheck skipped.")

    # Passes 2-3: in-domain only
    for pass_num in (2, 3):
        logger.info("=== PASS %d: %d in-domain queries ===", pass_num, len(in_domain))
        for i, q in enumerate(in_domain, 1):
            _run_and_append(q, pass_num, i, len(in_domain))

    total = len(all_records)
    logger.info("=== MEASUREMENT COMPLETE: %d total records ===", total)
    logger.info("Results written to: %s", output)
    return {"total_records": total, "output": str(output)}
