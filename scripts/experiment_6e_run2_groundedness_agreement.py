"""Experiment 6e (Run 2) — GPT-4o-mini vs Ollama qwen3:8b: groundedness scoring agreement.

CORRECTS RUN 1 (experiment_6e_local_vs_api.py): Run 1 measured explanation-generation
latency only — both models received pre-computed component scores and applied the same
weighted-sum formula, producing Pearson=1.0 trivially. Run 2 measures the intended
question: given the same (query, top-5 chunks), do the two models independently assign
similar groundedness scores?

"Embedding-cosine scorer" note: the production score_groundedness() is NOT keyword overlap.
It computes sentence-level OpenAI embedding cosine similarity with the query as a response
proxy. "Baseline" throughout this script means this production embedding-cosine scorer.

Design: RETRIEVE ONCE, SCORE TWICE, COMPARE THREE WAYS.
  - Retrieve + Cohere-rerank ONCE per query (10 calls + 1 preflight = 11 total).
  - Both LLMs independently score groundedness in [0,1] from the same (query, top-5
    chunk texts) context. Style and confidence remain deterministic.
  - Baseline = score_groundedness(query, top5) [embedding cosine, query-as-proxy].
  - Three Pearsons: (GPT, Ollama) primary; (GPT, baseline), (Ollama, baseline) secondary.

Pre-run hypotheses (stated before running):
  H1: Pearson(GPT, Ollama) ≈ 0.85-0.95. Phase 2 bimodal Cohere distribution makes the
      relevant/irrelevant chunk split unambiguous; both LLMs should agree on direction.
  H2: Pearson(GPT, baseline) ≈ 0.70-0.85. Same underlying signal, but LLMs add contextual
      reasoning cosine similarity can't. Expect rank-order agreement with score divergence.
  H3: Pearson(Ollama, baseline) ≈ similar to H2, close to GPT-baseline Pearson.
  H4: Structured-output success on harder task (score + weakest_dimension + explanation)
      will be 100% for GPT; 95-100% for Ollama. Run 1's 100% was on a trivial task.

Weakest-dimension attribution (stated before running):
  Correct weakest = component with largest positive shortfall below target:
    style_shortfall    = max(0, 0.90 − style)        [proxy: typically ≈ 0.40]
    ground_shortfall   = max(0, 0.60 − LLM_ground)   [varies by query]
    conf_shortfall     = max(0, 0.80 − confidence)    [varies by query]
  Tie-breaking: if top-two shortfalls within 0.05, naming either counts as correct.
  NOTE: style_shortfall ≈ 0.40 for all queries in the proxy regime (style ≈ 0.50),
  so style will be the "correct" weakest for most queries regardless of LLM scoring.
  This is a proxy-regime artifact (per Phase 4 finding); reported as a limitation.

Four-way decision framework (applied after data):
  Pearson(GPT, Ollama) > 0.85 AND structured-output success 100%  → HIGH AGREEMENT
  Pearson(GPT, Ollama) 0.50-0.85                                  → MEDIUM AGREEMENT
  Pearson(GPT, Ollama) < 0.50 OR structured-output failure rate > 5%  → LOW AGREEMENT
  [+ latency dimension from Run 1: Ollama 2.1x faster, confirmed real]
"""

from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, field_validator
from scipy.stats import pearsonr

load_dotenv()

if not os.environ.get("CO_API_KEY") and os.environ.get("COHERE_API_KEY"):
    os.environ["CO_API_KEY"] = os.environ["COHERE_API_KEY"]

import cohere
import instructor
import litellm

from src.eval.query_loader import load_queries
from src.evaluation.confidence_scorer import score_confidence
from src.evaluation.groundedness_scorer import score_groundedness
from src.rag.indexer import load_index
from src.rag.reranker import rerank
from src.rag.retriever import retrieve
from src.schemas import EmailMessage
from src.style.feature_extractor import extract_features
from src.style.profile_builder import load_profile
from src.style.style_scorer import score_style

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

QUERIES_PATH = Path("data/eval/queries_v1.json")
INDEX_DIR = Path("data/rag/faiss_index")
PROFILE_PATH = Path("data/models/torvalds_profile.json")
CHART_PATH = Path("docs/images/6e-run2-groundedness-agreement.png")

RETRIEVAL_TOP_N = 20
RERANK_TOP_N = 5
THRESHOLD = 0.75
FORMULA_WEIGHTS = (0.4, 0.4, 0.2)  # style, groundedness, confidence
SCORE_TARGETS = {"style": 0.90, "groundedness": 0.60, "confidence": 0.80}
TIE_TOL = 0.05  # within this shortfall delta, both tied dimensions are credited

GPT_MODEL = "gpt-4o-mini"
OLLAMA_MODEL = "ollama/qwen3:8b"
OLLAMA_DAEMON_URL = "http://localhost:11434"

CHUNK_TEXT_LIMIT = 500  # chars per chunk in prompt (prevents context overflow)

COHERE_INTER_QUERY_SLEEP = 10.0
COHERE_POST_PREFLIGHT_SLEEP = 10.0

# Pearson thresholds for four-way decision
HIGH_AGREEMENT = 0.85
LOW_AGREEMENT = 0.50
STRUCTURED_FAIL_THRESHOLD = 0.05  # > 5% failure rate → LOW AGREEMENT regardless

sep = "=" * 90
sep2 = "-" * 80


# ---------------------------------------------------------------------------
# Structured output model
# ---------------------------------------------------------------------------

class GroundednessAssessment(BaseModel):
    groundedness_score: float
    weakest_dimension: Literal["style", "groundedness", "confidence"]
    explanation: str

    @field_validator("groundedness_score")
    @classmethod
    def clamp(cls, v: float) -> float:
        return float(np.clip(v, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def _build_assessment_prompt(
    query: str,
    chunks: list,
    style: float,
    confidence: float,
) -> str:
    chunk_lines = []
    for i, rr in enumerate(chunks[:RERANK_TOP_N], start=1):
        text = rr.chunk.content[:CHUNK_TEXT_LIMIT].replace("\n", " ").strip()
        chunk_lines.append(f"[{i}] {text}")
    chunks_str = "\n".join(chunk_lines)

    return (
        f"You are evaluating a knowledge-grounded AI response for a leadership coaching system.\n\n"
        f"Query: {query}\n\n"
        f"Retrieved passages (top-5, used to ground an answer):\n{chunks_str}\n\n"
        f"Evaluate GROUNDEDNESS: how well do these passages directly support answering the query?\n\n"
        f"Scoring scale:\n"
        f"  0.0 = passages contain no information relevant to the query\n"
        f"  0.5 = passages contain some relevant content but key facts needed are missing\n"
        f"  1.0 = passages directly and completely ground an answer to the query\n\n"
        f"Other evaluation component scores (for reference only):\n"
        f"  Style score:      {style:.3f}  (target > 0.90)\n"
        f"  Confidence score: {confidence:.3f}  (target > 0.80)\n\n"
        f"Based on your groundedness estimate, identify which of 'style', 'groundedness', "
        f"or 'confidence' has the largest shortfall below its target "
        f"(targets: style=0.90, groundedness=0.60, confidence=0.80). "
        f"If two are within 0.05 of each other, name either.\n\n"
        f"Output three fields:\n"
        f"  groundedness_score: your estimate in [0.0, 1.0]\n"
        f"  weakest_dimension: one of 'style', 'groundedness', 'confidence'\n"
        f"  explanation: ≤ 20 words on the primary groundedness weakness"
    )


# ---------------------------------------------------------------------------
# LLM scoring helpers
# ---------------------------------------------------------------------------

_gpt_client = instructor.from_litellm(litellm.completion)
_ollama_client = instructor.from_litellm(litellm.completion, mode=instructor.Mode.JSON)


def _score_llm(
    client: instructor.Instructor,
    model: str,
    query: str,
    chunks: list,
    style: float,
    confidence: float,
) -> tuple[GroundednessAssessment | None, float, bool]:
    """Return (assessment, elapsed_s, success)."""
    prompt = _build_assessment_prompt(query, chunks, style, confidence)
    t0 = time.perf_counter()
    try:
        result: GroundednessAssessment = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_model=GroundednessAssessment,
            max_retries=2,
        )
        return result, time.perf_counter() - t0, True
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        print(f"    [WARN] {model} structured-output failed: {type(exc).__name__}: {exc}")
        return None, elapsed, False


# ---------------------------------------------------------------------------
# Weakest-dimension helpers
# ---------------------------------------------------------------------------

def _shortfalls(style: float, ground: float, confidence: float) -> dict[str, float]:
    return {
        "style": max(0.0, SCORE_TARGETS["style"] - style),
        "groundedness": max(0.0, SCORE_TARGETS["groundedness"] - ground),
        "confidence": max(0.0, SCORE_TARGETS["confidence"] - confidence),
    }


def _correct_weakest(sf: dict[str, float]) -> list[str]:
    """Return the correct weakest dimension(s), with tie-breaking at TIE_TOL."""
    best = max(sf.values())
    return [dim for dim, v in sf.items() if best - v <= TIE_TOL]


def _attribution_correct(claimed: str, correct_set: list[str]) -> bool:
    return claimed in correct_set


# ---------------------------------------------------------------------------
# Pre-flight helpers
# ---------------------------------------------------------------------------

def _daemon_reachable() -> bool:
    import urllib.request
    try:
        urllib.request.urlopen(f"{OLLAMA_DAEMON_URL}/api/tags", timeout=3)
        return True
    except Exception:
        return False


def _model_present(bare: str) -> bool:
    try:
        out = subprocess.check_output(["ollama", "list"], text=True, timeout=10)
        return bare in out
    except Exception:
        return False


def _query_as_email(qid: str, query: str) -> EmailMessage:
    return EmailMessage(
        sender="query-proxy",
        recipients=[],
        subject="",
        body=query,
        timestamp=datetime.now(tz=timezone.utc),
        message_id=qid,
        is_patch=False,
        quote_ratio=0.0,
    )


def _cohere_preflight() -> None:
    client = cohere.ClientV2(api_key=os.environ.get("CO_API_KEY", ""))
    resp = client.rerank(
        model="rerank-english-v3.0",
        query="health check",
        documents=["foo", "bar"],
        top_n=1,
    )
    if not resp.results:
        raise RuntimeError("Cohere preflight returned empty results.")
    print("cohere quota pre-check OK")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(sep)
    print("Experiment 6e Run 2 — GPT-4o-mini vs Ollama qwen3:8b: groundedness scoring agreement")
    print(sep)

    # --- Script-level assertions ---
    print("\n[Pre-flight assertions]")
    if not _daemon_reachable():
        raise RuntimeError("Ollama daemon not reachable at localhost:11434.")
    print("  Ollama daemon: reachable")
    bare = OLLAMA_MODEL.replace("ollama/", "")
    if not _model_present(bare):
        raise RuntimeError(f"Model '{bare}' not in `ollama list`.")
    print(f"  Ollama model:  {bare} present")

    # --- Load assets ---
    print(f"\nLoading assets...")
    index, metadata = load_index(INDEX_DIR)
    print(f"  Index: {index.ntotal} vectors")
    profile = load_profile(PROFILE_PATH)
    print(f"  Profile: {profile.leader_name}")
    queries = load_queries(QUERIES_PATH)
    print(f"  Queries: {len(queries)}")

    # --- Cohere preflight ---
    print()
    _cohere_preflight()
    print(f"Sleeping {COHERE_POST_PREFLIGHT_SLEEP}s after preflight ...")
    time.sleep(COHERE_POST_PREFLIGHT_SLEEP)

    # --- Pre-run hypotheses reminder ---
    print(f"\n[Hypotheses] H1: Pearson(GPT,Ollama)≈0.85-0.95  H2/H3: (LLM,baseline)≈0.70-0.85")
    print(f"             H4: structured-output success ≥95% for Ollama, 100% for GPT")
    print(f"\n{'Query':<8} {'KW':>6} {'GPT':>6} {'OL':>6} {'GPT_ms':>7} {'OL_ms':>7}"
          f" {'G_ok':>5} {'O_ok':>5} {'GPT_wd':<12} {'OL_wd':<12} {'Corr_wd':<8}")
    print(sep2)

    per_query: list[dict] = []

    for q_idx, qr in enumerate(queries):
        qid = qr["id"]
        query = qr["query"]

        # --- Retrieve + rerank ---
        candidates = retrieve(query, index, metadata, top_n=RETRIEVAL_TOP_N, provider="openai")
        top5 = rerank(query, candidates, top_n=RERANK_TOP_N)

        # --- Deterministic component scores ---
        fake_email = _query_as_email(qid, query)
        query_features = extract_features(fake_email)
        style = score_style(profile, query_features)
        baseline_ground = score_groundedness(query, top5)
        confidence = score_confidence(query, query, top5)

        # --- LLM groundedness scoring ---
        gpt_result, gpt_elapsed, gpt_ok = _score_llm(
            _gpt_client, GPT_MODEL, query, top5, style, confidence
        )
        ollama_result, ollama_elapsed, ollama_ok = _score_llm(
            _ollama_client, OLLAMA_MODEL, query, top5, style, confidence
        )

        gpt_ground = gpt_result.groundedness_score if gpt_ok else float("nan")
        ollama_ground = ollama_result.groundedness_score if ollama_ok else float("nan")
        gpt_wd = gpt_result.weakest_dimension if gpt_ok else "n/a"
        ollama_wd = ollama_result.weakest_dimension if ollama_ok else "n/a"

        # --- Weakest-dimension attribution ---
        # Use GPT's groundedness for GPT correctness, Ollama's for Ollama
        gpt_sf = _shortfalls(style, gpt_ground if gpt_ok else 0.0, confidence)
        oll_sf = _shortfalls(style, ollama_ground if ollama_ok else 0.0, confidence)
        gpt_correct_set = _correct_weakest(gpt_sf)
        oll_correct_set = _correct_weakest(oll_sf)
        gpt_wd_correct = _attribution_correct(gpt_wd, gpt_correct_set) if gpt_ok else False
        oll_wd_correct = _attribution_correct(ollama_wd, oll_correct_set) if ollama_ok else False
        wd_agree = (gpt_wd == ollama_wd) if (gpt_ok and ollama_ok) else False

        per_query.append({
            "id": qid,
            "query": query,
            "style": style,
            "confidence": confidence,
            "baseline_ground": baseline_ground,
            "gpt_ground": gpt_ground,
            "ollama_ground": ollama_ground,
            "gpt_elapsed": gpt_elapsed,
            "ollama_elapsed": ollama_elapsed,
            "gpt_ok": gpt_ok,
            "ollama_ok": ollama_ok,
            "gpt_expl": gpt_result.explanation if gpt_ok else "",
            "ollama_expl": ollama_result.explanation if ollama_ok else "",
            "gpt_wd": gpt_wd,
            "ollama_wd": ollama_wd,
            "gpt_wd_correct": gpt_wd_correct,
            "oll_wd_correct": oll_wd_correct,
            "wd_agree": wd_agree,
            "gpt_correct_set": gpt_correct_set,
            "oll_correct_set": oll_correct_set,
        })

        print(f"{qid:<8} {baseline_ground:>6.4f} "
              f"{gpt_ground:>6.4f} {ollama_ground:>6.4f} "
              f"{gpt_elapsed*1000:>7.0f} {ollama_elapsed*1000:>7.0f} "
              f"{'✓' if gpt_ok else '✗':>5} {'✓' if ollama_ok else '✗':>5} "
              f"{gpt_wd:<12} {ollama_wd:<12} {'✓' if wd_agree else '✗':<8}")

        if q_idx < len(queries) - 1:
            time.sleep(COHERE_INTER_QUERY_SLEEP)

    print(sep2)

    # --- Filter to successful rows for Pearson ---
    gpt_ok_rows = [r for r in per_query if r["gpt_ok"] and not np.isnan(r["gpt_ground"])]
    oll_ok_rows = [r for r in per_query if r["ollama_ok"] and not np.isnan(r["ollama_ground"])]
    both_ok = [r for r in per_query if r["gpt_ok"] and r["ollama_ok"]]

    gpt_grounds = [r["gpt_ground"] for r in both_ok]
    oll_grounds = [r["ollama_ground"] for r in both_ok]
    kw_grounds_both = [r["baseline_ground"] for r in both_ok]
    kw_gpt = [r["baseline_ground"] for r in gpt_ok_rows]
    kw_oll = [r["baseline_ground"] for r in oll_ok_rows]
    gpt_for_kw = [r["gpt_ground"] for r in gpt_ok_rows]
    oll_for_kw = [r["ollama_ground"] for r in oll_ok_rows]

    gpt_success = sum(1 for r in per_query if r["gpt_ok"]) / len(per_query)
    oll_success = sum(1 for r in per_query if r["ollama_ok"]) / len(per_query)

    # --- Pearson ---
    def _pearson(a: list[float], b: list[float]) -> tuple[float, float]:
        if len(a) < 3:
            return float("nan"), float("nan")
        r, p = pearsonr(a, b)
        return float(r), float(p)

    r_gpt_oll, p_gpt_oll = _pearson(gpt_grounds, oll_grounds)
    r_gpt_kw, p_gpt_kw = _pearson(gpt_for_kw, kw_gpt)
    r_oll_kw, p_oll_kw = _pearson(oll_for_kw, kw_oll)

    def _mae_std(a: list[float], b: list[float]) -> tuple[float, float]:
        diffs = [abs(x - y) for x, y in zip(a, b)]
        return float(np.mean(diffs)), float(np.std(diffs))

    mae_go, std_go = _mae_std(gpt_grounds, oll_grounds)
    mae_gk, std_gk = _mae_std(gpt_for_kw, kw_gpt)
    mae_ok, std_ok = _mae_std(oll_for_kw, kw_oll)

    print(f"\n--- Three Pearsons (n for each pair) ---")
    print(f"  Pearson(GPT, Ollama):   r={r_gpt_oll:+.4f}  p={p_gpt_oll:.4f}  "
          f"MAE={mae_go:.4f}±{std_go:.4f}  n={len(both_ok)}")
    print(f"  Pearson(GPT, baseline): r={r_gpt_kw:+.4f}  p={p_gpt_kw:.4f}  "
          f"MAE={mae_gk:.4f}±{std_gk:.4f}  n={len(gpt_ok_rows)}")
    print(f"  Pearson(Ollama, baseline): r={r_oll_kw:+.4f}  p={p_oll_kw:.4f}  "
          f"MAE={mae_ok:.4f}±{std_ok:.4f}  n={len(oll_ok_rows)}")

    print(f"\n--- Component score summary ---")
    all_kw = [r["baseline_ground"] for r in per_query]
    print(f"  baseline_ground: mean={np.mean(all_kw):.4f}  std={np.std(all_kw):.4f}  "
          f"min={np.min(all_kw):.4f}  max={np.max(all_kw):.4f}")
    if gpt_grounds:
        print(f"  gpt_ground:      mean={np.mean(gpt_grounds):.4f}  std={np.std(gpt_grounds):.4f}  "
              f"min={np.min(gpt_grounds):.4f}  max={np.max(gpt_grounds):.4f}")
    if oll_grounds:
        print(f"  ollama_ground:   mean={np.mean(oll_grounds):.4f}  std={np.std(oll_grounds):.4f}  "
              f"min={np.min(oll_grounds):.4f}  max={np.max(oll_grounds):.4f}")
    print(f"  GPT structured-output success: {gpt_success:.0%} ({sum(r['gpt_ok'] for r in per_query)}/10)")
    print(f"  Ollama structured-output success: {oll_success:.0%} ({sum(r['ollama_ok'] for r in per_query)}/10)")

    # --- Latency ---
    gpt_lats = [r["gpt_elapsed"] * 1000 for r in per_query if r["gpt_ok"]]
    oll_lats = [r["ollama_elapsed"] * 1000 for r in per_query if r["ollama_ok"]]
    print(f"\n--- Latency (ms) ---")
    print(f"  GPT-4o-mini:    mean={np.mean(gpt_lats):.0f}  std={np.std(gpt_lats):.0f}  "
          f"min={np.min(gpt_lats):.0f}  max={np.max(gpt_lats):.0f}")
    print(f"  Ollama qwen3:8b: mean={np.mean(oll_lats):.0f}  std={np.std(oll_lats):.0f}  "
          f"min={np.min(oll_lats):.0f}  max={np.max(oll_lats):.0f}")
    lat_ratio = np.mean(oll_lats) / max(np.mean(gpt_lats), 1.0)
    print(f"  Ratio (Ollama/GPT): {lat_ratio:.2f}x  "
          f"({'Ollama faster' if lat_ratio < 1.0 else 'Ollama slower'})")

    # --- Weakest-dimension attribution ---
    gpt_wd_acc = sum(r["gpt_wd_correct"] for r in per_query if r["gpt_ok"])
    oll_wd_acc = sum(r["oll_wd_correct"] for r in per_query if r["ollama_ok"])
    wd_agree_n = sum(r["wd_agree"] for r in per_query if r["gpt_ok"] and r["ollama_ok"])
    n_both = sum(1 for r in per_query if r["gpt_ok"] and r["ollama_ok"])

    print(f"\n--- Weakest-dimension attribution ---")
    print(f"  GPT accuracy:    {gpt_wd_acc}/{sum(r['gpt_ok'] for r in per_query)} "
          f"({gpt_wd_acc/max(sum(r['gpt_ok'] for r in per_query),1):.0%})")
    print(f"  Ollama accuracy: {oll_wd_acc}/{sum(r['ollama_ok'] for r in per_query)} "
          f"({oll_wd_acc/max(sum(r['ollama_ok'] for r in per_query),1):.0%})")
    print(f"  Inter-model agreement: {wd_agree_n}/{n_both} ({wd_agree_n/max(n_both,1):.0%})")
    print(f"\n  Per-query detail:")
    print(f"  {'Query':<8} {'KW_g':>6} {'GPT_g':>6} {'OL_g':>6} "
          f"{'GPT_wd':<14} {'Corr?':>5} {'OL_wd':<14} {'Corr?':>5} {'Agree':>6}")
    for r in per_query:
        gs = f"{r['gpt_ground']:.4f}" if r["gpt_ok"] else "fail"
        os_ = f"{r['ollama_ground']:.4f}" if r["ollama_ok"] else "fail"
        print(f"  {r['id']:<8} {r['baseline_ground']:>6.4f} {gs:>6} {os_:>6} "
              f"{r['gpt_wd']:<14} {'✓' if r['gpt_wd_correct'] else '✗':>5} "
              f"{r['ollama_wd']:<14} {'✓' if r['oll_wd_correct'] else '✗':>5} "
              f"{'✓' if r['wd_agree'] else '✗':>6}")

    # --- Hypothesis evaluation ---
    print(f"\n--- Hypothesis evaluation ---")
    print(f"  H1 (Pearson(GPT,Ollama)≈0.85-0.95): r={r_gpt_oll:.4f} → "
          f"{'CONFIRMED' if r_gpt_oll >= 0.85 else 'REFUTED' if r_gpt_oll < 0.50 else 'PARTIAL'}")
    print(f"  H2 (Pearson(GPT,baseline)≈0.70-0.85): r={r_gpt_kw:.4f} → "
          f"{'CONFIRMED' if 0.70 <= r_gpt_kw <= 0.85 else 'PARTIAL' if r_gpt_kw > 0.50 else 'REFUTED'}")
    print(f"  H3 (Pearson(Ollama,baseline)≈similar to H2): r={r_oll_kw:.4f} → "
          f"{'CONFIRMED' if abs(r_oll_kw - r_gpt_kw) < 0.15 else 'PARTIAL'}")
    print(f"  H4 (Ollama success≥95%): {oll_success:.0%} → "
          f"{'CONFIRMED' if oll_success >= 0.95 else 'REFUTED'}")

    # --- Four-way decision ---
    _four_way_decision(r_gpt_oll, oll_success, lat_ratio, r_gpt_kw, r_oll_kw)

    # --- Chart ---
    _save_chart(per_query, gpt_lats, oll_lats, r_gpt_oll, r_gpt_kw, r_oll_kw)
    print(f"\nChart saved: {CHART_PATH}")
    print(sep)


# ---------------------------------------------------------------------------
# Four-way decision
# ---------------------------------------------------------------------------

def _four_way_decision(
    r_gpt_oll: float,
    oll_success: float,
    lat_ratio: float,
    r_gpt_kw: float,
    r_oll_kw: float,
) -> None:
    print(f"\n--- Four-way decision ---")
    structured_fail = (1.0 - oll_success) > STRUCTURED_FAIL_THRESHOLD

    if r_gpt_oll >= HIGH_AGREEMENT and not structured_fail:
        band = "HIGH AGREEMENT"
        if lat_ratio < 1.0:
            latency_desc = f"Ollama {1/lat_ratio:.1f}x faster (ratio={lat_ratio:.2f}x)"
            rec = (
                "HIGH AGREEMENT + faster local → STRONG case for dev/prod split. "
                "Recommend Ollama for dev (scoring), GPT-4o-mini for prod. "
                "ADR-006 framing: two ADRs — one for dev/prod split (6e), one for "
                "methodology cluster (Phases 2, 4, 5)."
            )
        else:
            latency_desc = f"Ollama {lat_ratio:.1f}x slower (ratio={lat_ratio:.2f}x)"
            rec = (
                "HIGH AGREEMENT but Ollama slower → parity on quality, tradeoff on latency. "
                "Recommend GPT-4o-mini for both until local hardware improves, OR Ollama "
                "for offline/batch dev workflows where latency is not a constraint."
            )
    elif r_gpt_oll >= LOW_AGREEMENT and not structured_fail:
        band = "MEDIUM AGREEMENT"
        latency_desc = f"ratio={lat_ratio:.2f}x"
        rec = (
            "MEDIUM AGREEMENT → LLMs broadly agree on rank order but differ on absolute "
            "scores. Consider merging 6e with methodology cluster in a single ADR: "
            "'local is fast but scores diverge on edge cases.' "
            "Recommend GPT-4o-mini for prod; Ollama for dev with the caveat that "
            "absolute scores may drift from prod baseline."
        )
    else:
        band = "LOW AGREEMENT"
        latency_desc = f"ratio={lat_ratio:.2f}x"
        reason = "structured-output failure rate > 5%" if structured_fail else f"Pearson={r_gpt_oll:.4f} < {LOW_AGREEMENT}"
        rec = (
            f"LOW AGREEMENT ({reason}) → 6e collapses into the methodology cluster as "
            "a fourth instance of measurement-design limits. "
            "Recommend GPT-4o-mini for both environments. "
            "ADR-006 framing: single ADR covering methodology cluster including 6e."
        )

    print(f"  Band: {band}")
    print(f"  Latency (Ollama/GPT): {latency_desc}")
    print(f"  Pearson(GPT,Ollama)={r_gpt_oll:.4f}  "
          f"Pearson(GPT,baseline)={r_gpt_kw:.4f}  "
          f"Pearson(Ollama,baseline)={r_oll_kw:.4f}")
    print(f"\n  RECOMMENDATION: {rec}")


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def _save_chart(
    per_query: list[dict],
    gpt_lats: list[float],
    oll_lats: list[float],
    r_go: float,
    r_gk: float,
    r_ok: float,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    fig.suptitle(
        "Experiment 6e Run 2 — Groundedness Scoring Agreement: GPT-4o-mini vs Ollama qwen3:8b",
        fontsize=12, fontweight="bold",
    )

    # Axes
    ax_go = axes[0][0]  # GPT vs Ollama
    ax_gk = axes[0][1]  # GPT vs baseline
    ax_ok = axes[1][0]  # Ollama vs baseline
    ax_lat = axes[1][1]  # Latency

    both_ok = [r for r in per_query if r["gpt_ok"] and r["ollama_ok"]]
    qids = [r["id"] for r in both_ok]
    gpt_g = [r["gpt_ground"] for r in both_ok]
    oll_g = [r["ollama_ground"] for r in both_ok]
    kw_g = [r["baseline_ground"] for r in both_ok]

    for ax, xs, ys, xlabel, ylabel, title_r, color in [
        (ax_go, gpt_g, oll_g, "GPT-4o-mini groundedness", "Ollama qwen3:8b groundedness",
         f"Pearson(GPT, Ollama) r={r_go:.4f}", "#1f77b4"),
        (ax_gk, gpt_g, kw_g, "GPT-4o-mini groundedness", "Baseline (embedding-cosine) groundedness",
         f"Pearson(GPT, baseline) r={r_gk:.4f}", "#ff7f0e"),
        (ax_ok, oll_g, kw_g, "Ollama qwen3:8b groundedness", "Baseline (embedding-cosine) groundedness",
         f"Pearson(Ollama, baseline) r={r_ok:.4f}", "#2ca02c"),
    ]:
        ax.scatter(xs, ys, color=color, s=60, zorder=3)
        lo = min(min(xs), min(ys)) - 0.05
        hi = max(max(xs), max(ys)) + 0.05
        lo, hi = max(0.0, lo), min(1.0, hi)
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.0, label="y=x")
        for i, (x, y, qid) in enumerate(zip(xs, ys, qids)):
            ax.annotate(qid, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=6)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title_r, fontsize=9)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    # Latency bar
    n = len(gpt_lats)
    x = np.arange(n)
    lat_qids = [r["id"] for r in per_query if r["gpt_ok"]]
    ax_lat.bar(x - 0.2, gpt_lats, 0.4, label=f"GPT-4o-mini (mean={np.mean(gpt_lats):.0f}ms)",
               color="#1f77b4", alpha=0.8)
    ax_lat.bar(x + 0.2, oll_lats[:n], 0.4, label=f"Ollama qwen3:8b (mean={np.mean(oll_lats):.0f}ms)",
               color="#ff7f0e", alpha=0.8)
    ax_lat.set_xticks(x)
    ax_lat.set_xticklabels(lat_qids[:n], rotation=45, ha="right", fontsize=8)
    ax_lat.set_ylabel("Latency (ms)")
    lat_ratio = np.mean(oll_lats) / max(np.mean(gpt_lats), 1.0)
    ax_lat.set_title(f"Explanation latency (ratio Ollama/GPT = {lat_ratio:.2f}x)", fontsize=9)
    ax_lat.legend(fontsize=8)
    ax_lat.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(CHART_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
