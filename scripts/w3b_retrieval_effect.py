#!/usr/bin/env python3
"""W3b — Retrieval effect, isolated (cost spend, Ruby-approved).

Regenerates the 6 dedup-affected in-domain queries (q03, q07, q09, q10, q11, q14)
× 2 leaders × 3 passes through the full pipeline with the W2 dedup fix live.
HHEM@0.40 is held fixed (metric unchanged from W3a). Measures the marginal retrieval
effect as delta vs the W3a frozen-input baseline.

Output: results/w3b_retrieval_effect_day14.json (written incrementally).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("w3b")

# BLAS order fix: import HHEM (PyTorch/flan-t5) BEFORE FAISS so PyTorch's BLAS
# initializes first. Reversed order causes a segfault on macOS when T5 calls
# its first linear-algebra op into an already-initialized FAISS BLAS context.
# Monkey-patch ScoringEngine to reuse the single pre-loaded instance — avoids
# 36 separate flan-t5-base loads (one per pipeline run).
logger.info("Pre-loading HHEM model (must happen before FAISS import)...")
from src.evaluation.groundedness_scorer import HHEMGroundednessScorer as _HGSC
_shared_hhem = _HGSC()
logger.info("HHEM pre-loaded; patching ScoringEngine to reuse shared instance.")
from src.components import scoring_engine as _se_mod
_orig_se_init = _se_mod.ScoringEngine.__init__
def _patched_se_init(self, groundedness_scorer=None):  # type: ignore[misc]
    _orig_se_init(self, groundedness_scorer or _shared_hhem)
_se_mod.ScoringEngine.__init__ = _patched_se_init  # type: ignore[method-assign]

from src.eval.harness import run_leader_pair

# ── Scope ────────────────────────────────────────────────────────────────────

DEDUP_QUERY_IDS = {"q03", "q07", "q09", "q10", "q11", "q14"}
N_PASSES = 3
HHEM_THRESHOLD = 0.40
OUTPUT_PATH = ROOT / "results" / "w3b_retrieval_effect_day14.json"

# W3a HHEM values (frozen-input baseline, from w3a_metric_effect_day14.json)
W3A_HHEM = {
    "q03": {"torvalds": 0.3848, "kroah_hartman": 0.4813},
    "q07": {"torvalds": 0.2852, "kroah_hartman": 0.2524},
    "q09": {"torvalds": 0.3778, "kroah_hartman": 0.3426},
    "q10": {"torvalds": 0.5528, "kroah_hartman": 0.6004},
    "q11": {"torvalds": 0.5994, "kroah_hartman": 0.7245},
    "q14": {"torvalds": 0.3687, "kroah_hartman": 0.4414},
}
W3A_VERDICTS = {
    "q03": {"torvalds": "fallback", "kroah_hartman": "deliver"},
    "q07": {"torvalds": "fallback", "kroah_hartman": "fallback"},
    "q09": {"torvalds": "fallback", "kroah_hartman": "fallback"},
    "q10": {"torvalds": "deliver",  "kroah_hartman": "deliver"},
    "q11": {"torvalds": "deliver",  "kroah_hartman": "deliver"},
    "q14": {"torvalds": "fallback", "kroah_hartman": "deliver"},
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _verdict(hhem: float | None) -> str:
    if hhem is None:
        return "fallback"
    return "deliver" if hhem >= HHEM_THRESHOLD else "fallback"


def _save(obj: dict) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(obj, f, indent=2)


def _chunk_distinct_check(chunk_contents: list[dict]) -> dict:
    """Verify dedup fix engaged: all top-5 chunks have distinct content."""
    contents = [c["content"] for c in chunk_contents]
    n_total = len(contents)
    n_unique = len(set(contents))
    return {"total": n_total, "unique": n_unique, "all_distinct": n_total == n_unique}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> dict:
    with open(ROOT / "data" / "eval" / "queries.json") as f:
        all_queries = json.load(f)

    target_queries = [q for q in all_queries if q["id"] in DEDUP_QUERY_IDS]
    target_queries.sort(key=lambda q: q["id"])
    logger.info(
        "W3b scope: %d queries × 2 leaders × %d passes = %d pipeline runs",
        len(target_queries), N_PASSES, len(target_queries) * N_PASSES,
    )
    logger.info(
        "Estimated API spend: ~%d completions (pre-registered, Ruby-approved)",
        len(target_queries) * N_PASSES * 14,
    )

    all_pass_records: list[dict] = []
    # Accumulate per-query HHEM values across passes
    per_query_hhem: dict[str, dict[str, list[float]]] = {
        qid: {"torvalds": [], "kroah_hartman": []} for qid in DEDUP_QUERY_IDS
    }
    # Save first-pass chunk distinctness evidence
    dedup_evidence: dict[str, dict | None] = {qid: None for qid in DEDUP_QUERY_IDS}

    for pass_num in range(1, N_PASSES + 1):
        logger.info("=== PASS %d/%d ===", pass_num, N_PASSES)
        for q in target_queries:
            qid = q["id"]
            logger.info("  [%s pass=%d] %s", qid, pass_num, q["query"][:70])

            result = run_leader_pair(q["query"])

            t_hhem  = result["torvalds"]["groundedness_score"]
            kh_hhem = result["kroah_hartman"]["groundedness_score"]

            if t_hhem is not None:
                per_query_hhem[qid]["torvalds"].append(t_hhem)
            if kh_hhem is not None:
                per_query_hhem[qid]["kroah_hartman"].append(kh_hhem)

            # Record first-pass chunk distinctness for dedup verification
            if pass_num == 1:
                dedup_evidence[qid] = _chunk_distinct_check(
                    result["torvalds"]["chunk_contents"]
                )

            record = {
                "pass": pass_num,
                "query_id": qid,
                "query": q["query"],
                "torvalds": {
                    "hhem": round(t_hhem, 4) if t_hhem is not None else None,
                    "verdict": result["torvalds"]["decision"],
                },
                "kroah_hartman": {
                    "hhem": round(kh_hhem, 4) if kh_hhem is not None else None,
                    "verdict": result["kroah_hartman"]["decision"],
                },
            }
            all_pass_records.append(record)

            logger.info(
                "    T:  hhem=%.4f  (%s)  |  KH: hhem=%.4f  (%s)",
                t_hhem  or 0.0, result["torvalds"]["decision"],
                kh_hhem or 0.0, result["kroah_hartman"]["decision"],
            )

            # Incremental write after every pair so a mid-run crash loses at most one pair
            _save({"run": "w3b_retrieval_effect_day14", "status": "in_progress",
                   "all_pass_records": all_pass_records})

    # ── Summary ───────────────────────────────────────────────────────────────

    per_query_summary: dict[str, dict] = {}
    for qid in sorted(DEDUP_QUERY_IDS):
        per_query_summary[qid] = {}
        for lk in ("torvalds", "kroah_hartman"):
            vals = per_query_hhem[qid][lk]
            if not vals:
                per_query_summary[qid][lk] = {"error": "no scores collected"}
                continue

            v_min  = round(min(vals), 4)
            v_mean = round(mean(vals), 4)
            v_max  = round(max(vals), 4)
            spread = round(v_max - v_min, 4)

            w3a   = W3A_HHEM[qid][lk]
            delta  = round(v_mean - w3a, 4)
            # Is the delta within the 3-pass generation noise (spread)?
            within_noise = abs(delta) <= spread if spread > 0 else (delta == 0)

            verdicts = [p["torvalds" if lk == "torvalds" else "kroah_hartman"]["verdict"]
                        for p in all_pass_records
                        if p["query_id"] == qid]
            majority_verdict = max(set(verdicts), key=verdicts.count)

            per_query_summary[qid][lk] = {
                "passes_hhem": [round(v, 4) for v in vals],
                "min": v_min,
                "mean": v_mean,
                "max": v_max,
                "spread": spread,
                "majority_verdict": majority_verdict,
                "w3a_hhem": w3a,
                "w3a_verdict": W3A_VERDICTS[qid][lk],
                "delta_mean": delta,
                "within_noise": within_noise,
            }

    output = {
        "run": "w3b_retrieval_effect_day14",
        "date": "2026-06-05",
        "status": "complete",
        "n_passes": N_PASSES,
        "threshold_hhem": HHEM_THRESHOLD,
        "queries_scope": sorted(DEDUP_QUERY_IDS),
        "mechanism_note": (
            "W2 dedup fix live in Retriever.run() throughout this run. "
            "HHEM@0.40 held fixed (metric unchanged from W3a). "
            "Delta vs W3a isolates the retrieval contribution only. "
            "Pre-registered interpretation: if delta <= 3-pass spread, "
            "honest result is 'retrieval fix did not move grounding measurably above generation noise'."
        ),
        "dedup_evidence": dedup_evidence,
        "w3a_baseline_hhem": W3A_HHEM,
        "w3a_baseline_verdicts": W3A_VERDICTS,
        "all_pass_records": all_pass_records,
        "per_query_summary": per_query_summary,
    }

    _save(output)
    logger.info("W3b complete. Results written to %s", OUTPUT_PATH)

    # ── Console report ────────────────────────────────────────────────────────

    print("\n" + "=" * 72)
    print("W3b RETRIEVAL EFFECT REPORT (marginal delta vs W3a frozen baseline)")
    print("=" * 72)
    print(f"{'Query':<6} {'Leader':<15} {'W3a':>6} {'P1':>6} {'P2':>6} {'P3':>6} "
          f"{'min':>6} {'mean':>6} {'max':>6} {'delta':>7} {'noise?':>7} {'verdict(maj)':>13}")
    print("-" * 90)

    for qid in sorted(DEDUP_QUERY_IDS):
        for lk in ("torvalds", "kroah_hartman"):
            s = per_query_summary[qid][lk]
            if "error" in s:
                print(f"{qid:<6} {lk:<15} ERROR: {s['error']}")
                continue
            passes_str = "  ".join(f"{v:>6.4f}" for v in s["passes_hhem"])
            print(
                f"{qid:<6} {lk:<15} {s['w3a_hhem']:>6.4f}  {passes_str}  "
                f"{s['min']:>6.4f} {s['mean']:>6.4f} {s['max']:>6.4f} "
                f"{s['delta_mean']:>+7.4f} {'YES' if s['within_noise'] else 'NO':>7}  "
                f"{s['majority_verdict']:>13} (was {s['w3a_verdict']})"
            )

    print("\nDedup verification (first-pass chunks, all queries):")
    for qid in sorted(DEDUP_QUERY_IDS):
        ev = dedup_evidence.get(qid)
        if ev:
            print(f"  {qid}: {ev['unique']}/{ev['total']} distinct — "
                  f"{'OK' if ev['all_distinct'] else 'DUPLICATE FOUND!'}")

    return output


if __name__ == "__main__":
    main()
