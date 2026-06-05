#!/usr/bin/env python3
"""
W1b.2 — GROUNDEDNESS_MIN threshold derivation on HHEM's scale.
DERIVE AND SURFACE ONLY. Does NOT write GROUNDEDNESS_MIN.
Zero paid API calls — all local inference.

In-domain HHEM scores: reused from bakeoff_hhem_isolated_day14.json (per-query means,
averaged over 3 passes). OOD HHEM scores: computed fresh with vendored model (individual
scores were never stored).

Pre-registered method:
  - Positive class (fallback): oracle-ungrounded in-domain + OOD records
  - Negative class (deliver-worthy): oracle-grounded in-domain
  - Oracle-grounded: oracle_gf >= 0.50 (majority of spans strictly grounded per Opus markup)
  - Score: HHEM V0 aggregation (per-sentence max over top-5 chunks, mean over sentences)
  - Split: query-level (held-equal 7 queries = train, non-held-equal 7 = held-out)
  - Selection: maximize grounded-deliver-rate s.t. fallback-recall >= 0.90 (safety gate)
  - Comparator: Youden's J threshold (unconstrained)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).parent.parent

# Held-equal queries (same 7 used in the bake-off) → used as TRAIN split
HELD_EQUAL = {"q01", "q02", "q04", "q05", "q06", "q12", "q13"}
# Non-held-equal queries → used as HELD-OUT split
NON_HELD_EQUAL = {"q03", "q07", "q08", "q09", "q10", "q11", "q14"}

ORACLE_GF_CUTOFF = 0.50   # oracle_gf < 0.50 → oracle-ungrounded (fallback class)
SAFETY_MIN_RECALL = 0.90  # safety constraint: fallback-recall must be >= this

_HHEM_HUB_ID = "vectara/hallucination_evaluation_model"
_HHEM_REVISION = "8e4a2e6e96c708cc76c2344f7e4757df2515292c"
_MIN_SENTENCE_CHARS = 10


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


def split_sentences(text: str) -> list[str]:
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if len(s.strip()) >= _MIN_SENTENCE_CHARS]


def load_hhem_vendored():
    """Load vendored HHEM model using the correct predict() call path."""
    sys.path.insert(0, str(ROOT))
    from src.evaluation.hhem.modeling_hhem_v2 import HHEMv2ForSequenceClassification
    print("Loading vendored HHEM model (local_files_only)...")
    model = HHEMv2ForSequenceClassification.from_pretrained(
        _HHEM_HUB_ID,
        revision=_HHEM_REVISION,
        local_files_only=True,
    )
    model.eval()
    print("  HHEM loaded OK")
    return model


def hhem_v0_score(model, response_text: str, chunks: list[str], top_k: int = 5) -> float:
    """V0 aggregation: per-sentence max over top-k chunks, mean over sentences."""
    if not response_text or not chunks:
        return 0.0
    sentences = split_sentences(response_text)
    if not sentences:
        return 0.0
    top_chunks = chunks[:top_k]
    per_sentence_max: list[float] = []
    for sentence in sentences:
        pairs = [(chunk, sentence) for chunk in top_chunks]
        with torch.no_grad():
            raw_scores = model.predict(pairs)
        per_sentence_max.append(float(raw_scores.max().item()))
    return float(np.mean(per_sentence_max))


def score_ood_records(ood_records: list[dict], model) -> list[dict]:
    """Score all OOD records with HHEM V0 aggregation."""
    ood_scored = []
    for i, rec in enumerate(ood_records):
        qid = rec["query_id"]
        for leader in ("torvalds", "kroah_hartman"):
            lr = rec[leader]
            text = lr.get("clone_response_text") or ""
            chunks = [c["content"] for c in lr.get("chunk_contents", [])[:5]]
            score = hhem_v0_score(model, text, chunks, top_k=5)
            ood_scored.append({"query_id": qid, "leader": leader,
                               "hhem_score": round(score, 4), "label": "fallback"})
            print(f"  OOD {qid} {leader}: {score:.4f}")
    return ood_scored


def build_indomain_records(bakeoff: dict) -> list[dict]:
    """
    Extract per-query-mean HHEM scores and oracle_gf from the bakeoff artifact.
    Returns list of {query_id, leader, hhem_score, oracle_gf, oracle_label}.
    """
    gate2 = bakeoff["gates"]["gate2"]["per_query"]
    records = []
    for qid, entry in gate2.items():
        for leader, score_key, gf_key in [
            ("torvalds", "model_t", "oracle_t_gf"),
            ("kroah_hartman", "model_kh", "oracle_kh_gf"),
        ]:
            hhem_score = entry[score_key]
            oracle_gf = entry[gf_key]
            oracle_label = "deliver" if oracle_gf >= ORACLE_GF_CUTOFF else "fallback"
            records.append({
                "query_id": qid,
                "leader": leader,
                "hhem_score": round(hhem_score, 4),
                "oracle_gf": round(oracle_gf, 4),
                "oracle_label": oracle_label,
                "split": "train" if qid in HELD_EQUAL else "heldout",
            })
    return records


def threshold_sweep(scores_deliver: list[float], scores_fallback: list[float],
                    n_thresholds: int = 200) -> list[dict]:
    """
    Sweep thresholds from min to max of all scores.
    Returns list of {threshold, fallback_recall, grounded_deliver_rate, youden_j}.

    fallback_recall = fraction of fallback class scoring < threshold (routed to fallback)
    grounded_deliver_rate = fraction of deliver class scoring >= threshold (routed to deliver)
    """
    all_scores = sorted(set(scores_deliver + scores_fallback))
    # Add sentinel thresholds just below min and above max
    lo = min(all_scores) - 0.001
    hi = max(all_scores) + 0.001
    candidates = [lo] + all_scores + [hi]
    # Also add evenly-spaced candidates
    grid = np.linspace(lo, hi, n_thresholds).tolist()
    thresholds = sorted(set(candidates + grid))

    rows = []
    for t in thresholds:
        fr = (sum(1 for s in scores_fallback if s < t) / len(scores_fallback)
              if scores_fallback else 0.0)
        gdr = (sum(1 for s in scores_deliver if s >= t) / len(scores_deliver)
               if scores_deliver else 0.0)
        j = fr + gdr - 1.0
        rows.append({"threshold": round(t, 4),
                     "fallback_recall": round(fr, 4),
                     "grounded_deliver_rate": round(gdr, 4),
                     "youden_j": round(j, 4)})
    return rows


def find_safety_constrained(rows: list[dict]) -> dict | None:
    """
    Safety-constrained: highest threshold with fallback_recall >= SAFETY_MIN_RECALL,
    which maximizes grounded_deliver_rate.
    """
    candidates = [r for r in rows if r["fallback_recall"] >= SAFETY_MIN_RECALL]
    if not candidates:
        return None
    # Among those satisfying the safety constraint, maximize GDR (higher T = higher GDR
    # so take the highest T in the feasible set)
    return max(candidates, key=lambda r: (r["grounded_deliver_rate"], r["threshold"]))


def find_youden(rows: list[dict]) -> dict:
    """Unconstrained Youden's J maximizing threshold."""
    return max(rows, key=lambda r: r["youden_j"])


def per_leader_deliver_rate(indomain: list[dict], threshold: float, split: str | None = None) -> dict:
    """
    At a given threshold, compute per-leader deliver rate on the HELD-EQUAL queries only.
    Deliver rate = fraction of oracle-grounded responses scoring >= threshold (correct delivers).
    """
    for_analysis = indomain
    if split:
        for_analysis = [r for r in indomain if r["split"] == split]
    he_queries = {"q01", "q02", "q04", "q05", "q06", "q12", "q13"}
    he_records = [r for r in for_analysis if r["query_id"] in he_queries
                  and r["oracle_label"] == "deliver"]

    results = {}
    for leader in ("torvalds", "kroah_hartman"):
        lr_records = [r for r in he_records if r["leader"] == leader]
        if not lr_records:
            results[leader] = None
            continue
        deliver_rate = sum(1 for r in lr_records if r["hhem_score"] >= threshold) / len(lr_records)
        results[leader] = {
            "deliver_rate": round(deliver_rate, 4),
            "n_oracle_grounded": len(lr_records),
            "n_correctly_delivered": sum(1 for r in lr_records if r["hhem_score"] >= threshold),
            "scores": [r["hhem_score"] for r in lr_records],
        }
    if results.get("torvalds") and results.get("kroah_hartman"):
        results["gap"] = round(
            results["torvalds"]["deliver_rate"] - results["kroah_hartman"]["deliver_rate"], 4
        )
    return results


def run_split_analysis(indomain: list[dict], ood_records: list[dict], split_name: str,
                       split_key: str) -> dict:
    """
    Run threshold analysis for a given split.
    split_key: 'train' or 'heldout'
    """
    split_indomain = [r for r in indomain if r["split"] == split_key]
    # All OOD always included
    ood_scores = [r["hhem_score"] for r in ood_records]

    scores_deliver = [r["hhem_score"] for r in split_indomain if r["oracle_label"] == "deliver"]
    scores_fallback_indomain = [r["hhem_score"] for r in split_indomain if r["oracle_label"] == "fallback"]
    scores_fallback = scores_fallback_indomain + ood_scores

    print(f"\n  {split_name} split: {len(scores_deliver)} deliver, "
          f"{len(scores_fallback_indomain)} in-domain fallback, "
          f"{len(ood_scores)} OOD fallback, total fallback={len(scores_fallback)}")

    if not scores_fallback or not scores_deliver:
        return {"error": "Insufficient data in split"}

    rows = threshold_sweep(scores_deliver, scores_fallback)
    safety = find_safety_constrained(rows)
    youden = find_youden(rows)

    # Compact sweep: sample at 0.02 intervals for the report
    compact_rows = [r for r in rows if abs(r["threshold"] % 0.02) < 0.005 or
                    r["threshold"] in [row["threshold"] for row in [safety, youden] if row]]

    return {
        "split": split_name,
        "n_deliver": len(scores_deliver),
        "n_fallback_indomain": len(scores_fallback_indomain),
        "n_fallback_ood": len(ood_scores),
        "n_fallback_total": len(scores_fallback),
        "deliver_scores": sorted(scores_deliver),
        "fallback_scores": sorted(scores_fallback),
        "safety_constrained_threshold": safety,
        "youden_threshold": youden,
        "full_sweep": rows,
    }


def main():
    print("=" * 65)
    print("W1b.2 — GROUNDEDNESS_MIN derivation (DERIVE AND SURFACE ONLY)")
    print("API calls: 0 (all local inference, confirmed)")
    print(f"Oracle cutoff: oracle_gf >= {ORACLE_GF_CUTOFF} → deliver-worthy")
    print(f"Safety constraint: fallback-recall >= {SAFETY_MIN_RECALL}")
    print("=" * 65)

    # Load bakeoff artifact for in-domain HHEM scores and oracle labels
    bakeoff_path = ROOT / "results" / "bakeoff_hhem_isolated_day14.json"
    bakeoff = json.loads(bakeoff_path.read_text())
    indomain = build_indomain_records(bakeoff)

    print(f"\nIn-domain records: {len(indomain)} (14 queries × 2 clones, per-query means)")
    n_deliver = sum(1 for r in indomain if r["oracle_label"] == "deliver")
    n_fallback = sum(1 for r in indomain if r["oracle_label"] == "fallback")
    print(f"  Oracle-grounded (deliver):    {n_deliver}")
    print(f"  Oracle-ungrounded (fallback): {n_fallback}")
    print("  In-domain fallback records:")
    for r in indomain:
        if r["oracle_label"] == "fallback":
            print(f"    {r['query_id']} {r['leader']}: hhem={r['hhem_score']:.4f} oracle_gf={r['oracle_gf']:.4f}")

    # Load OOD records and score them with vendored HHEM
    day12_path = ROOT / "results" / "evaluation_day12.json"
    day12 = json.loads(day12_path.read_text())
    ood_records_raw = [r for r in day12 if r.get("axis") == "ood"]
    print(f"\nOOD records: {len(ood_records_raw)} (6 queries × 2 clones = 12 responses)")
    print("Scoring OOD responses with vendored HHEM...")

    model = load_hhem_vendored()
    ood_scored = score_ood_records(ood_records_raw, model)
    ood_scores_list = [r["hhem_score"] for r in ood_scored]
    print(f"  OOD HHEM range: min={min(ood_scores_list):.4f} max={max(ood_scores_list):.4f} "
          f"mean={np.mean(ood_scores_list):.4f}")

    # Run split analyses
    print("\n" + "=" * 65)
    print("THRESHOLD ANALYSIS")
    print("  Train split: held-equal queries (q01,q02,q04,q05,q06,q12,q13)")
    print("  Held-out split: non-held-equal queries (q03,q07,q08,q09,q10,q11,q14)")

    train_result = run_split_analysis(indomain, ood_scored, "Train (held-equal)", "train")
    heldout_result = run_split_analysis(indomain, ood_scored, "Held-out (non-held-equal)", "heldout")

    # Compact display
    for result in [train_result, heldout_result]:
        if "error" in result:
            print(f"\n{result['split']}: {result['error']}")
            continue
        print(f"\n{result['split']}")
        s = result["safety_constrained_threshold"]
        y = result["youden_threshold"]
        if s:
            print(f"  Safety-constrained (fallback-recall>={SAFETY_MIN_RECALL}):")
            print(f"    threshold={s['threshold']:.3f}  fallback-recall={s['fallback_recall']:.3f}  "
                  f"grounded-deliver-rate={s['grounded_deliver_rate']:.3f}  youden={s['youden_j']:.3f}")
        else:
            print(f"  Safety-constrained: NO FEASIBLE THRESHOLD (safety constraint too strict for this N)")
        print(f"  Youden's J (unconstrained):")
        print(f"    threshold={y['threshold']:.3f}  fallback-recall={y['fallback_recall']:.3f}  "
              f"grounded-deliver-rate={y['grounded_deliver_rate']:.3f}  youden={y['youden_j']:.3f}")

    # Print sweep table (0.05-interval keypoints)
    print("\n" + "=" * 65)
    print("TRAIN SPLIT — FULL THRESHOLD SWEEP (representative points)")
    print(f"{'T':>8} {'FB-Recall':>10} {'GDR':>8} {'Youden-J':>10}")
    print("-" * 40)
    sweep = train_result.get("full_sweep", [])
    displayed = set()
    for row in sweep:
        t = row["threshold"]
        # Show points at 0.05 increments plus special thresholds
        show = any(abs(t - v) < 0.003 for v in np.arange(0.0, 1.01, 0.05))
        special_thresholds = []
        if train_result.get("safety_constrained_threshold"):
            special_thresholds.append(train_result["safety_constrained_threshold"]["threshold"])
        if train_result.get("youden_threshold"):
            special_thresholds.append(train_result["youden_threshold"]["threshold"])
        show = show or any(abs(t - v) < 0.003 for v in special_thresholds)
        if show and round(t, 2) not in displayed:
            displayed.add(round(t, 2))
            marker = ""
            if train_result.get("safety_constrained_threshold") and abs(t - train_result["safety_constrained_threshold"]["threshold"]) < 0.003:
                marker += " [SAFETY]"
            if train_result.get("youden_threshold") and abs(t - train_result["youden_threshold"]["threshold"]) < 0.003:
                marker += " [YOUDEN]"
            print(f"{t:8.3f} {row['fallback_recall']:10.3f} {row['grounded_deliver_rate']:8.3f} {row['youden_j']:10.3f}{marker}")

    # Per-leader deliver-rate bias at proposed threshold
    print("\n" + "=" * 65)
    print("PER-LEADER DELIVER-RATE ON HELD-EQUAL QUERIES (bias check)")
    if train_result.get("safety_constrained_threshold"):
        t_safety = train_result["safety_constrained_threshold"]["threshold"]
        print(f"At safety-constrained threshold={t_safety:.3f}:")
        bias = per_leader_deliver_rate(indomain, t_safety)
        for leader in ("torvalds", "kroah_hartman"):
            if bias.get(leader):
                b = bias[leader]
                print(f"  {leader}: deliver_rate={b['deliver_rate']:.3f}  "
                      f"({b['n_correctly_delivered']}/{b['n_oracle_grounded']} oracle-grounded scores >= {t_safety:.3f})")
                print(f"    Scores: {b['scores']}")
        if bias.get("gap") is not None:
            print(f"  Gap (torvalds - kroah_hartman): {bias['gap']:+.4f}")

    if train_result.get("youden_threshold"):
        t_youden = train_result["youden_threshold"]["threshold"]
        if train_result.get("safety_constrained_threshold") and abs(t_safety - t_youden) > 0.01:
            print(f"At Youden threshold={t_youden:.3f}:")
            bias_y = per_leader_deliver_rate(indomain, t_youden)
            for leader in ("torvalds", "kroah_hartman"):
                if bias_y.get(leader):
                    b = bias_y[leader]
                    print(f"  {leader}: deliver_rate={b['deliver_rate']:.3f}  "
                          f"({b['n_correctly_delivered']}/{b['n_oracle_grounded']} scores >= {t_youden:.3f})")
            if bias_y.get("gap") is not None:
                print(f"  Gap (torvalds - kroah_hartman): {bias_y['gap']:+.4f}")

    # Comparison to current GROUNDEDNESS_MIN=0.60
    print("\n" + "=" * 65)
    print("COMPARISON: current GROUNDEDNESS_MIN=0.60 vs proposed threshold")
    for t in [0.60]:
        rows_match = [r for r in sweep if abs(r["threshold"] - t) < 0.005]
        if rows_match:
            row = rows_match[0]
            print(f"  threshold=0.60: fallback-recall={row['fallback_recall']:.3f}  "
                  f"GDR={row['grounded_deliver_rate']:.3f}  youden={row['youden_j']:.3f}")

    # Save results
    output = {
        "run": "w1b2_threshold_derivation",
        "date": "2026-06-04",
        "api_calls": 0,
        "method": {
            "indomain_hhem_source": "bakeoff_hhem_isolated_day14.json (per-query means, reused)",
            "ood_hhem_source": "vendored HHEMv2ForSequenceClassification, scored fresh (never stored individually)",
            "oracle_label_rule": f"oracle_gf >= {ORACLE_GF_CUTOFF} → deliver-worthy (strictly-grounded fraction per Opus markup)",
            "oracle_gf_source": "bakeoff_hhem_isolated_day14.json gate2 oracle_*_gf fields",
            "split_rule": "query-level: held-equal 7 queries = train, non-held-equal 7 = held-out",
            "safety_constraint": f"fallback-recall >= {SAFETY_MIN_RECALL}",
            "selection": "highest threshold with fallback-recall >= safety constraint (maximizes GDR)",
        },
        "indomain_records": indomain,
        "ood_scored": ood_scored,
        "train_analysis": {k: v for k, v in train_result.items() if k != "full_sweep"},
        "heldout_analysis": {k: v for k, v in heldout_result.items() if k != "full_sweep"},
        "train_sweep_table": train_result.get("full_sweep", []),
        "heldout_sweep_table": heldout_result.get("full_sweep", []),
    }
    out_path = ROOT / "results" / "w1b2_threshold_day14.json"
    out_path.write_text(json.dumps(output, indent=2, cls=NumpyEncoder))
    print(f"\nResults written to {out_path}")
    print("GROUNDEDNESS_MIN NOT modified. Surface only — awaiting Ruby confirmation.")


if __name__ == "__main__":
    main()
