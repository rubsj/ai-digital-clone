"""Analyze re-evaluation results for STOP GATE 1.5b.

Reads results/evaluation_day12_reeval.json and produces:
- In-domain 2x2 grid per leader per pass
- Per-leader deliver rate vs ADR-015 floors (T>=6/14, KH>=5/14)
- 3-run variance
- Flag-clean deliver check (RC-2 read)
- Low-confidence-only-blocked count
- RC-3 (fabricated trigger_category) recurrence check
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

REEVAL_PATH = Path("results/evaluation_day12_reeval.json")
TORVALDS_FLOOR = 6 / 14   # 42.9%
KH_FLOOR = 5 / 14          # 35.7%
VALID_TRIGGER_CATEGORIES = {
    "low_groundedness", "off_domain", "hallucination_risk",
    "chunk_mismatch", "empty_retrieval",
}
GROUNDEDNESS_MIN = 0.60
STYLE_MIN = 0.70
CONFIDENCE_MIN = 0.80


def compute_flags(rec: dict) -> list[str]:
    flags = []
    if rec["groundedness_score"] < GROUNDEDNESS_MIN:
        flags.append("low_groundedness")
    if rec["style_score"] < STYLE_MIN:
        flags.append("low_style")
    if rec["confidence_score"] < CONFIDENCE_MIN:
        flags.append("low_confidence")
    return flags


def main() -> None:
    with open(REEVAL_PATH) as f:
        records = json.load(f)

    print(f"Total records loaded: {len(records)}")
    passes = sorted({r["pass"] for r in records})
    print(f"Passes: {passes}")

    # ─── Per-pass per-leader deliver/fallback grid ───────────────────────────
    print("\n" + "=" * 72)
    print("IN-DOMAIN 2x2 GRID (per pass, per leader)")
    print("=" * 72)

    for pass_num in passes:
        pass_recs = [r for r in records if r["pass"] == pass_num]
        print(f"\n--- Pass {pass_num} ({len(pass_recs)} query pairs) ---")
        for leader in ("torvalds", "kroah_hartman"):
            deliver = sum(1 for r in pass_recs if r[leader]["decision"] == "deliver")
            fallback = sum(1 for r in pass_recs if r[leader]["decision"] == "fallback")
            total = deliver + fallback
            rate = deliver / total if total else 0
            floor = TORVALDS_FLOOR if leader == "torvalds" else KH_FLOOR
            floor_pct = floor * 100
            cleared = rate >= floor
            print(f"  {leader:<16} deliver={deliver}/{total} ({rate*100:.1f}%)  "
                  f"floor={floor_pct:.1f}%  {'CLEARS' if cleared else 'BELOW FLOOR'}")

    # ─── Per-leader deliver rate across all passes ───────────────────────────
    print("\n" + "=" * 72)
    print("PER-LEADER DELIVER RATE — ALL 3 PASSES")
    print("=" * 72)
    for leader in ("torvalds", "kroah_hartman"):
        floor = TORVALDS_FLOOR if leader == "torvalds" else KH_FLOOR
        rates = []
        for pass_num in passes:
            pass_recs = [r for r in records if r["pass"] == pass_num]
            deliver = sum(1 for r in pass_recs if r[leader]["decision"] == "deliver")
            total = len(pass_recs)
            rates.append((pass_num, deliver, total, deliver / total))
        deliver_counts = [r[1] for r in rates]
        rate_vals = [r[3] for r in rates]
        print(f"\n  {leader}")
        for p, d, t, rt in rates:
            cleared = rt >= floor
            print(f"    Pass {p}: {d}/{t} ({rt*100:.1f}%)  {'CLEARS' if cleared else 'BELOW FLOOR'}")
        print(f"    Range: {min(deliver_counts)}-{max(deliver_counts)}/14 "
              f"({min(rate_vals)*100:.1f}%–{max(rate_vals)*100:.1f}%)")
        print(f"    Floor: {floor*100:.1f}%  "
              f"{'ALL PASSES CLEAR' if min(rate_vals)>=floor else 'FLOOR NOT CLEARED (worst pass)'}")

    # ─── Flag-clean deliver check (RC-2 read) ────────────────────────────────
    print("\n" + "=" * 72)
    print("FLAG-CLEAN DELIVER CHECK (RC-2 READ)")
    print("= A flag-clean record has gs>=0.60 + no low_groundedness,")
    print("  ss>=0.70 + no low_style, AND no low_confidence.")
    print("= Partially-clean: only low_confidence flag (groundedness and style clean)")
    print("=" * 72)

    flag_clean_fallbacks = []
    flag_clean_delivers = []
    partial_clean_fallbacks = []   # clean on gs+style, blocked only by low_confidence

    for rec in records:
        qid = rec["query_id"]
        for leader in ("torvalds", "kroah_hartman"):
            lr = rec[leader]
            flags = compute_flags(lr)
            decision = lr["decision"]
            gs = lr["groundedness_score"]
            ss = lr["style_score"]
            cs = lr["confidence_score"]

            gs_clean = gs >= GROUNDEDNESS_MIN
            ss_clean = ss >= STYLE_MIN
            fully_clean = len(flags) == 0
            confidence_only = flags == ["low_confidence"]
            gs_and_style_clean = gs_clean and ss_clean

            if fully_clean:
                row = (rec["pass"], qid, leader, decision, gs, ss, cs, flags)
                if decision == "deliver":
                    flag_clean_delivers.append(row)
                else:
                    flag_clean_fallbacks.append(row)

            if confidence_only and gs_and_style_clean:
                partial_clean_fallbacks.append(
                    (rec["pass"], qid, leader, decision, gs, ss, cs, flags,
                     lr.get("trigger_category"), lr.get("routing_reasoning","")[:120])
                )

    print(f"\nFully flag-clean records: {len(flag_clean_delivers)+len(flag_clean_fallbacks)}")
    print(f"  → DELIVER:  {len(flag_clean_delivers)}")
    print(f"  → FALLBACK: {len(flag_clean_fallbacks)}  ← RC-2 evidence if any")

    if flag_clean_delivers:
        print("\n  Flag-clean DELIVERS:")
        for p, q, l, d, gs, ss, cs, f in flag_clean_delivers:
            print(f"    pass={p} {q} {l:<15} gs={gs:.3f} ss={ss:.3f} cs={cs:.3f}")

    if flag_clean_fallbacks:
        print("\n  Flag-clean FALLBACKS (RC-2):")
        for p, q, l, d, gs, ss, cs, f in flag_clean_fallbacks:
            lr = next(r[l] for r in records if r["pass"]==p and r["query_id"]==q)
            reason = lr.get("routing_reasoning", "")[:200]
            tc = lr.get("trigger_category")
            print(f"    pass={p} {q} {l:<15} gs={gs:.3f} ss={ss:.3f} cs={cs:.3f}  tc={tc}")
            print(f"      routing: {reason[:150]}")

    print(f"\nPartially-clean (gs+style clean, only low_confidence blocked): {len(partial_clean_fallbacks)}")
    if partial_clean_fallbacks:
        for p, q, l, d, gs, ss, cs, f, tc, reason in partial_clean_fallbacks[:10]:
            print(f"  pass={p} {q} {l:<15} gs={gs:.3f} ss={ss:.3f} cs={cs:.3f} flags={f}  "
                  f"decision={d}  tc={tc}")

    # ─── Low-confidence-only-blocked count ───────────────────────────────────
    print("\n" + "=" * 72)
    print("LOW-CONFIDENCE-ONLY-BLOCKED (groundedness and style clear, cs < 0.80)")
    print("=" * 72)
    conf_only_blocked = []
    for rec in records:
        for leader in ("torvalds", "kroah_hartman"):
            lr = rec[leader]
            flags = compute_flags(lr)
            if flags == ["low_confidence"] and lr["decision"] == "fallback":
                conf_only_blocked.append((rec["pass"], rec["query_id"], leader,
                                          lr["groundedness_score"],
                                          lr["style_score"],
                                          lr["confidence_score"]))
    print(f"Records blocked SOLELY by low_confidence (fallback only): {len(conf_only_blocked)}")
    unique_queries = {(q, l) for _, q, l, *_ in conf_only_blocked}
    print(f"Unique (query, leader) pairs: {len(unique_queries)}")
    if conf_only_blocked:
        for p, q, l, gs, ss, cs in conf_only_blocked:
            print(f"  pass={p} {q} {l:<15} gs={gs:.3f} ss={ss:.3f} cs={cs:.3f}")

    # ─── RC-3: fabricated trigger_category ───────────────────────────────────
    print("\n" + "=" * 72)
    print("RC-3 CHECK: trigger_category alignment with actual flags")
    print("= RC-3 was q14 Torvalds pass-1: trigger_category=low_groundedness")
    print("  when flags were ['low_confidence','low_style'] — no low_groundedness flag.")
    print("= Check: any fallback record where trigger_category=low_groundedness")
    print("  but no low_groundedness flag in the actual deterministic flags.")
    print("=" * 72)
    rc3_instances = []
    trigger_cat_distribution: dict[str, int] = {}

    for rec in records:
        for leader in ("torvalds", "kroah_hartman"):
            lr = rec[leader]
            if lr["decision"] != "fallback":
                continue
            tc = lr.get("trigger_category")
            flags = compute_flags(lr)
            if tc:
                trigger_cat_distribution[tc] = trigger_cat_distribution.get(tc, 0) + 1
            # RC-3: trigger_category=low_groundedness but no low_groundedness in flags
            if tc == "low_groundedness" and "low_groundedness" not in flags:
                rc3_instances.append((rec["pass"], rec["query_id"], leader,
                                      lr["groundedness_score"], flags, tc,
                                      lr.get("routing_reasoning","")[:120]))

    print(f"\nRC-3 instances (tc=low_groundedness, no low_groundedness flag): {len(rc3_instances)}")
    if rc3_instances:
        print("  RECURS:")
        for p, q, l, gs, f, tc, reason in rc3_instances:
            print(f"  pass={p} {q} {l:<15} gs={gs:.3f} flags={f} tc={tc}")
            print(f"    reasoning: {reason}")
    else:
        print("  NOT RECURRED — RC-3 resolved with the flag fix.")

    print(f"\ntrigger_category distribution (fallback records):")
    for tc, count in sorted(trigger_cat_distribution.items(), key=lambda x: -x[1]):
        print(f"  {tc:<30} {count}")

    # ─── trigger_category integrity assertion ────────────────────────────────
    print("\n" + "=" * 72)
    print("TRIGGER_CATEGORY INTEGRITY ASSERTION")
    print("=" * 72)
    violations = []
    for rec in records:
        for leader in ("torvalds", "kroah_hartman"):
            lr = rec[leader]
            decision = lr["decision"]
            tc = lr.get("trigger_category")
            # non-null iff fallback
            if decision == "deliver" and tc is not None:
                violations.append((rec["pass"], rec["query_id"], leader,
                                    "deliver but tc non-null", tc))
            elif decision == "fallback" and tc is None:
                violations.append((rec["pass"], rec["query_id"], leader,
                                    "fallback but tc is None", tc))
            elif decision == "fallback" and tc not in VALID_TRIGGER_CATEGORIES:
                violations.append((rec["pass"], rec["query_id"], leader,
                                    f"invalid tc", tc))
    if violations:
        print(f"FAIL: {len(violations)} violations")
        for v in violations[:10]:
            print(f"  {v}")
    else:
        print(f"PASS: no violations across {len(records)} pair-records")


if __name__ == "__main__":
    main()
