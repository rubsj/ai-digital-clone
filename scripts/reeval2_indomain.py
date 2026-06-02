"""Re-evaluation harness — in-domain only, 3 passes (ADR-018 Phase 1.6.5).

Runs the 14 in-domain queries through both leaders for 3 passes (84 records).
GatekeeperAgent is now deterministic (ADR-018). Writes to
results/evaluation_day12_reeval2.json; prior run files are preserved unchanged.

Call-count estimate: 42 pairs × ~14 LLM calls per pair ≈ 588 chat completions
plus ~84 batched embed_openai calls.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval.harness import IN_DOMAIN_CATEGORIES, run_leader_pair

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

QUERIES_PATH = Path("data/eval/queries.json")
OUTPUT_PATH = Path("results/evaluation_day12_reeval2.json")
N_PASSES = 3


def main() -> None:
    with open(QUERIES_PATH) as f:
        all_queries = json.load(f)

    in_domain = [q for q in all_queries if q["category"] in IN_DOMAIN_CATEGORIES]
    logger.info("Loaded %d in-domain queries (OOD excluded per ADR-017)", len(in_domain))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_records: list[dict] = []
    total_pairs = len(in_domain) * N_PASSES
    pair_idx = 0

    for pass_num in range(1, N_PASSES + 1):
        logger.info("=== PASS %d: %d in-domain queries ===", pass_num, len(in_domain))
        for q in in_domain:
            pair_idx += 1
            t0 = perf_counter()
            logger.info(
                "Pass %d [%d/%d] %s (%s)",
                pass_num, pair_idx, total_pairs, q["id"], q["category"],
            )
            result = run_leader_pair(q["query"])
            elapsed = round(perf_counter() - t0, 1)

            record = {
                "pass": pass_num,
                "query_id": q["id"],
                "query": q["query"],
                "category": q["category"],
                "axis": "in_domain",
                "expected_behavior": q["expected_behavior"],
                "retriever_call_count": result["retriever_call_count"],
                "torvalds": result["torvalds"],
                "kroah_hartman": result["kroah_hartman"],
                "pair_elapsed_s": elapsed,
            }
            all_records.append(record)

            with open(OUTPUT_PATH, "w") as f:
                json.dump(all_records, f, indent=2, default=str)

            logger.info(
                "  T=%-8s KH=%-8s  gs_T=%.3f gs_KH=%.3f  flags_T=%s flags_KH=%s  (%.1fs)",
                result["torvalds"]["decision"],
                result["kroah_hartman"]["decision"],
                result["torvalds"]["groundedness_score"] or 0,
                result["kroah_hartman"]["groundedness_score"] or 0,
                result["torvalds"]["flags"],
                result["kroah_hartman"]["flags"],
                elapsed,
            )

    logger.info(
        "=== RE-EVAL COMPLETE: %d records written to %s ===",
        len(all_records), OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
