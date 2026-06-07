#!/usr/bin/env python3
"""
W1b.0 groundedness bake-off — Day 14.
Three candidates: HHEM-2.1-Open, MiniCheck flan-t5-large, DeBERTa-v3 NLI cross-encoder.
Scored against Day-13 oracle (markup_input.json + markup_output.json).
Zero paid API calls — all local inference.

transformers==5.5.0, sentence-transformers==5.3.0
"""

import json
import re
import sys
import traceback
import numpy as np
import torch
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).parent.parent
HELD_EQUAL = {"q01", "q02", "q04", "q05", "q06", "q12", "q13"}


class NumpyEncoder(json.JSONEncoder):
    """Handle numpy scalars that json.dumps refuses to serialize."""
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    markup_in = json.loads((ROOT / "docs/experiments/day13/markup_input.json").read_text())
    markup_out = json.loads((ROOT / "docs/experiments/day13/markup_output.json").read_text())
    blind_key = json.loads((ROOT / "docs/experiments/day13/blind_key.json").read_text())
    day12 = json.loads((ROOT / "results/evaluation_day12.json").read_text())
    ood = [r for r in day12 if r.get("axis") == "ood"]
    print(f"Data loaded: {len(markup_in)} in-domain records, {len(ood)} OOD records")
    return markup_in, markup_out, blind_key, ood


def split_sentences(text):
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sents if len(s.strip()) >= 10]


# ---------------------------------------------------------------------------
# Scorer classes
# ---------------------------------------------------------------------------

def _best_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class DeBERTaScorer:
    name = "deberta_v3_nli"
    hf_id = "cross-encoder/nli-deberta-v3-base"

    def __init__(self):
        print(f"Loading {self.name}...")
        import torch
        device = _best_device()
        from sentence_transformers.cross_encoder import CrossEncoder
        self.model = CrossEncoder(self.hf_id, num_labels=3, device=str(device))
        self.model.model.eval()
        cfg = self.model.model.config
        id2label = cfg.id2label if hasattr(cfg, "id2label") else {}
        print(f"  {self.name} device={device} labels={id2label}")
        print(f"  {self.name} ready")

    def score_batch(self, pairs):
        arr = self.model.predict(pairs, apply_softmax=True, batch_size=32,
                                 show_progress_bar=False)
        # Column 1 = entailment (contradiction=0, entailment=1, neutral=2)
        return arr[:, 1].tolist()


class HHEMScorer:
    name = "hhem_2_1_open"
    hf_id = "vectara/hallucination_evaluation_model"

    def __init__(self):
        print(f"Loading {self.name}...")
        device = _best_device()
        from transformers import pipeline
        self.pipe = pipeline(
            "text-classification",
            model=self.hf_id,
            trust_remote_code=True,
            device=device,
        )
        # Probe to detect which label means consistent/factual.
        # An identical source+hypothesis pair should be maximally consistent.
        probe = self.pipe([{"text": "The cat sat on the mat.", "text_pair": "The cat sat on the mat."}])
        self._factual_label = probe[0]["label"]
        print(f"  {self.name} device={device} factual_label='{self._factual_label}' "
              f"(probe score={probe[0]['score']:.3f})")
        print(f"  {self.name} ready")

    def score_batch(self, pairs, chunk=16):
        inputs = [{"text": p, "text_pair": h} for p, h in pairs]
        results = []
        for i in range(0, len(inputs), chunk):
            outs = self.pipe(inputs[i : i + chunk])
            for out in outs:
                s = out["score"] if out["label"] == self._factual_label else 1.0 - out["score"]
                results.append(float(s))
        return results


class MiniCheckScorer:
    name = "minicheck_flan_t5_large"
    hf_id = "lytang/MiniCheck-Flan-T5-Large"

    def __init__(self):
        print(f"Loading {self.name} ({self.hf_id})...")
        self.device = _best_device()
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        self.tok = AutoTokenizer.from_pretrained(self.hf_id)
        self.model = T5ForConditionalGeneration.from_pretrained(self.hf_id)
        self.model.to(self.device)
        self.model.eval()
        self.eos = self.tok.eos_token
        self.dec_start = self.model.config.decoder_start_token_id
        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"  {self.name} device={self.device} EOS='{self.eos}' dec_start={self.dec_start} params={n_params:.0f}M")
        print(f"  {self.name} ready")

    def score_batch(self, pairs, batch_size=16):
        # Format: "predict: {doc}{eos}{claim}"
        texts = ["predict: " + p + self.eos + h for p, h in pairs]
        all_scores = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tok(batch, return_tensors="pt", max_length=1024,
                           truncation=True, padding=True)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            dec_ids = torch.full((len(batch), 1), self.dec_start,
                                 dtype=torch.long, device=self.device)
            with torch.no_grad():
                out = self.model(**enc, decoder_input_ids=dec_ids)
            logits = out.logits[:, 0, :].cpu()    # first decoder position, shape (B, vocab)
            lab_logits = logits[:, [3, 209]]       # 3=no-support, 209=support
            probs = torch.softmax(lab_logits, dim=-1)
            all_scores.extend(probs[:, 1].tolist())  # support probability
        return all_scores


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def score_indomain(markup_in, scorer):
    """
    For each (record, clone) in markup_in: per-sentence max over chunks, mean over
    sentences. Returns dict (ri, clone_key) -> (record_score, {span_idx: max_score}).
    """
    all_pairs = []
    meta = []   # (ri, clone_key, si, ci)

    for ri, rec in enumerate(markup_in):
        chunks = [c["content"] for c in rec["chunks"]]   # 5 chunks
        for clone_key in ("clone_A", "clone_B"):
            for si, span in enumerate(rec[clone_key]["spans"]):
                for ci, chunk in enumerate(chunks):
                    all_pairs.append((chunk, span["text"]))
                    meta.append((ri, clone_key, si, ci))

    print(f"  Scoring {len(all_pairs)} in-domain (chunk, span) pairs...")
    raw = scorer.score_batch(all_pairs)

    # Max over chunks for each (ri, clone_key, si)
    span_max = {}
    for (ri, ck, si, ci), score in zip(meta, raw):
        k = (ri, ck, si)
        if k not in span_max or score > span_max[k]:
            span_max[k] = score

    # Group by (ri, clone_key)
    grouped = {}
    for (ri, ck, si), ms in span_max.items():
        grouped.setdefault((ri, ck), {})[si] = ms

    return {k: (float(np.mean(list(sd.values()))), sd) for k, sd in grouped.items()}


def score_ood(ood_records, scorer):
    """
    Score OOD records using clone_response_text + chunk_contents.
    Returns dict (oi, leader) -> ood_score.
    """
    all_pairs = []
    meta = []   # (oi, leader, si, ci)

    for oi, rec in enumerate(ood_records):
        for leader in ("torvalds", "kroah_hartman"):
            lr = rec[leader]
            text = lr.get("clone_response_text") or ""
            chunks = [c["content"] for c in (lr.get("chunk_contents") or [])][:5]
            sents = split_sentences(text)
            for si, sent in enumerate(sents):
                for ci, chunk in enumerate(chunks):
                    all_pairs.append((chunk, sent))
                    meta.append((oi, leader, si, ci))

    print(f"  Scoring {len(all_pairs)} OOD (chunk, span) pairs...")
    raw = scorer.score_batch(all_pairs)

    span_max = {}
    for (oi, leader, si, ci), score in zip(meta, raw):
        k = (oi, leader, si)
        if k not in span_max or score > span_max[k]:
            span_max[k] = score

    grouped = {}
    for (oi, leader, si), ms in span_max.items():
        grouped.setdefault((oi, leader), {})[si] = ms

    return {k: float(np.mean(list(sd.values()))) for k, sd in grouped.items()}


# ---------------------------------------------------------------------------
# Oracle helpers
# ---------------------------------------------------------------------------

def oracle_gf_by_query(markup_out, blind_key):
    """
    Returns {query_id: {torvalds: mean_grounded_fraction, kroah_hartman: mean_gf}}.
    Grounded fraction = grounded_spans / total_spans, averaged across the 3 passes.
    """
    per_query = {}
    for key_str, entry in markup_out.items():
        qid, _ = key_str.rsplit("_p", 1)
        key_map = blind_key[qid]
        for clone_key in ("clone_A", "clone_B"):
            leader = key_map[clone_key[-1]]   # "A" or "B"
            spans = entry[clone_key]
            total = len(spans)
            grounded = sum(1 for v in spans.values() if v["label"] == "grounded")
            gf = grounded / total if total else 0.0
            per_query.setdefault(qid, {}).setdefault(leader, []).append(gf)

    return {qid: {l: float(np.mean(vs)) for l, vs in ls.items()}
            for qid, ls in per_query.items()}


# ---------------------------------------------------------------------------
# Gate computation
# ---------------------------------------------------------------------------

def compute_gates(markup_in, markup_out, blind_key, in_scores, ood_scores, model_name):
    oracle = oracle_gf_by_query(markup_out, blind_key)

    # Per-query mean model scores (averaged across 3 passes)
    query_model = {}
    for ri, rec in enumerate(markup_in):
        qid = rec["query_id"]
        key_map = blind_key[qid]
        for clone_key in ("clone_A", "clone_B"):
            leader = key_map[clone_key[-1]]
            score, _ = in_scores[(ri, clone_key)]
            query_model.setdefault(qid, {}).setdefault(leader, []).append(score)

    qmeans = {qid: {l: float(np.mean(vs)) for l, vs in ls.items()}
              for qid, ls in query_model.items()}

    # GATE 1 — held-equal queries: mean |score_T - score_KH| <= 0.05, no systematic lean
    g1_diffs = []
    g1_lean = {"torvalds": 0, "kroah_hartman": 0}
    g1_per_q = {}
    for qid in HELD_EQUAL:
        t = qmeans[qid]["torvalds"]
        kh = qmeans[qid]["kroah_hartman"]
        diff = abs(t - kh)
        g1_diffs.append(diff)
        direction = "torvalds" if t > kh else "kroah_hartman"
        g1_lean[direction] += 1
        g1_per_q[qid] = {"t": round(t, 4), "kh": round(kh, 4), "diff": round(diff, 4), "leans": direction}

    g1_mean_diff = float(np.mean(g1_diffs))
    g1_systematic = max(g1_lean.values()) == 7   # all 7 same direction = systematic
    g1_pass = g1_mean_diff <= 0.05 and not g1_systematic

    # GATE 2 — per-query direction agreement with oracle on >= 12/14
    g2_agree = 0
    g2_per_q = {}
    for qid in sorted(qmeans.keys()):
        t_score = qmeans[qid]["torvalds"]
        kh_score = qmeans[qid]["kroah_hartman"]
        model_t_wins = t_score > kh_score

        o_t = oracle[qid]["torvalds"]
        o_kh = oracle[qid]["kroah_hartman"]
        oracle_t_wins = o_t >= o_kh   # raw GF comparison, no equal band

        agree = model_t_wins == oracle_t_wins
        if agree:
            g2_agree += 1
        g2_per_q[qid] = {
            "model_t": round(t_score, 4), "model_kh": round(kh_score, 4),
            "oracle_t_gf": round(o_t, 4), "oracle_kh_gf": round(o_kh, 4),
            "model_t_wins": model_t_wins, "oracle_t_wins": oracle_t_wins,
            "agree": agree,
        }

    g2_pass = g2_agree >= 12

    # GATE 3 — |r(score, length)| <= 0.20 and p > 0.05
    lengths = []
    scores_flat = []
    for ri, rec in enumerate(markup_in):
        for clone_key in ("clone_A", "clone_B"):
            span_chars = sum(len(s["text"]) for s in rec[clone_key]["spans"])
            score, _ = in_scores[(ri, clone_key)]
            lengths.append(span_chars)
            scores_flat.append(score)

    r_corr, p_corr = stats.pearsonr(lengths, scores_flat)
    g3_pass = abs(r_corr) <= 0.20 and p_corr > 0.05

    # GATE 4 — grounded-vs-OOD AUC >= 0.85, or every OOD below grounded min
    in_flat = [score for score, _ in in_scores.values()]
    ood_flat = list(ood_scores.values())
    labels_g4 = [1] * len(in_flat) + [0] * len(ood_flat)
    g4_auc = float(roc_auc_score(labels_g4, in_flat + ood_flat))
    in_min = min(in_flat)
    ood_max = max(ood_flat)
    g4_all_ood_below = ood_max < in_min
    g4_pass = g4_auc >= 0.85 or g4_all_ood_below

    # TIEBREAK — sentence-level AUC vs oracle labels (grounded=1, inferable/free=0)
    oracle_labels_sent = []
    model_scores_sent = []
    for ri, rec in enumerate(markup_in):
        qid = rec["query_id"]
        pass_i = rec["pass"]
        key_str = f"{qid}_p{pass_i}"
        oracle_entry = markup_out.get(key_str, {})
        for clone_key in ("clone_A", "clone_B"):
            _, span_dict = in_scores.get((ri, clone_key), (0.0, {}))
            for si_str, oval in oracle_entry.get(clone_key, {}).items():
                si = int(si_str)
                if si in span_dict:
                    oracle_labels_sent.append(1 if oval["label"] == "grounded" else 0)
                    model_scores_sent.append(span_dict[si])

    pos = sum(oracle_labels_sent)
    neg = len(oracle_labels_sent) - pos
    if pos > 0 and neg > 0:
        tiebreak_auc = float(roc_auc_score(oracle_labels_sent, model_scores_sent))
    else:
        tiebreak_auc = 0.5

    return {
        "model": model_name,
        "gate1": {
            "mean_abs_diff": round(g1_mean_diff, 4),
            "threshold": 0.05,
            "lean_counts": g1_lean,
            "systematic_lean": g1_systematic,
            "pass": g1_pass,
            "per_query": g1_per_q,
        },
        "gate2": {
            "agreements": g2_agree,
            "of": 14,
            "threshold": 12,
            "pass": g2_pass,
            "per_query": g2_per_q,
        },
        "gate3": {
            "pearson_r": round(float(r_corr), 4),
            "abs_r": round(abs(float(r_corr)), 4),
            "p_value": round(float(p_corr), 4),
            "threshold_abs_r": 0.20,
            "pass": g3_pass,
        },
        "gate4": {
            "auc": round(g4_auc, 4),
            "threshold_auc": 0.85,
            "all_ood_below_grounded_min": g4_all_ood_below,
            "in_domain_min": round(in_min, 4),
            "ood_max": round(ood_max, 4),
            "pass": g4_pass,
        },
        "tiebreak_sentence_auc": round(tiebreak_auc, 4),
        "tiebreak_sent_count": len(oracle_labels_sent),
        "all_gates_pass": g1_pass and g2_pass and g3_pass and g4_pass,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("W1b.0 Groundedness Bake-off — Day 14")
    print("API calls: 0 (all local inference, confirmed)")
    print("=" * 60)

    markup_in, markup_out, blind_key, ood_records = load_data()

    # Load all three scorers; record any failures
    scorer_classes = [DeBERTaScorer, HHEMScorer, MiniCheckScorer]
    scorers = []
    load_errors = {}
    for cls in scorer_classes:
        try:
            scorers.append(cls())
        except Exception as e:
            load_errors[cls.name] = str(e)
            print(f"ERROR loading {cls.name}: {e}")
            traceback.print_exc()

    if not scorers:
        print("No models loaded; aborting.")
        sys.exit(1)

    all_gates = []
    all_errors = dict(load_errors)

    for scorer in scorers:
        print(f"\n{'='*40}")
        print(f"Scoring: {scorer.name}")
        try:
            in_scores = score_indomain(markup_in, scorer)
            ood_scores = score_ood(ood_records, scorer)
            gates = compute_gates(markup_in, markup_out, blind_key,
                                  in_scores, ood_scores, scorer.name)
            all_gates.append(gates)
            status = "ALL PASS" if gates["all_gates_pass"] else "FAIL"
            print(f"  {scorer.name}: {status}")
        except Exception as e:
            all_errors[scorer.name] = str(e)
            print(f"ERROR scoring {scorer.name}: {e}")
            traceback.print_exc()

    # Determine winner
    passers = [g for g in all_gates if g["all_gates_pass"]]
    if len(passers) > 1:
        passers.sort(key=lambda g: g["tiebreak_sentence_auc"], reverse=True)
        winner = passers[0]["model"]
        winner_reason = "tiebreak"
    elif len(passers) == 1:
        winner = passers[0]["model"]
        winner_reason = "sole_passer"
    else:
        winner = None
        winner_reason = "none_clears"

    output = {
        "run": "W1b0_bakeoff_day14",
        "date": "2026-06-03",
        "api_calls": 0,
        "locked_criterion": {
            "gate1": "mean |score_T - score_KH| <= 0.05 on 7 held-equal queries, no systematic lean",
            "gate2": "per-query T-vs-KH direction agrees with oracle on >= 12/14",
            "gate3": "|r(score, length)| <= 0.20 and p > 0.05",
            "gate4": "grounded-vs-OOD AUC >= 0.85, or every OOD below grounded minimum",
            "tiebreak": "highest sentence-level AUC vs oracle labels (grounded=1, other=0)",
        },
        "held_equal_queries": sorted(HELD_EQUAL),
        "model_ids": {
            "deberta_v3_nli": "cross-encoder/nli-deberta-v3-base",
            "hhem_2_1_open": "vectara/hallucination_evaluation_model",
            "minicheck_flan_t5_large": "lytang/MiniCheck-Flan-T5-Large",
        },
        "gates": all_gates,
        "load_or_scoring_errors": all_errors,
        "winner": winner,
        "winner_reason": winner_reason,
    }

    out_path = ROOT / "results" / "bakeoff_w1b0_day14.json"
    out_path.write_text(json.dumps(output, indent=2, cls=NumpyEncoder))
    print(f"\nResults written to {out_path}")

    # Summary table
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Model':<28} {'G1':>6} {'G2':>6} {'G3':>6} {'G4':>6} {'TB AUC':>8}")
    print("-" * 60)
    for g in all_gates:
        g1 = "PASS" if g["gate1"]["pass"] else f"FAIL({g['gate1']['mean_abs_diff']:.3f})"
        g2 = "PASS" if g["gate2"]["pass"] else f"FAIL({g['gate2']['agreements']}/14)"
        g3 = "PASS" if g["gate3"]["pass"] else f"FAIL(r={g['gate3']['pearson_r']:.2f})"
        g4 = "PASS" if g["gate4"]["pass"] else f"FAIL({g['gate4']['auc']:.3f})"
        tb = f"{g['tiebreak_sentence_auc']:.4f}"
        print(f"{g['model']:<28} {g1:>6} {g2:>6} {g3:>6} {g4:>6} {tb:>8}")

    print()
    if winner:
        print(f"WINNER: {winner}  (reason: {winner_reason})")
    else:
        print("RESULT: NONE CLEARS — no model passed all four gates. STOP.")

    for name, err in load_errors.items():
        print(f"  LOAD ERROR {name}: {err}")


if __name__ == "__main__":
    main()
