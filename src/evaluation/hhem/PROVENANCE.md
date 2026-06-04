# HHEM-2.1-Open — Vendored source

Source repository: vectara/hallucination_evaluation_model (HuggingFace Hub)
Pinned commit: 8e4a2e6e96c708cc76c2344f7e4757df2515292c
Files vendored: modeling_hhem_v2.py, configuration_hhem_v2.py
Vendor date: 2026-06-04

## Changes applied

### Change 1 — `all_tied_weights_keys` class attribute

One line added to `HHEMv2ForSequenceClassification` in `modeling_hhem_v2.py`:

```python
all_tied_weights_keys: dict = {}
```

Why: transformers 5.x accesses `all_tied_weights_keys` on the model instance during
weight loading (modeling_utils.py lines 4526, 4615, 4634, 4740). The remote class
never sets this attribute, raising `AttributeError`. HHEM ties no weights, so the
correct value is an empty dict. This is a load-time fix only; no inference logic is
changed. See ADR-020 Consequences for the full dependency analysis.

### Change 2 — `tie_weights()` override

Method added to `HHEMv2ForSequenceClassification` in `modeling_hhem_v2.py`:

```python
def tie_weights(self) -> None:
    super().tie_weights()
    self.t5.tie_weights()
```

Why: transformers 5.x weight loading replaces tensor objects rather than copying into
them. This breaks the `encoder.embed_tokens → shared` weight tying that
`T5ForTokenClassification.__init__` establishes. `from_pretrained` calls
`tie_weights()` after loading; this override restores the T5-internal tie at that
point so `predict()` produces discriminating scores. Without this, the model returns
flat ~0.5 scores regardless of input.

This invariant must live in the model, not in the calling code — any caller that
constructs `HHEMv2ForSequenceClassification` directly (not via
`HHEMGroundednessScorer`) would otherwise get a silently broken model. Smoke-check
confirmed: grounded ≈ 0.97, ungrounded ≈ 0.007, gap ≈ 0.963 (both before and after
this relocation). See ADR-020 Consequences.

## trust_remote_code

`trust_remote_code=True` is intentionally NOT used on the load path. The project owns
the vendored source directly; remote-code execution from the hub is unnecessary and
removed. See ADR-020.

## Upstream status

As of 2026-06-04, `refs/main` on the Vectara hub points to commit
`8e4a2e6e96c708cc76c2344f7e4757df2515292c` — the only published snapshot. No
transformers-5.x-compatible upstream revision exists.
