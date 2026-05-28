# ADR-007: LLM Roles in the Pipeline

**Project:** P6: Torvalds Digital Clone
**Category:** Architecture
**Status:** Accepted (rewritten for v2, 2026-05-26)
**Date:** 2026-04-27

---

## Context

v1 ADR-007 asked a narrow question: could GPT-4o-mini's LLM-as-judge groundedness scores be trusted in production, measured against the deterministic cosine baseline. The answer was yes, the LLM agreed with the baseline closely enough, and v1 used the LLM for both numerical scoring and explanation generation.

That question is obsolete in v2. v2 removes LLMs from numerical scoring entirely: the ScoringEngine Component computes the three scores (style, groundedness, confidence) deterministically (ADR-003, ADR-004), and LLM use moves to the places where reasoning is the point. So the v2 question is broader than the viability of one scoring path. Across the whole pipeline, where do LLMs earn their cost, and where are they the wrong tool? This ADR consolidates that boundary, building on the architectural correction in ADR-001 where each Flow step became a real Agent or a real Component.

---

## Decision

LLMs run at four pipeline locations, each because reasoning over inputs is the actual work:

- CloneAgent generates the response in the leader's voice, a generative task no deterministic template does well.
- EvaluatorAgent turns the deterministic scores into a human-readable explanation and a list of flags, qualitative work over numbers the ScoringEngine already computed.
- GatekeeperAgent makes the deliver-or-fallback routing decision by reasoning over the scores in the context of the retrieved chunks and the response, rather than reducing them to one number.
- FallbackAgent writes a leader-appropriate fallback message when the Gatekeeper declines to deliver.

LLMs are deliberately kept out of four other places. Numerical scoring is deterministic math in the ScoringEngine. Retrieval is FAISS search plus Cohere reranking, with no reasoning LLM in the chain. Orchestration is the Flow decorators (`@start`, `@listen`, `@router`), which are plain control flow. And the Flow's own routing is not LLM-driven; only the GatekeeperAgent decision that the `@router` step calls is. The principle is simple: put an LLM where reasoning is the work, and keep it out where reasoning is the obstacle.

---

## Alternatives Considered

- **LLM everywhere, including numerical scoring (the v1 pattern).** Rejected for v2. Numerical scoring is math, and an LLM on that path adds non-determinism and latency with no quality gain over the cosine baseline. The v1 result is the evidence: the LLM-judge correlated with the cosine baseline at Pearson 0.82, which argues for keeping the cosine scorer and dropping the LLM from that path, not the reverse.
- **LLM nowhere (fully deterministic).** Rejected. Leader-voice response generation cannot be done well by a template, and the routing judgment that absorbs the style-score asymmetry (ADR-013) needs qualitative reasoning. Stripping every LLM out collapses the system to a templated chatbot.
- **LLM only for response generation (CloneAgent), deterministic everywhere else.** Rejected. GatekeeperAgent's reasoning is the mechanism that absorbs the measurement asymmetry, and FallbackAgent's leader-voice message is the user-facing quality on the path most queries actually take. EvaluatorAgent's explanation is the third case, the part that makes the system inspectable. Each of those earns its LLM call.

---

## Quantified Validation

- The one hard number in this ADR is the retired one. The v1 LLM-as-judge groundedness scores correlated with the deterministic cosine baseline at Pearson 0.82 (0.8172, p=0.0039) on the v1 query set. v2 cites that number as the reason the LLM-scoring path was removed rather than kept: the LLM agreed with cosine, so cosine wins on determinism and cost. PRD §10.5 marks the experiment obsolete and points here.
- The four LLM placements are each justified in their own Batch A ADR, and this one consolidates the cross-cutting principle rather than re-deriving them. ADR-009 sets the Agent-versus-Component criterion that decides what counts as an LLM placement at all. The specific placements are ADR-010 for GatekeeperAgent routing, ADR-011 for the EvaluatorAgent hybrid, ADR-012 for the FallbackAgent failsafe.
- ScoringEngine's determinism is the other half of the decision, grounded in ADR-004 for the groundedness cosine and ADR-003 for the hand-crafted style features. No LLM touches those numbers.
- The boundary shows up in the import graph. PRD §5.1 specifies the four Agents' LLM use and §5.2 specifies the three Components as deterministic `run()` classes with no LLM calls. The v2 Component code lands in the Day 10-11 rework, so this is the specified boundary rather than code already on disk. In v1 the reasoning-LLM calls live in `src/evaluation/evaluator.py` and `src/fallback/unstyled_responder.py`, which v2 re-homes into EvaluatorAgent and FallbackAgent.

---

## Consequences

Every Agent in the v2 inventory now has an explicit LLM-use justification that traces back to this ADR, and any Agent added in future work has to pass the same test: reasoning over inputs has to be the actual work, not decoration. The flip side is enforced architecturally. Components stay LLM-free, and the CLAUDE.md architecture honesty check greps `src/components/*.py` for LLM-tool imports (`litellm`, `openai`, `cohere`, `instructor`) and flags a warning at the session stop gate if any appears.

The Ollama-versus-GPT comparison from the v1 ADR is dropped from this record. The decision is a single model, GPT-4o-mini, for all four Agents; local-model substitution is deferred post-portfolio (PRD §3.5 D4).

(In Java terms this is the choice between a deterministic strategy pattern and a rules engine that reasons with side effects: put the reasoning engine where adaptation is needed and keep the strategy where behavior must stay testable, because the wrong placement yields either a rigid system that cannot adapt or an unpredictable one that cannot be tested.)
