# ADR-016: Evaluation Methodology, Three-Layer Approach

**Status:** Accepted

**Date:** 2026-05-26

## Context

P6 v2 has three distinct evaluation surfaces. One is per-Component math correctness, where ScoringEngine's cosine and similarity calculations have exact answers. Another is per-Agent behavior under controlled inputs, where CloneAgent and GatekeeperAgent have to behave correctly given a fixed query and fixed scores. The third is end-to-end system behavior, where the full Flow runs the v2 query set and the routing outcomes are what a user would see.

Without explicit layering these surfaces bleed into each other. Unit tests drift into asserting system behavior, and system tests fail for reasons a unit test should have caught, which leaves no layer that reliably owns a given class of failure. The question this ADR answers is what each layer measures and what it guarantees.

Day 8 is the reason the layering has to be explicit. The end-to-end verification pass found failures that the per-step tests had been passing over, the clearest being a reranker unit test that mocked the success path and stayed green for two months while the live Cohere call was broken on every run (`docs/day8-findings.md`, Finding 2). A system-level failure had been hiding behind passing per-step claims, which is exactly the gap a three-layer methodology is built to close.

## Decision

Three evaluation layers, each with a defined guarantee.

- Layer 1, unit, runs continuously. Every Agent and Component has unit tests under `tests/unit/` that mock the LLM calls and stay deterministic, with a coverage target of at least 90% on `src/` (PRD §2.9). These run in CI on every commit.
- Layer 2, integration, runs per Agent or Component. Each unit is tested in isolation against its real dependency: a real LLM call for an Agent, or a real backend like FAISS for a Component. LLM responses are recorded and replayed in CI so the layer stays deterministic, and the assertions are on contract behavior rather than exact output text. These live under `tests/integration/`, and the replay harness (the planned `tests/integration/conftest.py`) is created during the rework.
- Layer 3, system, runs end-to-end. The v2 query set runs through the full Flow via `cli evaluate`, producing JSON with the scores and routing decision for each query, along with its latency. The Day 11 evaluation report captures this run against the 2x2 routing grid and the comparison to the Day 8 baseline, plus the PRD §2 scorecard.

A methodology document explains the three layers and how regression detection works across them. Both the report (`docs/day11-evaluation.md`) and the methodology document (`docs/evaluation-methodology.md`) are planned Day 12 deliverables (PRD §7.5.1), not artifacts that exist yet.

## Alternatives Considered

- Two layers, unit and system, skipping integration. Rejected. CloneAgent and GatekeeperAgent need contract tests with real LLM calls that are neither unit tests nor full-system runs. Without the integration layer those tests either bloat the unit layer and slow CI, or get deferred into system runs where they lose isolation.
- Four layers, adding a smoke layer between integration and system. Rejected as overkill. The v2 query set is small, 20 queries per leader, so a full system run is not slow enough to need a smoke pass in front of it. A smoke layer would add maintenance without covering anything the other three layers miss.
- LLM-as-judge as its own evaluation layer. Rejected as a layer, kept as an element inside Layer 3. EvaluatorAgent's explanation already supplies qualitative LLM judgment over the deterministic scores (ADR-007), so making it a separate layer would multiply evaluation surfaces without adding rigor.

## Quantified Validation

- The motivating evidence is the Day 8 Layer-3 gap, and it is measured rather than hypothetical. End-to-end verification surfaced a silently broken Cohere reranker that a mocked unit test had reported as passing for two months (`docs/day8-findings.md`, Finding 2 and the side-effect-verification follow-up). A layer that exercises the real or recorded call is the only one that catches this class of failure.
- The same Day 8 run surfaced the score-distribution problem behind the routing. The in-domain deliver rate measured 39% with a leader-asymmetric style signal (Finding 1 and the scorecard), a system-level property that no per-Component or per-Agent test reports on. This is the second failure class that only Layer 3 sees.
- The three-layer structure is specified for the rework, not standing infrastructure. PRD §2.8 names the three layers as an integration target and PRD §2.9 sets the at-least-90% coverage target along with the per-layer test rows. The recorded-LLM replay harness for Layer 2 is added during the rework, so the coverage figure is a target the rework has to reach rather than a measured result.
- The Layer-3 output this methodology defines is what ADR-015's acceptance criteria judge. EvaluatorAgent's contract, the Layer-2 target named in ADR-011, is one of the per-Agent contracts Layer 2 covers.

## Consequences

Test files organize by layer, so the layer a failure lands in names its class. A red Layer-1 test points at a logic bug and a red Layer-2 test at a broken contract, while a red Layer-3 run is a system regression of the kind Day 8 exposed. CI runs Layer 1 on every commit and runs Layers 2 and 3 on pull request and on release, which keeps the fast deterministic checks continuous and the slower real-dependency checks gated. The methodology document becomes a portable artifact reusable in P7 and later projects. The Day 11 evaluation report stays scoped to Layer-3 results, so Layer-1 and Layer-2 outcomes do not appear in it. (In Java and TS terms this matches the Mike Cohn test pyramid, unit over integration over end-to-end, with the Layer-3 system evaluation doubling as the regression suite.)
