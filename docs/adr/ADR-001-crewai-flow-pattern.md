# ADR-001: CrewAI Flow with Real Agents at Each Step

**Project:** P6: Torvalds Digital Clone
**Category:** Architecture
**Status:** Accepted (rewritten for v2, 2026-05-26)
**Date:** 2026-04-03

---

## Context

CrewAI offers three orchestration approaches. Two are Crews, Sequential and Hierarchical; the third is Flows. v1's ADR-001 chose Flow and that choice still holds. What v1 got wrong was the layer below the choice: every Flow step was wired as a Python function calling deterministic pipeline code, and four of those functions were labeled "agents" in the documentation.

The v2 question is not which orchestration pattern to use. The answer is still Flow. The question is what runs at each Flow step. The v2 PRD defines what the system has to be: §2 sets the acceptance criteria, including the routing-correctness headline and the multi-agent orchestration goal, and §3 sets the architectural intent of four LLM-driven Agents over three deterministic Components. Measuring the v1 implementation against those requirements showed it did not meet them. The multi-agent claim was directory-deep rather than execution-deep: only one of five "agents" used the CrewAI Agent abstraction, while the other four were Python functions with the vocabulary applied as a label rather than as structure.

The architecture had to change because the requirements were not being met, not because an audit produced a finding. Day 8 verification is the date the gap was measured, not the reason for the rework. The evidence is the v1 ADR's own Decision section, which states that only ChatStyleAgent used the Agent abstraction and that "the other four agents are just functions." The v1 code says the same thing. `src/agents/style_crew.py` is the only file that imports `from crewai import Agent, Crew, LLM, Task` and constructs an `Agent(role, goal, backstory)`. Alongside it, `rag_agent.py` and `evaluator_steps.py` are plain Python classes and `fallback_steps.py` is a plain function, none of them touching CrewAI.

---

## Decision

CrewAI Flow stays as the deterministic orchestrator backbone. Each Flow step calls either a real CrewAI Agent (CloneAgent, EvaluatorAgent, GatekeeperAgent, FallbackAgent) or a real Component (Retriever). The Flow is the orchestration and there is no separate PlannerAgent. State is managed through `Flow[CloneState]`, a Pydantic `BaseModel` that CrewAI populates incrementally as each step completes. Conditional branching happens through the `@router` decorator on the GatekeeperAgent step, which returns the string `"deliver"` or `"fallback"`.

This is the central architectural correction in v2. The Flow shell is unchanged from v1: the same Flow decorators (`@start`, `@listen`, `@router`) define the same step order. What changes is what runs inside each step, moving from Python functions to real Agents and Components. v1 named the Flow itself "the PlannerAgent" and ran its evaluator and fallback steps as direct function calls. v2 keeps the Flow as plain orchestration. It promotes the steps where LLM reasoning is the work into real Agents and reclassifies the retrieval step as the deterministic Retriever Component.

---

## Alternatives Considered

- **Sequential Crew.** Rejected. Tasks run in a fixed order with no native conditional branching, so skipping FallbackAgent when GatekeeperAgent decides to deliver would need a no-op task or post-hoc filtering. The pipeline requires real branching and Sequential cannot express it.
- **Hierarchical Crew.** Rejected. A Manager Agent making an LLM-based delegation decision on every step adds one to two seconds of latency per step with no autonomy benefit, because the pipeline order is already fixed. Hierarchical Crews are also documented as fragile in production, with the Towards Data Science analysis from November 2025 recording deployments where they looped instead of delegating.
- **Flow with Python-function steps (the v1 pattern).** Rejected for v2. It forfeits the Agent abstraction benefits that matter for an LLM step, like role/goal/backstory prompt scaffolding and Instructor-validated structured output with automatic retry. It is also the exact shape that made v1's multi-agent claim directory-deep rather than execution-deep.

---

## Quantified Validation

- The bar the v1 system was measured against is the v2 requirements set, not a separate audit. PRD §2 fixes the acceptance criteria, including routing-correctness as the headline metric and multi-agent orchestration as a goal, and PRD §3 fixes the four-Agent-plus-three-Component architecture. The Day 8 verification run is the empirical anchor that exposed the gap against that bar.
- Of five units called "agent," exactly one was real: ChatStyleAgent in `src/agents/style_crew.py`, which wraps an `Agent` plus a `Task` in a single-agent `Crew`. The other four were Python functions or classes wrapped in Flow decorators. PRD §12.2 records the same mapping, and the v1 ADR-001 Decision text being replaced says it outright.
- CrewAI's own engineering guidance backs the v2 shape. The December 2025 CrewAI blog recommends a deterministic Flow backbone whose individual steps lean on different levels of agent autonomy, and the DocuSign case study migrated from Sequential Crews to Flows to get conditional logic with typed state. Both points carried the v1 decision and carry forward, reframed for the real-Agents-at-each-step pattern.
- The concrete v2 shape is four real Agents at the four steps where LLM reasoning is the work, plus one Component step for retrieval. PRD §5.5 gives the Flow code skeleton: the retrieve step runs the Retriever Component, each LLM-reasoning step runs its Agent, and the `@router` step calls GatekeeperAgent to return the deliver-or-fallback string.
- The v1 decorator POC (`scratch/flow_poc.py`) validated the decorator wiring in isolation and nothing more. Its routing test is not the v2 routing path; routing is now GatekeeperAgent's string decision rather than a numeric threshold (ADR-010).

---

## Consequences

This is the decision the rest of the v2 ADRs build on. The Flow shell stays, and what runs inside it moves from Python functions carrying multi-agent vocabulary to real Agents where LLM reasoning lives and real Components where determinism lives. The visible artifact of that move is the `src/agents/` versus `src/components/` split that ADR-009 defines and ADR-014 inventories.

The CrewAI dependency surface stays small and isolated to `src/flow.py` and the Agent classes, so a future Flows API change is contained to those files rather than spreading across the pipeline. The `crewai>=0.108.0` pin in `pyproject.toml` carries forward from v1, where it was set because the CrewAI API changed between minor versions.

The dual-leader comparison still runs two Flow instances over one query, retrieving once and styling twice. The detail now lives in ADR-005 rather than here, including the `CloneState.chunks` early-exit that lets the second leader's retrieve step skip the embed-and-rerank path. (In Java terms this is closer to Spring Integration flow definitions calling typed beans than to a saga orchestrator, where the Flow is structure and the Agents and Components are behavior.)
