# ADR-009: Agent vs Component Distinction

**Status:** Accepted
**Date:** 2026-05-26

## Context

v1's architecture used "agent" informally across Flow steps that mixed LLM-driven and deterministic work. Of the five v1 units called "agent," one was a CrewAI Agent and four were Python functions wrapped in Flow `@listen` decorators. The label was doing no technical work.

v2 fixes this by tying "Agent" to a specific criterion: LLM-driven reasoning using the CrewAI Agent abstraction. "Component" means deterministic Python with no LLM calls. PRD §3.3 and §3.4 articulate the underlying principle: LLMs where reasoning adds value, deterministic code where computation is the work.

## Decision

An Agent is LLM-driven reasoning work using the CrewAI Agent abstraction with `role`, `goal`, and `backstory`, and lives in `src/agents/`. A Component is deterministic Python with a `run()` method and zero LLM calls, and lives in `src/components/`. The criterion is binary, enforced in file structure and CI grep checks, and applies uniformly in code, comments, docstrings, commit messages, and ADRs.

## Alternatives Considered

- Keep v1's informal vocabulary. Without a forced criterion, "agent" drifts to mean "thing in the pipeline" and the term loses its technical content. Future maintainers cannot rely on the vocabulary to describe the implementation.
- Call everything an Agent (broaden the term). This dilutes "multi-agent" until it means nothing more than "more than one file," and removes the architectural clarity the file structure and CI checks rely on.
- Call everything a Component (narrow the term, drop the CrewAI Agent abstraction). The CrewAI Agent abstraction adds real value where LLM reasoning is the work: `role/goal/backstory` shapes the prompt, Instructor handles structured output, the Crew runner adds retry.

## Quantified Validation

- The v1-to-v2 mapping recorded in PRD §1 "What's Different in v2" and PRD §12.2 "Mapping of v1 Code to v2 Architecture": of the five v1 units called "agent," one (ChatStyleAgent, now CloneAgent) was a CrewAI Agent and four were Python functions wrapped in Flow `@listen` decorators. v2 formalizes which of those become Agents under the new criterion and which become Components.
- The criterion's effectiveness in v2 is enforced by CI grep during the rework, per CLAUDE.md Verification Protocol Component 3 architecture honesty check.
- The v1-to-v2 mapping is the load-bearing evidence for this decision.

## Consequences

- Positive: file structure (`src/agents/` vs `src/components/`) makes the architecture self-documenting. New code lands in the right place by default.
- Positive: code review becomes a binary check rather than a judgement call. CI enforces it.
- Positive: the criterion answers "what makes this multi-agent" via file structure and CI checks, not branding.
- Negative: vocabulary discipline overhead. Every PR comment, commit message, doc, and session note has to use the right word.
- Negative: hybrid cases exist. EvaluatorAgent is an Agent that delegates numerical work to ScoringEngine Component; the distinction has to be expressed inside one Agent's responsibility surface (ADR-011 covers the hybrid pattern).

(Similar to Spring's `@Service` vs `@Repository` stereotype distinction: both are Spring beans, but the stereotype declares intent and constrains where each can be used, so the type name carries technical content rather than being a loose label.)
