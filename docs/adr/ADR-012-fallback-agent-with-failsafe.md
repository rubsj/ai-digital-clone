# ADR-012: LLM-Driven FallbackAgent with Templated Failsafe

**Status:** Accepted
**Date:** 2026-05-26

## Context

v1's FallbackAgent was a Python function returning templated text with a hardcoded acknowledgment and three mock time slots. It produced the same output regardless of query topic, leader voice, or trigger reason.

The fallback path is dominant, not exceptional. Day 8's three-run comparison showed fallback rates of 95%, 90%, and 72.5% across the v1-broken, v1-fixed, and v2-final runs. Even on the best v2 run, 30 to 40 percent of queries take this path. A system whose dominant path is a mechanical template is barely multi-agent.

PRD §2.6 requires a contextual acknowledgment, 2-3 in-domain redirections when adjacent topics exist, a calendar mock with three slots, and a working templated failsafe. The v1 template satisfied none of these. Production reliability also requires a non-LLM path for the fallback so the system degrades gracefully if the LLM is down or rate-limited. A fallback-from-fallback that crashes is worse than a templated default.

## Decision

FallbackAgent is a CrewAI Agent that generates a leader-voiced acknowledgment, 2-3 in-domain redirections inferred from the retrieved chunks, a calendar booking link, and three mock time slots, using the `trigger_reason` from GatekeeperAgent for context. A 5-line try/except wraps the LLM call and activates a templated failsafe path on any LLM failure. The failsafe returns the v1-style template with the leader name substituted, so the system always returns a usable `FallbackResponse`.

## Alternatives Considered

- Keep v1's templated fallback (no Agent). The path is too dominant for a mechanical template (72.5% on the v2 final run). A multi-agent system whose primary user-facing experience is a fixed string is misnamed.
- LLM-driven fallback with no failsafe. Production reliability requires the LLM-failure case to degrade rather than crash. The failsafe is five lines, has no LLM dependency, and removes a class of cascading failure.
- Pre-compute fallback responses per query category. Query types are not bounded. Pre-computation either grows into a giant lookup table or collapses back to a small number of templates, which is the v1 design.

## Quantified Validation

- Fallback rate at threshold 0.75: 95% (v1 May-23, Cohere broken), 90% (v1 with Cohere fixed), 72.5% (v2 final run, 40 records). Source: "Three-run comparison" table.
- In-domain fallback rate on v2 final: 60.7% (17 of 28 in-domain records routed to fallback). Source: PRD §2 scorecard row §2d in day8-findings.md.
- Per-leader fallback rates on v2 final: Torvalds 14/20 (70.0%), Kroah-Hartman 15/20 (75.0%). Source: same table.
- Mean fallback latency on v2 final: 11,445 ms (fallback fires from the score path, so the full pipeline ran before fallback was decided). Source: same table. Confirms the fallback path's user-visible weight in the system.

The dominant-path framing rests on these numbers: fallback is 60 to 75 percent of in-domain user experience in the measured runs. An LLM-driven agent there is justified by frequency, not by edge-case polish.

## Consequences

- Positive: fallback reads coherently with the rest of the system. Leader voice carries through, so Torvalds and Kroah-Hartman fallbacks sound different.
- Positive: in-domain redirection turns a refusal into a partial answer, pointing the user at adjacent questions the system can handle.
- Positive: the failsafe means LLM downtime does not cascade into a system crash on the dominant path.
- Positive: fallback transparency works. The Agent can explain why fallback was triggered using the `trigger_reason` from GatekeeperAgent, which the customized requirements asked for.
- Negative: ~1-2s LLM latency on the fallback path, on top of the work the pipeline already did before routing decided fallback. Covered by the 8s end-to-end budget but tight.
- Negative: more prompt-engineering surface. Leader voice and trigger explanation both have to be tuned, and the templated failsafe has to be kept current with the schema.

(Similar to a circuit breaker with a degraded-mode fallback in resilience4j or Hystrix: the rich path is primary, and a cheap deterministic path activates on primary failure so the dominant path always has a response.)
