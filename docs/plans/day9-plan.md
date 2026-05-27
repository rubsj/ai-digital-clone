# Day 9 ADR Batch A Plan

Plan for four v2 ADRs to be drafted by Sonnet in a follow-up session. This file is planning input only; no ADR files are written this session.

Sources read for this plan: `docs/PRD.md` (§3.3, §3.4, §3.5, §4.1-§4.4, §5.1, §7.5), `docs/day8-findings.md`, `docs/notes/adr-batch-a-context.md`, `CLAUDE.md` Writing Rules.

Format contract per CLAUDE.md: 5 H2 sections in order — Context, Decision, Alternatives Considered, Quantified Validation, Consequences. Java/TS parallel is an inline parenthetical at the end of Consequences, not a section. No bold emotional labels. No em-dashes, no tricolons, no "not just X but Y."

---

## ADR-009: Agent vs Component Distinction

### Context (outline)

- v1's architecture used "agent" informally across Flow steps that mixed LLM-driven and deterministic work. The term was not tied to a specific technical criterion.
- v2 formalizes the distinction: "Agent" carries a specific, enforceable meaning tied to LLM-driven reasoning using the CrewAI Agent abstraction; "Component" carries a specific meaning tied to deterministic Python.
- The formalization supports v2's broader goals: file-structure self-documentation (`src/agents/` vs `src/components/`), code review as a binary check rather than judgement, and CI enforcement of the architectural rule.
- The criterion needs to hold over time — in code, comments, docstrings, commit messages, and ADRs — so future maintainers (including Ruby reading this in six months) can rely on the vocabulary meaning what it says.
- PRD §3.3 and §3.4 articulate the underlying engineering principle: LLMs where reasoning adds value, deterministic code where computation is the work.

### Decision (verbatim draft)

An Agent is LLM-driven reasoning work using the CrewAI Agent abstraction with `role`, `goal`, and `backstory`, and lives in `src/agents/`. A Component is deterministic Python with a `run()` method and zero LLM calls, and lives in `src/components/`. The criterion is binary, enforced in file structure and CI grep checks, and applies uniformly in code, comments, docstrings, commit messages, and ADRs.

### Alternatives Considered

- **Keep v1's informal vocabulary.** Rejected. Without a forced criterion, "agent" drifts to mean "thing in the pipeline" and the term loses its technical content. Future maintainers cannot rely on the vocabulary to describe the implementation.
- **Call everything an Agent (broaden the term).** Rejected. Dilutes "multi-agent" until it conveys nothing more than "more than one file," and removes the architectural clarity the file structure and CI checks rely on.
- **Call everything a Component (narrow the term, drop the CrewAI Agent abstraction).** Rejected. The CrewAI Agent abstraction adds real value where LLM reasoning is the work: `role/goal/backstory` shapes the prompt, Instructor handles structured output, the Crew runner adds retry. Throwing it away costs more than it saves.

### Quantified Validation (evidence list)

- The v1-to-v2 mapping recorded in PRD §1 "What's Different in v2" and PRD §12.2 "Mapping of v1 Code to v2 Architecture": of the five v1 units called "agent," one (ChatStyleAgent, now CloneAgent) was a CrewAI Agent and four were Python functions wrapped in Flow `@listen` decorators. v2 formalizes which of those become Agents under the new criterion and which become Components.
- The criterion's effectiveness in v2 is validated by CI grep enforcement during the rework (per CLAUDE.md Verification Protocol Component 3 architecture honesty check), not by retrospective measurement.
- This ADR is primarily architectural reasoning; the v1-to-v2 mapping is the load-bearing evidence.

### Consequences (outline + Java/TS parenthetical)

- Positive: file structure (`src/agents/` vs `src/components/`) makes the architecture self-documenting. New code lands in the right place by default.
- Positive: code review becomes a binary check rather than a judgement call. CI enforces it.
- Positive: the criterion gives a defensible answer to "what makes this multi-agent" when Ruby reads this in six months or another reviewer picks up the code, grounded in file structure and CI checks rather than branding.
- Negative: vocabulary discipline overhead. Every PR comment, commit message, doc, and session note has to use the right word.
- Negative: hybrid cases exist. EvaluatorAgent is an Agent that delegates numerical work to ScoringEngine Component; the distinction has to be expressed inside one Agent's responsibility surface (ADR-011 covers the hybrid pattern).

Java/TS parenthetical (end of Consequences):
*(Similar to Spring's `@Service` vs `@Repository` stereotype distinction: both are Spring beans, but the stereotype declares intent and constrains where each can be used, so the type name carries technical content rather than being a loose label.)*

---

## ADR-010: LLM-Driven Routing via GatekeeperAgent

### Context (outline)

- v1 routed deliver vs fallback with a fixed weighted formula: `final = 0.4×style + 0.4×ground + 0.2×conf`, then a hard threshold at `final ≥ 0.75`.
- Day 8 verification on the v2 query set found the formula does not discriminate on this corpus and query set. Three independent failure modes: (a) compresses three signals into one number and loses interaction (high style + low groundedness reads the same as mediocre across the board, even though one is dangerous and the other unimpressive), (b) the score distribution is corpus-dependent so no threshold ports, (c) the 0.4/0.4/0.2 weights came from intuition rather than measurement.
- The fallback path is dominant under any threshold reachable in v2 (30-70% range depending on choice), so routing quality directly shapes user experience, not just an edge case.
- A correct routing decision needs to reason about flag interactions that no formula reduction can express. EvaluatorAgent's `flags` list surfaces things like "response mentions slab allocator but chunks discuss buddy allocator" which must force fallback regardless of scores.

### Decision (verbatim draft)

Replace the weighted formula and 0.75 threshold with a GatekeeperAgent that reasons over the query, the styled response, the retrieved chunks, EvaluatorAgent's three individual scores, and the explanation and flags. The Agent outputs a `RoutingDecision` containing `decision: Literal["deliver", "fallback"]`, a `reasoning` string that must reference specific scores and flags, and an optional `trigger_reason` passed to FallbackAgent. The Agent runs at `temperature=0` with Instructor-structured output; no `final_score` field exists anywhere in v2.

### Alternatives Considered

- **Recalibrate the formula weights via sensitivity sweep.** Rejected. Day 6's scoring-weight sensitivity experiment was a null result (PRD §10.3 marks it obsolete); the score band is too narrow on this corpus for reweighting to move the deliver/fallback split meaningfully. The problem is signal compression, not weight choice.
- **Replace the single threshold with multi-criteria thresholds (e.g., `style > 0.7 AND ground > 0.5`).** Rejected. Pushes the same problem up one level: thresholds still arbitrary, still corpus-dependent, and still cannot react to flag interactions like a confident-hallucination pattern where high style hides low groundedness.
- **Remove routing entirely; always deliver with score annotations.** Rejected. PRD §1 product principle says "better to say 'I don't know' than to fabricate while pretending to be the leader." Removing the deliver/fallback decision violates this and produces confident hallucinations on out-of-domain queries — the exact failure mode the Category 5 OOD probes are designed to surface.

### Quantified Validation (evidence list)

From `docs/day8-findings.md` — verified numbers:
- At threshold 0.75 the v1 May-23 eval (Cohere broken) produced 19/20 (95.0%) fallback. Source: day8-findings.md "Three-run comparison" table.
- At threshold 0.75 the v1 + Cohere working control run produced 18/20 (90.0%) fallback. Source: same table.
- The v2 final run (40 records, Cohere working) produced 29/40 (72.5%) fallback at threshold 0.75. Source: same table.
- Scored-record style means in the v2 final run: Torvalds 0.9025, Kroah-Hartman 0.8355, asymmetry +0.067 favoring Torvalds. Source: "Finding 1" table.
- Scored-record groundedness mean (v2 final): 0.6258 (up from 0.5173 broken-Cohere baseline). Source: "Three-run comparison."
- OOD fallback rate: 12/12 (100%) on Category 5 queries with zero hallucinations. Source: routing-correctness 2×2.
- **q12 binary-search regression (load-bearing).** Same query, same index, same threshold (0.75): v1 May-23 delivered at `final=0.7525`; v1 + Cohere working and v2 q12 both fell back. day8-findings.md "Verification 2" traces the flip to downstream LLM and groundedness-scoring stochasticity combined with single-good-chunk corpus shape, not to any infrastructure change. This is the strongest direct evidence that no fixed threshold is stable on this corpus: routing across runs flips with no change to scoring inputs the formula can see.

### Consequences (outline + Java/TS parenthetical)

- Positive: routing decisions become defensible. The Agent's `reasoning` string is the artifact you point at in interviews and review.
- Positive: ports to a new corpus or query set without re-tuning a threshold; the LLM reasons over whatever the scores happen to be on that corpus.
- Positive: handles confident-hallucination via flag awareness. High style with low groundedness can be routed to fallback because the flag is explicit, not because the formula caught it (it wouldn't).
- Positive: aligns with PRD §2.1 routing-correctness as the headline metric and the E1/E2 acceptance criteria in ADR-015.
- Negative: ~1-2s additional LLM latency per routing decision (covered by the 8s end-to-end budget per PRD §2.7).
- Negative: LLM reasoning has variance. Mitigated by `temperature=0`, Instructor structured output, and a prompt that demands explicit reference to specific scores and flags.
- Negative: harder to unit-test than a formula. Mitigated by Layer-2 integration tests with recorded LLM responses per ADR-016.

Java/TS parenthetical (end of Consequences):
*(Similar to replacing a Drools-style rule engine with a domain expert service that reasons over the same facts: the rule engine is faster and deterministic but only useful when the rules cleanly partition the input space, and when they cannot, you need judgement rather than more rules.)*

---

## ADR-011: EvaluatorAgent Hybrid Design

### Context (outline)

- EvaluatorAgent has two jobs in v2 that pull in opposite directions: produce three numerical scores (style, groundedness, confidence) and produce a human-readable explanation plus a list of flags for GatekeeperAgent to reason over.
- Job 1 is computation, not interpretation. Cosine similarity on style vectors and sentence-level groundedness math are deterministic operations with an exact answer; the LLM has no judgement to add over the math itself, and routing decisions downstream depend on trustworthy, reproducible scores.
- Job 2 is good LLM territory. Identifying that a response mentions slab allocator while chunks discuss buddy allocator is interpretation over text, which deterministic Python cannot do without becoming an LLM.
- The vocabulary lock from ADR-009 has to accommodate a hybrid: an Agent that internally delegates the deterministic part to a Component (ScoringEngine) rather than two siblings at the Flow level.
- The v1 EvaluationResult had a `final_score` field consumed by the routing formula. In v2 there is no `final_score`; GatekeeperAgent reasons over the three individual scores plus explanation and flags (per ADR-010).

### Decision (verbatim draft)

EvaluatorAgent is a real CrewAI Agent that delegates numerical scoring to the ScoringEngine Component and uses the LLM only to generate the explanation and flags. The Agent's output is an `EvaluationResult` with three scores from ScoringEngine, an LLM-generated `explanation` string that references the scores, and an LLM-identified `flags: list[str]`. `EvaluationResult` has no `final_score` field, and the weighted formula is removed from the codebase.

### Alternatives Considered

- **Pure Component (no LLM, just scoring).** Rejected. This is the v1 design. It loses the explanation and flag capability that GatekeeperAgent needs to do better-than-formula routing. Without flags, GatekeeperAgent is back to reasoning over three numbers, which barely improves on the formula and re-creates the Day 8 problem.
- **Pure Agent (LLM does scoring too).** Rejected. The deterministic cosine math is already strongly correlated with LLM judgment on groundedness (Pearson 0.82, PRD §10.5), so an LLM scorer adds no measurable lift while widening the system's failure surface and making scores non-reproducible. The deterministic backbone is what the rest of the system reasons over.
- **Two separate top-level units (ScoringAgent + ExplanationAgent).** Rejected. ScoringAgent would fail the ADR-009 criterion (no LLM reasoning) and collapse to a Component anyway. The split would be one Agent plus one Component named badly, which is the hybrid pattern with worse naming. The hybrid framing is cleaner because the Component is internal to the Agent's responsibility surface.

### Quantified Validation (evidence list)

- Pearson 0.82 correlation between deterministic cosine groundedness scoring and GPT-4o-mini judgment (PRD §10.5). The deterministic scorer is already strongly correlated with LLM judgment on this task, so the LLM has nothing to add to the scoring job that the Component cannot already do. Adding LLM scoring on top would expand the system's failure surface without measurable lift.
- Day 8 surfaced cases where high style scores masked low groundedness (Finding 1: Torvalds style mean 0.9025 vs groundedness mean 0.6258, 0.28 gap). Flag-aware interpretation is what makes ADR-010's routing improvement viable; pure scoring without flags re-creates the Day 8 problem.
- Beyond these two pointers, the decision rests on architectural reasoning rather than measurement: numerical computation belongs to deterministic code; interpretation belongs to LLMs.

### Consequences (outline + Java/TS parenthetical)

- Positive: each part of the work uses the right tool. Scoring stays deterministic, debuggable, and testable with synthetic inputs. Explanation and flags gain LLM interpretation.
- Positive: `EvaluationResult` becomes a richer artifact for GatekeeperAgent than three numbers, which is what makes ADR-010's routing improvement viable.
- Positive: enforces ADR-009's vocabulary inside a single responsibility surface without diluting either side.
- Negative: introduces an Agent-calls-Component pattern that has to be made explicit in code structure (ScoringEngine is injected into EvaluatorAgent, not co-located).
- Negative: integration testing requires recording the LLM call for the explanation and flag portion. Handled by Layer-2 contract tests per ADR-016.

No Java/TS parenthetical for ADR-011. The Spring `@Service`-injecting-`@Repository` analogy describes layering by concern, while the hybrid Agent is about delegating computation inside one reasoning unit; the mapping is loose enough that the parenthetical dilutes rather than sharpens. Per the Writing Rules going-forward rule: include a Java/TS parallel only when a Java/TS developer would naturally say "oh, like X" on reading the Consequences; skip it when the parallel needs explanation to land.

---

## ADR-012: LLM-Driven FallbackAgent with Templated Failsafe

### Context (outline)

- v1's FallbackAgent was a Python function returning templated text with hardcoded acknowledgment and three mock time slots. It produced the same output regardless of query topic, leader voice, or trigger reason.
- The fallback path is dominant, not exceptional. Day 8's three-run comparison showed fallback rates of 95%, 90%, and 72.5% across the v1-broken / v1-fixed / v2-final runs. Even at the optimistic end of v2's reachable operating range, 30-40% of queries take this path. A system whose dominant path is mechanical template is barely multi-agent in user-perceived experience.
- PRD §2.6 requires contextual acknowledgment, 2-3 in-domain redirections when adjacent topics exist, calendar mock with three slots, and a working templated failsafe — none of which the v1 template satisfied meaningfully.
- Production reliability requires a non-LLM path for the fallback so the system degrades gracefully if the LLM is down or rate-limited. A fallback-from-fallback that crashes is worse than a templated default.

### Decision (verbatim draft)

FallbackAgent is a real CrewAI Agent that generates a leader-voiced acknowledgment, 2-3 in-domain redirections inferred from the retrieved chunks, a calendar booking link, and three mock time slots, using the `trigger_reason` from GatekeeperAgent for context. A 5-line try/except wraps the LLM call and activates a templated failsafe path on any LLM failure, returning the v1-style template with the leader name substituted so the system always returns a usable `FallbackResponse`.

### Alternatives Considered

- **Keep v1's templated fallback (no Agent).** Rejected. The path is too important to leave mechanical. Day 8 fallback rates (72.5% on the v2 final run) make this the dominant path. A multi-agent system whose dominant user-facing experience is mechanical template is misnamed.
- **LLM-driven fallback with no failsafe.** Rejected. Production reliability requires the LLM-failure case to degrade rather than crash. The failsafe is five lines, has no LLM dependency, and removes a class of cascading failure.
- **Pre-compute fallback responses per query category.** Rejected. Query types are not bounded. Pre-computation either grows into a giant lookup table or collapses back to a small number of templates, which is what v1 had.

### Quantified Validation (evidence list)

From `docs/day8-findings.md`:
- Fallback rate at threshold 0.75: 95% (v1 May-23, Cohere broken), 90% (v1 with Cohere fixed), 72.5% (v2 final run, 40 records). Source: "Three-run comparison" table.
- In-domain fallback rate on v2 final: 60.7% (17 of 28 in-domain records routed to fallback). Source: PRD §2 scorecard row §2d in day8-findings.md.
- Per-leader fallback rates on v2 final: Torvalds 14/20 (70.0%), Kroah-Hartman 15/20 (75.0%). Source: same table.
- Mean fallback latency on v2 final: 11,445 ms (fallback fires from the score path, full pipeline ran before fallback was decided). Source: same table. Confirms the fallback path's user-visible weight in the system.

These numbers establish the dominant-path framing: fallback is between 60% and 75% of in-domain user experience in the measured runs. An LLM-driven agent there is justified by frequency, not by edge-case polish.

### Consequences (outline + Java/TS parenthetical)

- Positive: fallback reads coherently with the rest of the system. Leader voice carries through; Torvalds and Kroah-Hartman fallbacks sound different.
- Positive: in-domain redirection turns a refusal into a partial success by pointing the user at adjacent questions the system can answer.
- Positive: the failsafe means LLM downtime does not cascade into a system crash on the dominant path.
- Positive: satisfies the customized requirements' fallback transparency expectation (the Agent can explain why fallback was triggered using the `trigger_reason` from GatekeeperAgent).
- Negative: ~1-2s LLM latency on the fallback path, on top of the work the pipeline already did before routing decided fallback. Covered by the 8s end-to-end budget but tight.
- Negative: more prompt-engineering surface — leader voice and trigger explanation both have to be tuned, and the templated failsafe has to be kept current with the schema.

Java/TS parenthetical (end of Consequences):
*(Similar to a circuit breaker with a degraded-mode fallback in resilience4j or Hystrix: the rich path is the primary, and a cheap deterministic path activates on primary failure so the dominant user path never collapses to nothing.)*
