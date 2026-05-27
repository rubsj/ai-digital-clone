# ADR Batch A Context Brief

> Source notes assembled from prior planning sessions for use as input to the Opus ADR planning prompt. Captures reasoning, rejected alternatives, and Day 8 evidence that informed ADRs 009-012. Read alongside `docs/PRD.md` (especially §3.3, §3.4, §4.1-§4.4, §7.5, §12.3) and `docs/day8-findings.md`. Where this brief and the PRD disagree, the PRD wins.

---

## ADR-009: Agent vs Component Distinction

### The problem this ADR solves

P6 v1 claimed a 5-agent multi-agent architecture. Day 8 verification revealed that 4 of those 5 "agents" were Python functions wrapped in CrewAI Flow `@listen` decorators — no LLM reasoning, no CrewAI Agent abstraction, no role/goal/backstory. Only ChatStyleAgent (renamed CloneAgent in v2) was a real Agent. The system claimed multi-agent and delivered single-agent.

The root cause was vocabulary drift. "Flow step" and "agent" were used interchangeably in documentation and code. Without a vocabulary that forced the distinction, the architecture stopped being multi-agent during implementation while the documentation kept claiming it was.

### The criterion v2 adopts

An **Agent** is LLM-driven reasoning work using the CrewAI Agent abstraction. It has `role`, `goal`, `backstory`. It makes LLM calls. It produces output that requires reasoning, interpretation, or generation.

A **Component** is deterministic Python. It has a `run()` method. It makes zero LLM calls. It produces output that is computed, measured, or retrieved.

The criterion is binary and verifiable. CI grep checks enforce it. A class named `*Agent` that lacks a CrewAI Agent fails the build.

### Where the criterion applies

**Agent territory:**
- Reasoning about trade-offs (GatekeeperAgent weighing style vs groundedness)
- Natural language generation (CloneAgent producing leader-voice prose)
- Interpretation, not just computation (EvaluatorAgent explaining scores)
- Qualitative output (FallbackAgent crafting context-aware redirections)

**Component territory:**
- Measurement (StyleProfileBuilder extracting 15 features per email)
- Search (Retriever finding nearest chunks via FAISS)
- Mathematical work (ScoringEngine computing cosine similarities)
- Cases where LLMs would degrade quality (numerical computation is the canonical example: LLMs hallucinate numbers)

### Alternatives that were considered and rejected

1. **Keep v1's loose vocabulary.** Rejected. v1's failure was precisely this. "Agent" used informally to mean "a thing in the pipeline" produced an architecture-vs-implementation gap that took until Day 8 to surface.

2. **Call everything an Agent (broaden the term).** Rejected. This is what marketing-driven multi-agent frameworks do. It dilutes the term until "multi-agent" means nothing more than "has more than one file." Interview-defensibility collapses.

3. **Call everything a Component (narrow the term, drop the Agent abstraction).** Rejected. The CrewAI Agent abstraction adds real value where LLM reasoning is the work — role/goal/backstory shapes the prompt, structured output via Instructor, retry logic. Throwing it away to make vocabulary simpler costs more than it saves.

### Evidence to cite

This ADR is primarily architectural reasoning, not empirical measurement. The supporting evidence is:
- v1 had 1 real Agent (ChatStyleAgent) despite claiming 5. Verified by file-by-file Day 8 audit. Cite `docs/day8-findings.md` and the v1-to-v2 mapping table in PRD §12.2.
- The senior-engineer answer to "why multi-agent over one big prompt" (specialization, testability, debuggability, observability) requires the vocabulary distinction to be meaningful. See PRD §3.4.

### Consequences worth surfacing

- Positive: file structure (`src/agents/` vs `src/components/`) makes the architecture self-documenting. New code lands in the right place by default.
- Positive: code review has a binary check rather than a judgment call. CI enforces it.
- Negative: the distinction adds vocabulary discipline overhead. Every PR comment, every commit message, every doc has to use the right word.
- Negative: edge cases exist. EvaluatorAgent is a hybrid (Agent that delegates to Component) — see ADR-011 for how that's handled.

### Java/TS parallel

Similar to Spring's `@Service` vs `@Repository` stereotype distinction. Both are Spring beans, but the stereotype declares intent and constrains where they can be used. `@Service` carries business logic; `@Repository` carries data access. Calling a `@Repository` a "service" in conversation is exactly the v1 failure mode that ADR-009 prevents in code.

---

## ADR-010: LLM-Driven Routing via GatekeeperAgent

### The problem this ADR solves

v1 routed between deliver and fallback using a fixed formula:

```
final_score = 0.4 × style_score + 0.4 × groundedness_score + 0.2 × confidence_score
if final_score >= 0.75: deliver
else: fallback
```

Day 8 evaluation revealed this formula does not discriminate. Across 20 queries, the final score distribution sat in a tight band from 0.56 to 0.76 (mean 0.66, range ~0.20 wide). No threshold value produced a meaningful split. At 0.75, almost nothing cleared. At 0.50, everything cleared. To hit a 30-40% fallback rate (the PRD §2 target), you'd need a threshold around 0.66, which means calling 0.67 "good" and 0.65 "bad" — a 0.02 gap that is meaningless.

### Why the formula fails

Three independent failure modes:
1. The formula compresses three signals into one number, losing the ability to reason about their interaction. A response with style 0.95 and groundedness 0.30 (confidently wrong) gets the same final score as a response with style 0.62 and groundedness 0.63 (mediocre across the board). One is dangerous; the other is just unimpressive. The formula can't distinguish them.
2. The score distribution is corpus-dependent. The textbook corpus and the LKML email corpus produce scores in a narrow band on this query set. A different corpus would shift the band, requiring re-tuning the threshold. The formula is not portable.
3. Weights are unjustified by data. The 0.4/0.4/0.2 split came from intuition, not measurement. No experiment validated those weights.

### The v2 decision

Replace the formula with an LLM-driven GatekeeperAgent. The Agent receives:
- The original query
- The styled response
- The retrieved chunks
- EvaluationResult (three scores + explanation + flags from EvaluatorAgent)
- The leader name

It reasons over those inputs and outputs a `RoutingDecision` with `decision: Literal["deliver", "fallback"]`, a `reasoning` string, and (if fallback) a `trigger_reason` string for FallbackAgent to use.

The reasoning prompt explicitly demands the Agent reference specific scores and flags. Example reasoning patterns the Agent can follow:
- "Style is high (0.93) but groundedness is low (0.41). The response sounds confident but the chunks don't support the central claim. Fallback."
- "Style and groundedness are both moderate (0.78, 0.71). The response cites the correct chunks. The query is in-domain. Deliver."
- "EvaluatorAgent flagged a possible misattribution (response mentions slab allocator; chunks discuss buddy allocator). Fallback regardless of scores."

### Alternatives that were considered and rejected

1. **Recalibrate the formula weights via Day 6 sensitivity sweep.** Rejected. Day 6 was a corpus-level experiment that produced null results. The score distribution is too narrow for weights to matter; any reweighting moves the band by less than the band width itself.

2. **Replace the single threshold with multi-criteria thresholds (e.g., style > 0.7 AND groundedness > 0.5).** Rejected. This is the formula problem one level up. The thresholds are still arbitrary, still corpus-dependent, and still can't reason about flag interactions. It also doesn't address the confident-hallucination case where high style hides low groundedness.

3. **Remove routing entirely; always deliver with score annotations.** Rejected. The PRD §1 product principle is "better to say 'I don't know' than to fabricate while pretending to be the leader." Removing the deliver-vs-fallback decision violates this. The system would produce confident hallucinations on out-of-domain queries.

### Evidence to cite

From `docs/day8-findings.md` (exact numbers — Opus should re-read and verify):
- 20-query evaluation, final score range 0.5589 to 0.7564
- Mean 0.6609, distribution width ~0.20
- Style mean 0.8155, only 4/20 above PRD §2a target of 0.90
- Groundedness mean 0.4585, only 1/20 above PRD §2b target of 0.60
- At threshold 0.75: ~95% fallback rate
- At threshold 0.50: 0% fallback rate
- No threshold produces the PRD §2 target band of 30-40% fallback
- Cohere reranker silent failure during the run (separate ADR-002 correction)

### Consequences worth surfacing

- Positive: routing decisions become defensible. The Agent's reasoning is the artifact you point at in interviews.
- Positive: portable to new corpora without re-tuning a threshold.
- Positive: handles confident-hallucination case via flag awareness.
- Negative: ~1-2s of LLM latency per routing decision (covered by the 8s end-to-end budget).
- Negative: LLM reasoning has variance. Mitigated by temperature=0, Instructor-structured output, and prompt-engineered demand for explicit score references.
- Negative: harder to unit-test. Compensated by recording-LLM integration tests (Layer 2 per ADR-016).

### Java/TS parallel

Similar to replacing a rule engine (Drools-style `if-then` chains over fact properties) with a domain expert service that reasons over the same facts. The rule engine is faster and deterministic, but only useful when the rules cleanly partition the input space. When the rules can't discriminate, you need judgment, not more rules.

---

## ADR-011: EvaluatorAgent Hybrid Design

### The problem this ADR solves

EvaluatorAgent has two jobs that pull in opposite directions:
1. Score the response numerically (style cosine, groundedness cosine, confidence heuristic)
2. Explain what's happening and flag specific issues for downstream reasoning

Job 1 is bad LLM territory. LLMs hallucinate numbers. They'll happily produce "groundedness: 0.73" with no actual computation behind it. Numerical scoring belongs to deterministic Python.

Job 2 is good LLM territory. "Response mentions slab allocator but chunks discuss buddy allocator" is interpretation, not computation. It requires reading the response and the chunks and noticing the mismatch. Deterministic Python can't do this without becoming an LLM.

### The v2 decision

EvaluatorAgent is a hybrid:
- The Agent is real (LLM-driven for explanation and flags).
- It delegates numerical scoring to ScoringEngine (a Component, deterministic Python).
- Output: `EvaluationResult` with three scores (from ScoringEngine), an LLM-generated `explanation: str`, and a list of `flags: list[str]`.
- Critically: `EvaluationResult` has NO `final_score` field. The weighted formula is gone. GatekeeperAgent reasons over individual scores per ADR-010.

The Agent's CrewAI definition has the role/goal/backstory focused on the explanation-and-flagging job. The prompt receives the response, chunks, and the three numerical scores already computed by ScoringEngine. It asks the LLM to interpret, not to compute.

### Alternatives that were considered and rejected

1. **Pure Component (no LLM, just scoring).** Rejected. This was v1's design. Loses the explanation and flag capability. GatekeeperAgent needs flags to make good routing decisions (e.g., "misattribution detected" is a flag that should force fallback regardless of scores). Without flags, GatekeeperAgent is back to reasoning over just three numbers, which barely improves over the formula.

2. **Pure Agent (LLM does scoring too).** Rejected. LLMs hallucinate numbers. Asking GPT-4o-mini to compute cosine similarity between style vectors is asking for fabricated values. The Day 8 finding about score distribution is itself contingent on having trustworthy scores; replacing the scoring math with an LLM would make the system non-debuggable.

3. **Two separate Agents (ScoringAgent + ExplanationAgent).** Rejected. ScoringAgent would not be an Agent under ADR-009 (no LLM reasoning). It would be a Component. So the "two Agents" framing collapses to "one Agent (Explanation) plus one Component (Scoring)" — which is exactly the hybrid design, just named worse. The hybrid framing is cleaner because the Component is internal to the Agent's responsibility surface.

### Evidence to cite

This ADR is primarily architectural reasoning. Supporting evidence:
- LLM hallucination on numerical tasks is well-documented (cite a general source if Opus has one, otherwise note as "established LLM limitation").
- The flag capability matters specifically because Day 8 surfaced cases where high style score masked low groundedness (confident hallucination). Flags catch this where scores alone don't.
- ADR-007 (groundedness scorer validation) showed the cosine scorer agrees with GPT-4o-mini's judgment at Pearson 0.82. This means deterministic scoring is good enough; we don't need LLM scoring on top.

### Consequences worth surfacing

- Positive: each part of the work uses the right tool. Scoring stays deterministic and debuggable. Explanation gains LLM interpretation.
- Positive: `EvaluationResult` is a richer artifact for GatekeeperAgent than three numbers would be.
- Negative: introduces an Agent that calls a Component, which is a new pattern. Code structure has to make this clear (the ScoringEngine is injected into EvaluatorAgent, not called as a sibling).
- Negative: testing EvaluatorAgent requires recording the LLM call for the explanation/flag part. Layer 2 integration tests (per ADR-016) handle this.

### Java/TS parallel

Similar to a Spring `@Service` that injects a `@Repository` for the data-access work but adds business reasoning on top. The Service is the unit of business logic; the Repository is the unit of data access. Pretending the Service must do everything itself, or that the Repository should add business logic, gets the layering wrong.

---

## ADR-012: LLM-Driven FallbackAgent with Templated Failsafe

### The problem this ADR solves

v1's FallbackAgent was a Python function that returned templated text:

```
"I'm not confident I can answer this accurately in [leader]'s style.
Would you like to book a call? Available slots: [slot1, slot2, slot3]"
```

Two problems:
1. The fallback path is dominant, not exceptional. Day 8 showed 30-95% of queries can route to fallback depending on threshold choice. Even at the optimistic end of v2's target band, 30-40% of queries take this path. A system where the dominant path is mechanical template is barely multi-agent in user-perceived experience.
2. The template doesn't adapt. It produces the same text regardless of query topic, leader voice, or trigger reason. A Torvalds fallback sounds the same as a Kroah-Hartman fallback. An out-of-domain question gets the same response as a domain question that failed groundedness. The fallback experience is jarringly inconsistent with the rest of the system.

### The v2 decision

FallbackAgent is a real CrewAI Agent. Inputs:
- The original query
- The leader name and style profile (for voice)
- The `trigger_reason` from GatekeeperAgent (why we're falling back)
- The retrieved chunks (for context-aware redirection)

The Agent generates:
- A leader-voiced acknowledgment of the limitation
- A suggestion of related in-domain questions the system could answer (uses the chunks to identify what's nearby)
- A calendar booking offer with three mocked time slots
- A short explanation of why the fallback was triggered (for transparency, per the original customization #10)

A templated failsafe path activates if the LLM call fails. The failsafe is a 5-line try/except that returns the v1-style template. The failsafe exists to prevent cascading failure when the LLM is down or rate-limited.

### Why the failsafe matters

Production systems need a non-LLM fallback for the fallback. If the LLM API is down and the user is already on the fallback path because GatekeeperAgent decided the response wasn't good enough, falling back from the fallback to nothing produces a system that crashes silently on its dominant path. The 5-line try/except is cheap insurance.

### Alternatives that were considered and rejected

1. **Keep v1's templated fallback (no Agent).** Rejected. The path is too important to leave mechanical. 30-70% of queries land here. A multi-agent system whose dominant path isn't multi-agent is misnamed.

2. **LLM-driven fallback with no failsafe.** Rejected. Production reliability requires the LLM-failure case to degrade gracefully, not crash. The failsafe is cheap (5 lines, no dependency on LLM) and removes a class of failure.

3. **Pre-compute fallback responses for each query type.** Rejected. The number of query types isn't bounded. Pre-computation either becomes a giant lookup table or collapses back to a small number of templates, which is what v1 had.

### Evidence to cite

- Day 8 finding: 95% fallback rate at threshold 0.75; 0% at 0.50. Even the realistic operating range (which GatekeeperAgent will produce) is 30-70%. Cite this from `docs/day8-findings.md`.
- The original customization #10 (in the customized requirements) explicitly called for fallback to explain why it was triggered. v1's template only barely satisfied this; v2's Agent satisfies it more meaningfully.

### Consequences worth surfacing

- Positive: fallback feels coherent with the rest of the system. Leader voice carries through.
- Positive: in-domain redirection turns a failure into a partial success. User gets pointed at questions the system can answer.
- Positive: failsafe means LLM downtime doesn't crash the fallback path.
- Negative: ~1-2s latency on fallback path (in addition to the work already done before fallback was decided).
- Negative: more prompt-engineering surface. Both the leader voice and the trigger explanation have to be tuned.

### Java/TS parallel

Similar to a circuit breaker pattern with a degraded-mode fallback. The primary path (LLM-generated fallback prose) is the rich version; the degraded mode (templated text) activates on primary failure. The circuit breaker pattern in resilience4j or Hystrix is the closest direct parallel — both have the "do the rich thing, but have a cheap fallback if the rich thing fails" shape.

---

## Cross-cutting notes for ADR authoring

### What changed from v1 framing to v2 framing

ADR-009 and ADR-014 are new vocabulary work. ADRs 010, 011, 012 each replace or substantially extend a v1 decision:
- ADR-010 replaces the weighted formula + 0.75 threshold (which was in v1's ADR-007 or implicit in the PRD)
- ADR-011 redefines EvaluatorAgent from "Python function producing scores + formula output" to "Agent producing scores + explanation + flags"
- ADR-012 redefines FallbackAgent from "Python function producing template" to "Agent producing context-aware prose with template failsafe"

The ADRs should be written as standalone decisions, not as deltas to v1. The reader should understand the v2 decision without needing to know v1's design.

### Format reminders (from CLAUDE.md Writing Rules)

- 5 H2 sections only, in order: Context, Decision, Alternatives Considered, Quantified Validation, Consequences
- No Cross-References, Interview Signal, or Java/TS Parallel as separate H2
- Java/TS parallel goes inline at the end of Consequences as a complete sentence in parentheses
- No bold emotional labels
- Short declarative sentences. No em-dashes, no tricolons, no "not just X but Y" structure
- For ADRs where quantified evidence is architectural rather than measured, write "Validated by architectural review, not measurement" or similar honest framing rather than fabricating numbers

### Where to find authoritative sources

- PRD §3.3, §3.4: vocabulary lock and multi-agent reasoning (ADR-009)
- PRD §4.1-§4.4: condensed decisions for all four ADRs
- PRD §7.5 and §12.3: ADR inventory and status
- `docs/day8-findings.md`: empirical evidence for ADR-010 and (indirectly) ADR-012
- `CLAUDE.md` Vocabulary Lock section: the canonical Agent vs Component definition (ADR-009)
- `CLAUDE.md` Writing Rules → ADR-specific: format contract
