# ADR-014: Agent and Component Inventory

**Status:** Accepted

**Date:** 2026-05-26

## Context

ADR-009 set the criterion that separates an Agent from a Component. An Agent is LLM-driven reasoning built on the CrewAI Agent abstraction with role, goal, and backstory; a Component is deterministic Python with a `run()` method and no LLM call. That criterion is written to be reusable across projects, so it stays deliberately abstract about which units a given system contains.

P6 v2 needs the concrete companion to that rule, an inventory of the specific units and the reason each is classified the way it is. Without it the architecture lives only in the `src/agents/` and `src/components/` directory listings, which is implementation rather than a decision record. A reviewer reading a file tree cannot reconstruct why Retriever is a Component or why GatekeeperAgent is an Agent. This is the reason for the two-ADR split: ADR-009 carries the reusable criterion and ADR-014 applies it to P6, which keeps the criterion portable while keeping the inventory auditable.

The inventory also fixes the orchestrator's status. The Flow is plain orchestration and is not itself an Agent, the central correction in the ADR-001 rewrite, and there is no PlannerAgent (PRD §3.2).

## Decision

P6 v2 has exactly four Agents, three Components, and one Flow orchestrator, specified in PRD §5.1 and §5.2.

**2026-06-01 inventory correction (ADR-018):** Agent count 4 → 3; Component count 3 → 4. GatekeeperAgent was reclassified from Agent to Component (renamed Gatekeeper) because ADR-018 replaced its LLM decision with deterministic arithmetic; it now satisfies the ADR-009 Component criterion (deterministic `run()`, no LLM). File moved from `src/agents/gatekeeper_agent.py` to `src/components/gatekeeper.py`. Readers following ADR-018's or ADR-010's file references should look there.

The three Agents are LLM-driven, specified in PRD §5.1 to live under `src/agents/`:

- CloneAgent generates the response in the leader's voice (PRD §5.1.1).
- EvaluatorAgent is a hybrid: it calls the ScoringEngine Component for the numerical scores and uses an LLM only to produce the explanation and flags (PRD §5.1.2, ADR-011).
- FallbackAgent writes a leader-appropriate fallback message and carries a short templated failsafe for when the LLM call fails (PRD §5.1.4).

The four Components are deterministic, specified in PRD §5.2 with a `run()` method under `src/components/`:

- Retriever embeds the query and searches FAISS for the top-20 candidates, then reranks to the top-5 with Cohere, with no LLM in the chain (PRD §5.2.1).
- StyleProfileBuilder parses the leader's mbox and extracts the 15 style features per email, then builds the StyleProfile (PRD §5.2.2).
- ScoringEngine computes the three quality scores (style, groundedness, confidence) with deterministic math and no LLM (PRD §5.2.3).
- Gatekeeper makes the deterministic deliver-or-fallback routing decision from computed scores; no LLM (ADR-018). File: `src/components/gatekeeper.py`.

The orchestrator is DigitalCloneFlow, a CrewAI `Flow[CloneState]` subclass whose decorators (`@start`, `@listen`, `@router`) define the step order (PRD §3.2). The Flow is not an Agent, and there is no PlannerAgent.

## Alternatives Considered

- Classify Retriever and ScoringEngine as Agents. Rejected. Both are deterministic pipelines with no LLM reasoning step. Retriever's chain is embedding and FAISS search followed by a Cohere rerank, and ScoringEngine is cosine and similarity math. Wrapping either in the CrewAI Agent abstraction adds prompt scaffolding with no autonomy benefit, and it fails the ADR-009 criterion.
- Make DigitalCloneFlow an Agent, a PlannerAgent sitting on top of the Flow. Rejected. The Flow already owns the step order and the typed state, and the routing happens through its `@router` decorator, so there is no per-step LLM reasoning in the Flow itself to justify an Agent wrapper. This is the "agent that manages agents" anti-pattern, and it was part of the v1 confusion the rework removes, since v1 referred to the Flow itself as the PlannerAgent.
- Collapse CloneAgent and FallbackAgent into one Agent with a mode parameter. Rejected. The two have different goals (generate a response versus gracefully decline), and they carry different prompts and different evaluation expectations. Merging them obscures responsibility and complicates testing, where two Agents give two prompts and two test suites.

## Quantified Validation

- The inventory is a specification, not a measurement. PRD §5.1 specifies the four Agents and PRD §5.2 specifies the three Components, with each unit's role and its input and output contract. Those sections are the authoritative source this ADR justifies.
- The count and the classification principle come from PRD §3.2 (the architecture at a glance) and PRD §3.3, which states the rule as LLMs where reasoning adds value and deterministic code where computation is the work. That is the same criterion ADR-009 records.
- The inventory is a deliberate reclassification rather than an accident, and PRD §12.2 maps it unit by unit. The v1 `rag_agent.py` façade becomes the Retriever Component. The v1 `evaluator_steps.py` and `fallback_steps.py`, which were Python functions, become the real EvaluatorAgent and FallbackAgent. The v1 Flow `@router` that returned a string becomes the GatekeeperAgent that the `@router` step now calls. The one v1 unit that was already a real Agent, `style_crew.py`, becomes CloneAgent and stays an Agent.
- This ADR rests on that specification and mapping rather than on a measured result, so the verification it points forward to is the Day 14 codebase audit (PRD §12.5), which checks the implemented `src/` against this inventory.

## Consequences

Once the rework lands, the directory structure mirrors the inventory: three files in `src/agents/` and four in `src/components/`, with nothing outside those directories named like an Agent. (Corrected 2026-06-01 per ADR-018: Gatekeeper moved from `src/agents/` to `src/components/`.) The inventory is a gate rather than a snapshot, because any Agent or Component added in later work has to pass the ADR-009 criterion before it earns a place in either directory. The orchestrator's fixed status, plain Flow and no PlannerAgent, means future steps are added as Agents or Components rather than by growing the Flow into something that reasons. (In Java terms the dividing line is closer to a strategy that reasons against a worker that computes than to Spring's `@Service` versus `@Component` stereotypes, since what classifies a unit here is whether an LLM does the reasoning.)
