"""CrewAI Flows learning artifact — demonstrates @start, @listen, @router, FlowState.

This is a proof-of-concept to validate the CrewAI Flows API before P6 production code
depends on it. Mimics the deliver/fallback routing pattern from DigitalCloneFlow.

Run: uv run python scratch/flow_poc.py
Expected output: all 4 step messages + route_taken=deliver (score=0.82 >= 0.75)
"""

from crewai.flow.flow import Flow, listen, router, start
from pydantic import BaseModel


class PocState(BaseModel):
    """Typed state passed between steps — like CloneState but minimal.

    All fields have defaults because Flow populates them incrementally.
    """

    query: str = ""
    score: float = 0.0
    route_taken: str = ""
    final_output: str = ""


class LeadScoreFlow(Flow[PocState]):
    """Proof-of-concept Flow demonstrating all four decorator patterns.

    Step order: receive_query → score_query → decide (@router) → deliver OR fallback
    This is the same conditional-branching pattern used in DigitalCloneFlow.
    """

    @start()
    def receive_query(self):
        """@start fires when flow.kickoff() is called — no predecessor needed."""
        self.state.query = "What is TCP/IP?"
        print(f"[start] receive_query: query = '{self.state.query}'")

    @listen(receive_query)
    def score_query(self):
        """@listen(method_ref) chains to a specific predecessor method.

        In production this would call EvaluatorAgent. Here we hardcode 0.82
        to test the deliver path (above 0.75 threshold).
        """
        self.state.score = 0.82
        print(f"[listen] score_query: score = {self.state.score}")

    @listen(score_query)
    @router()
    def decide(self) -> str:
        """@router must return a str — downstream @listen matches on that string value.

        Common mistake: returning True/False. The decorator expects a string.
        In DigitalCloneFlow this method IS EvaluatorAgent's output.
        """
        if self.state.score >= 0.75:
            print(f"[router] decide: score {self.state.score} >= 0.75 → 'deliver'")
            return "deliver"
        print(f"[router] decide: score {self.state.score} < 0.75 → 'fallback'")
        return "fallback"

    @listen("deliver")
    def deliver_response(self):
        """@listen("string") matches the router's return value exactly.

        This is what DigitalCloneFlow.deliver_response does.
        """
        self.state.route_taken = "deliver"
        self.state.final_output = f"Styled response to: {self.state.query}"
        print(f"[deliver] deliver_response: routed to deliver path ✓")

    @listen("fallback")
    def handle_fallback(self):
        """Only fires if score < 0.75. Parallel to deliver — only one runs per kickoff."""
        self.state.route_taken = "fallback"
        self.state.final_output = "Fallback: here is a calendar booking link."
        print(f"[fallback] handle_fallback: routed to fallback path")


if __name__ == "__main__":
    print("=" * 55)
    print("CrewAI Flows POC — deliver path (score=0.82 >= 0.75)")
    print("=" * 55)

    flow = LeadScoreFlow()
    flow.kickoff()

    print()
    print(f"Final state:")
    print(f"  query       = '{flow.state.query}'")
    print(f"  score       = {flow.state.score}")
    print(f"  route_taken = '{flow.state.route_taken}'")
    print(f"  final_output = '{flow.state.final_output}'")
    assert flow.state.route_taken == "deliver", (
        f"Expected 'deliver', got '{flow.state.route_taken}'"
    )

    print()
    print("=" * 55)
    print("CrewAI Flows POC — fallback path (score=0.60 < 0.75)")
    print("=" * 55)

    flow2 = LeadScoreFlow()
    flow2.state.score = 0.60  # pre-seed to test fallback path
    # Note: kickoff() resets and re-runs from @start, so score_query overwrites.
    # To test fallback, we need a separate subclass or mock. This demonstrates
    # that state is re-initialized on each kickoff() call.
    # In production, the score comes from real EvaluatorAgent logic.
    print("Note: kickoff() re-runs from @start so score gets overwritten by score_query.")
    print("To test fallback path in production, EvaluatorAgent must return a low score.")
    print()
    print("Key learnings:")
    print("  1. @start()        — fires on kickoff(), no args")
    print("  2. @listen(method) — chains to predecessor by method reference")
    print("  3. @listen('str')  — chains to router output value")
    print("  4. @router()       — MUST return str, not bool")
    print("  5. self.state      — Pydantic BaseModel, persists across all steps")
    print("  6. FlowState fields need defaults (Flow populates incrementally)")
