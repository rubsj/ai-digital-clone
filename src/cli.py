"""Click CLI for Torvalds Digital Clone.

Commands: learn, index, query, compare, evaluate.
All commands wrap existing facades — no direct LiteLLM/FAISS/Cohere imports.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import click

from src.agents.rag_agent import RAGAgent
from src.config import load_config
from src.flow import DigitalCloneFlow
from src.flow import compare_leaders as _compare_leaders
from src.rag.chunker import chunk_documents
from src.rag.corpus_loader import load_corpus
from src.schemas import FallbackResponse
from src.style.email_parser import parse_mbox
from src.style.feature_extractor import extract_features
from src.style.profile_builder import build_profile_batch, save_profile

_LEADER_DISPLAY: dict[str, str] = {
    "torvalds": "Linus Torvalds",
    "kroah-hartman": "Greg Kroah-Hartman",
}

_CONFIG_KEY: dict[str, str] = {
    "torvalds": "torvalds",
    "kroah-hartman": "kroah_hartman",
}


@click.group()
def cli() -> None:
    """Torvalds Digital Clone — command-line interface.

    \b
    Typical workflow (run once per leader / corpus refresh):
      1. learn     — build a style profile from a leader's LKML mbox.
      2. index     — build the FAISS retrieval index from the corpus.

    \b
    Then any number of:
      3. query     — ask a single question as one leader.
         compare   — ask the same question to both leaders side-by-side.
         evaluate  — run a JSON query set through both leaders and
                     produce score JSON + PRD §7d charts.

    `learn` and `index` are independent and can run in either order, but
    both must complete before `query`, `compare`, or `evaluate` will work.
    All commands wrap `DigitalCloneFlow` — they never call LiteLLM / FAISS /
    Cohere directly.
    """


@cli.command()
@click.option("--leader", type=click.Choice(["torvalds", "kroah-hartman"]), required=True)
@click.option("--mbox", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--output", type=click.Path(path_type=Path), default=None)
def learn(leader: str, mbox: Path | None, output: Path | None) -> None:
    """Build a leader's style profile from their LKML mbox (rebuild from scratch).

    Parses the mbox, filters by sender, extracts 15-dimensional style
    features per email, and aggregates them into a StyleProfile JSON.

    \b
    When to run:
      - First-time setup for a new leader.
      - When you have a refreshed mbox and want to rebuild the profile
        from scratch (no incremental update is exposed).

    \b
    Required inputs:
      --leader   one of {torvalds, kroah-hartman}
      --mbox     optional; defaults to the leader's path in config
      --output   optional; defaults to the leader's profile path

    Produces a StyleProfile JSON at the output path; downstream `query`
    and `compare` will read this profile to style their responses.
    """
    config = load_config()
    leader_cfg = config.leaders[_CONFIG_KEY[leader]]
    mbox_path = mbox or Path(leader_cfg.mbox_path)
    output_path = output or Path(leader_cfg.profile_path)

    click.echo(f"Parsing {mbox_path} for {leader_cfg.name}…")
    emails = parse_mbox(mbox_path, sender_filter=leader_cfg.email_filter)
    click.echo(f"  Parsed {len(emails)} emails.")

    features_list = [extract_features(e) for e in emails]
    profile = build_profile_batch(leader_cfg.name, features_list)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_profile(profile, output_path)
    click.echo(f"  Profile saved to {output_path} ({profile.email_count} emails).")


@cli.command()
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path), default=None)
def index(config_path: Path | None) -> None:
    """Build the FAISS retrieval index from the textbook corpus.

    Loads the open-phi/textbooks computer-science slice, chunks each
    document per the chunking config, embeds the chunks with the
    configured embedding model, and writes a FAISS index to disk.

    \b
    When to run:
      - First-time setup of the RAG pipeline.
      - Whenever you change embedding model, chunking config, or want
        to pick up new corpus content.

    \b
    Required inputs:
      --config   optional; defaults to the standard AppConfig load path

    Produces a FAISS index on disk that `query`, `compare`, and
    `evaluate` will load at runtime to retrieve grounding chunks.
    """
    config = load_config(config_path)
    click.echo("Loading corpus…")
    docs = load_corpus()
    click.echo(f"  Loaded {len(docs)} documents.")
    chunks = chunk_documents(docs, config)
    click.echo(f"  Created {len(chunks)} chunks.")
    agent = RAGAgent(config=config)
    agent.build(chunks)
    click.echo("  FAISS index saved.")


@cli.command()
@click.argument("query_text")
@click.option("--leader", type=click.Choice(["torvalds", "kroah-hartman"]), required=True)
def query(query_text: str, leader: str) -> None:
    """Ask a single question as one leader through the full pipeline.

    Runs the full DigitalCloneFlow: retrieve → rerank → style →
    evaluate → route. If the final score is below the 0.75 threshold
    (ADR-005), a FallbackResponse is returned with a calendar booking
    link instead of a styled answer.

    \b
    When to use:
      - Ad-hoc exploration of how a single leader would respond to a
        question, with full score breakdown.

    Prerequisites: `learn` (for the chosen leader) and `index` must
    both have been run at least once.

    Output: response text + style / groundedness / confidence / final
    scores + explanation, printed to the terminal.
    """
    display_name = _LEADER_DISPLAY[leader]
    click.echo(f"Querying as {display_name}…")
    flow = DigitalCloneFlow()
    flow.kickoff(inputs={"query": query_text, "leader": display_name})
    result = flow.state.final_output

    if isinstance(result, FallbackResponse):
        click.echo("\n[FALLBACK TRIGGERED]")
        click.echo(f"Reason    : {result.trigger_reason}")
        click.echo(f"Summary   : {result.context_summary}")
        click.echo(f"Unstyled  : {result.unstyled_response}")
        click.echo(f"Calendar  : {result.calendar_link}")
    else:
        ev = result.evaluation
        click.echo(f"\n{display_name} responds:\n{result.response}")
        click.echo(
            f"\nScores — style: {ev.style_score:.2f}  "
            f"groundedness: {ev.groundedness_score:.2f}  "
            f"confidence: {ev.confidence_score:.2f}  "
            f"final: {ev.final_score:.2f}"
        )
        click.echo(f"Explanation: {ev.explanation}")


@cli.command("compare")
@click.argument("query_text")
def compare_cmd(query_text: str) -> None:
    """Ask the same question to both leaders side-by-side.

    Optimization: retrieves grounding chunks once (single FAISS call +
    rerank), then runs the style + evaluate stages twice — once per
    leader. Returns both responses with their score breakdowns for
    direct comparison.

    \b
    When to use:
      - Demonstrating leader-style differentiation on identical input.
      - Qualitative spot-checks of profile distinctness.

    Prerequisites: `learn` (for BOTH leaders) and `index` must both
    have been run.

    Output: two response blocks (Torvalds + Kroah-Hartman) with scores,
    printed side-by-side to the terminal.
    """
    click.echo("Running dual-leader comparison…")
    result = _compare_leaders(query_text)

    for label, resp in [
        ("Linus Torvalds", result.torvalds),
        ("Greg Kroah-Hartman", result.kroah_hartman),
    ]:
        click.echo(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
        if isinstance(resp, FallbackResponse):
            click.echo("[FALLBACK TRIGGERED]")
            click.echo(f"Reason  : {resp.trigger_reason}")
            click.echo(f"Summary : {resp.context_summary}")
            click.echo(f"Calendar: {resp.calendar_link}")
        else:
            click.echo(resp.response)
            ev = resp.evaluation
            click.echo(
                f"style: {ev.style_score:.2f}  groundedness: {ev.groundedness_score:.2f}  "
                f"confidence: {ev.confidence_score:.2f}  final: {ev.final_score:.2f}"
            )


@cli.command()
@click.option(
    "--queries",
    type=click.Path(exists=True, path_type=Path),
    default=Path("data/eval/queries_v1.json"),
)
@click.option("--output-dir", type=click.Path(path_type=Path), default=Path("results/"))
def evaluate(queries: Path, output_dir: Path) -> None:
    """Run a JSON query set through both leaders and write the score report.

    Iterates over a JSON list of evaluation queries, runs each one
    through both leaders via DigitalCloneFlow, aggregates per-query /
    per-leader scores and fallback flags, and writes a timestamped
    JSON report to the output directory.

    \b
    When to use:
      - End-of-iteration regression check against a stable query set.
      - Producing the portfolio score charts (PRD §7d) — Phase 3 will
        wire chart generation into this command.

    \b
    Required inputs:
      --queries     path to a JSON query set; defaults to
                    data/eval/queries_v1.json
      --output-dir  where the report (and Phase 3 charts) land;
                    defaults to results/

    Prerequisites: `learn` (for BOTH leaders) and `index` must both
    have been run.

    Output: results/evaluation_<timestamp>.json. Phase 3 adds 5 PNGs
    to results/charts/ on each run.
    """
    with open(queries) as f:
        query_set = json.load(f)

    click.echo(f"Running {len(query_set)} queries through both leaders…")
    records: list[dict] = []
    for item in query_set:
        q = item["query"]
        click.echo(f"  [{item['id']}] {q[:60]}…")
        for cfg_key, display_name in [
            ("torvalds", "Linus Torvalds"),
            ("kroah_hartman", "Greg Kroah-Hartman"),
        ]:
            flow = DigitalCloneFlow()
            flow.kickoff(inputs={"query": q, "leader": display_name})
            out = flow.state.final_output
            if isinstance(out, FallbackResponse):
                records.append({"id": item["id"], "leader": cfg_key, "fallback": True})
            else:
                ev = out.evaluation
                records.append(
                    {
                        "id": item["id"],
                        "leader": cfg_key,
                        "fallback": False,
                        "style_score": ev.style_score,
                        "groundedness_score": ev.groundedness_score,
                        "confidence_score": ev.confidence_score,
                        "final_score": ev.final_score,
                    }
                )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"evaluation_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    click.echo(f"\nResults written to {out_path}")


if __name__ == "__main__":
    cli()
