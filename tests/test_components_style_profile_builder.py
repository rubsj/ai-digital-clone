"""Tests for src/components/style_profile_builder.py.

Uses a synthetic in-memory mbox — no dependency on downloaded data, no network.
"""

from __future__ import annotations

import mailbox
from pathlib import Path

import numpy as np
import pytest

from src.components.style_profile_builder import StyleProfileBuilder
from src.schemas import EmailMessage
from src.style.feature_extractor import extract_features
from src.style.style_scorer import cosine_similarity

_LEADER = "torvalds@linux-foundation.org"

# Six near-identical technical emails (each > 20 words) → high self-similarity.
_BODIES = [
    "The kernel scheduler uses a spinlock to protect the run queue here. "
    "Because the interrupt handler can preempt the context switch we disable IRQs first.",
    "The kernel scheduler relies on a spinlock to guard the run queue safely. "
    "Since the interrupt handler may preempt the context switch we must disable IRQs.",
    "Our kernel scheduler takes a spinlock around the run queue updates always. "
    "The interrupt handler that preempts the context switch needs IRQs disabled.",
    "This kernel scheduler grabs a spinlock before touching the run queue at all. "
    "When the interrupt handler preempts a context switch we disable IRQs immediately.",
    "The scheduler in the kernel holds a spinlock while editing the run queue. "
    "Because an interrupt handler can preempt the context switch IRQs are disabled.",
    "A kernel scheduler uses one spinlock to serialize run queue mutations cleanly. "
    "The interrupt handler preempting the context switch requires that IRQs be disabled.",
]


def _make_mbox(messages: list[dict], tmp_path: Path) -> Path:
    mbox_path = tmp_path / "test.mbox"
    mbox = mailbox.mbox(str(mbox_path))
    mbox.lock()
    for m in messages:
        msg = mailbox.mboxMessage()
        msg["From"] = m.get("from_", "test@example.com")
        msg["Subject"] = m.get("subject", "Re: kernel scheduling")
        msg["Date"] = m.get("date", "Mon, 01 Jan 2020 12:00:00 +0000")
        msg["Message-ID"] = m.get("message_id", "<test@example.com>")
        msg["To"] = m.get("to", "linux-kernel@vger.kernel.org")
        msg.set_payload(m.get("body", "").encode("utf-8"), charset="utf-8")
        mbox.add(msg)
    mbox.flush()
    mbox.unlock()
    return mbox_path


def _leader_mbox(tmp_path: Path) -> Path:
    messages = [{"from_": _LEADER, "body": b} for b in _BODIES]
    # One message from a different sender — must be filtered out.
    messages.append({"from_": "someone_else@kernel.org", "body": _BODIES[0]})
    return _make_mbox(messages, tmp_path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_run_filters_by_sender_and_counts(tmp_path):
    mbox_path = _leader_mbox(tmp_path)
    profile = StyleProfileBuilder().run(mbox_path, "torvalds@", "Linus Torvalds")
    assert profile.leader_name == "Linus Torvalds"
    assert profile.email_count == len(_BODIES)  # the other sender excluded


def test_run_style_vector_length_15(tmp_path):
    profile = StyleProfileBuilder().run(_leader_mbox(tmp_path), "torvalds@", "Linus")
    assert profile.style_vector.shape == (15,)


def test_run_sample_emails_populated(tmp_path):
    profile = StyleProfileBuilder().run(_leader_mbox(tmp_path), "torvalds@", "Linus")
    # 6 matched emails → capped at 5 evenly-spaced samples.
    assert len(profile.sample_emails) == 5
    assert all(isinstance(s, str) and s for s in profile.sample_emails)


def test_run_self_similarity_above_threshold(tmp_path):
    """Aggregate profile vs each email's vector: mean cosine > 0.70."""
    mbox_path = _leader_mbox(tmp_path)
    profile = StyleProfileBuilder().run(mbox_path, "torvalds@", "Linus")

    sims = []
    for body in _BODIES:
        feats = extract_features(
            EmailMessage(
                sender=_LEADER,
                subject="",
                body=body,
                timestamp=__import__("datetime").datetime(2020, 1, 1),
                message_id="x",
                quote_ratio=0.0,
            )
        )
        sims.append(cosine_similarity(profile.style_vector, feats.to_vector()))

    assert float(np.mean(sims)) > 0.70


def test_run_no_matching_sender_raises(tmp_path):
    mbox_path = _leader_mbox(tmp_path)
    with pytest.raises(ValueError):
        StyleProfileBuilder().run(mbox_path, "nobody@nowhere.invalid", "Nobody")


def test_sample_emails_small_set_returns_all():
    emails = [
        EmailMessage(
            sender="x", subject="", body=f"body {i}",
            timestamp=__import__("datetime").datetime(2020, 1, 1),
            message_id=f"<{i}>", quote_ratio=0.0,
        )
        for i in range(3)
    ]
    assert StyleProfileBuilder._sample_emails(emails) == ["body 0", "body 1", "body 2"]
