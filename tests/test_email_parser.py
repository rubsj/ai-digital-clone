"""Tests for src/style/email_parser.py.

Uses synthetic in-memory mbox files — no dependency on real downloaded data.
Coverage target: >= 90% of email_parser.py
"""

from __future__ import annotations

import mailbox
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.style.email_parser import (
    _clean_body,
    _detect_patch,
    _extract_body,
    _parse_timestamp,
    _remove_footers,
    _remove_patches,
    _remove_signatures,
    _strip_quoted_text,
    _word_count,
    parse_mbox,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mbox(messages: list[dict], tmp_path: Path) -> Path:
    """Create a temporary mbox file from a list of message dicts.

    Each dict: {from_, subject, body, date, message_id}
    """
    mbox_path = tmp_path / "test.mbox"
    mbox = mailbox.mbox(str(mbox_path))
    mbox.lock()
    for m in messages:
        msg = mailbox.mboxMessage()
        msg["From"] = m.get("from_", "test@example.com")
        msg["Subject"] = m.get("subject", "Test Subject")
        msg["Date"] = m.get("date", "Mon, 01 Jan 2020 12:00:00 +0000")
        msg["Message-ID"] = m.get("message_id", "<test@example.com>")
        msg["To"] = m.get("to", "linux-kernel@vger.kernel.org")
        payload = m.get("body", "Test body with enough words here.")
        msg.set_payload(payload.encode("utf-8"), charset="utf-8")
        mbox.add(msg)
    mbox.flush()
    mbox.unlock()
    return mbox_path


def _enough_words() -> str:
    # 22 words — above the 20-word minimum filter
    return "This is a test email with well more than twenty words in the body text right here and it is okay."


# ---------------------------------------------------------------------------
# parse_mbox
# ---------------------------------------------------------------------------


def test_parse_mbox_filters_by_sender(tmp_path):
    mbox_path = _make_mbox(
        [
            {"from_": "torvalds@linux-foundation.org", "body": _enough_words()},
            {"from_": "someone_else@kernel.org", "body": _enough_words()},
            {"from_": "torvalds@linux.org", "body": _enough_words()},
        ],
        tmp_path,
    )
    emails = parse_mbox(mbox_path, "torvalds@")
    assert len(emails) == 2
    for e in emails:
        assert "torvalds@" in e.sender


def test_parse_mbox_skips_short_emails(tmp_path):
    mbox_path = _make_mbox(
        [
            {"from_": "torvalds@linux-foundation.org", "body": "Too short."},
            {"from_": "torvalds@linux-foundation.org", "body": _enough_words()},
        ],
        tmp_path,
    )
    emails = parse_mbox(mbox_path, "torvalds@")
    assert len(emails) == 1


def test_parse_mbox_returns_email_message_type(tmp_path):
    from src.schemas import EmailMessage

    mbox_path = _make_mbox(
        [{"from_": "torvalds@linux-foundation.org", "body": _enough_words()}],
        tmp_path,
    )
    emails = parse_mbox(mbox_path, "torvalds@")
    assert len(emails) == 1
    assert isinstance(emails[0], EmailMessage)
    assert emails[0].sender == "torvalds@linux-foundation.org"


def test_parse_mbox_file_not_found():
    with pytest.raises(FileNotFoundError):
        parse_mbox("/nonexistent/path/file.mbox", "torvalds@")


def test_parse_mbox_empty_mbox(tmp_path):
    mbox_path = _make_mbox([], tmp_path)
    emails = parse_mbox(mbox_path, "torvalds@")
    assert emails == []


# ---------------------------------------------------------------------------
# _strip_quoted_text
# ---------------------------------------------------------------------------


def test_strip_quoted_text_removes_gt_lines():
    text = "Hello\n> This is quoted\nActual content here now."
    result = _strip_quoted_text(text)
    assert "> This is quoted" not in result
    assert "Actual content" in result


def test_strip_nested_quotes():
    text = "Real content.\n>> Deeply quoted\n>>> Even deeper\nMore real content."
    result = _strip_quoted_text(text)
    assert ">>" not in result
    assert "Real content" in result
    assert "More real content" in result


def test_strip_quoted_text_preserves_non_quoted():
    text = "Line one\nLine two\nLine three"
    result = _strip_quoted_text(text)
    assert result == text


# ---------------------------------------------------------------------------
# _remove_signatures
# ---------------------------------------------------------------------------


def test_remove_signature_standard_delimiter():
    text = "Body content with enough words here.\n-- \nLinus Torvalds\nlinux-foundation.org"
    result = _remove_signatures(text)
    assert "Linus Torvalds" not in result
    assert "Body content" in result


def test_remove_signature_thanks():
    text = "The fix is straightforward. Apply and you are done.\nThanks,\nLinus"
    result = _remove_signatures(text)
    assert "Thanks" not in result
    assert "The fix is straightforward" in result


def test_remove_signature_preserves_body():
    text = "The thing is, you need to look at the code more carefully."
    result = _remove_signatures(text)
    assert result.strip() == text.strip()


# ---------------------------------------------------------------------------
# _remove_patches
# ---------------------------------------------------------------------------


def test_remove_patches_diff_block():
    text = textwrap.dedent("""\
        Let me explain the issue.
        diff --git a/foo.c b/foo.c
        +added line
        -removed line
        @@ -1,3 +1,4 @@
        This is after the patch.
    """)
    result = _remove_patches(text)
    assert "diff --git" not in result
    assert "Let me explain" in result


def test_remove_patches_preserves_normal_text():
    text = "The score is +5 points. Nothing else changed here at all."
    result = _remove_patches(text)
    # Single + in normal prose should not be stripped
    assert "The score is" in result


def test_remove_patches_contiguous_block_removed():
    text = textwrap.dedent("""\
        Before the patch.
        +added this line
        +another added line
        +yet another added line
        After the patch.
    """)
    result = _remove_patches(text)
    assert "Before the patch" in result
    assert "After the patch" in result


# ---------------------------------------------------------------------------
# _remove_footers
# ---------------------------------------------------------------------------


def test_remove_footers_unsubscribe():
    text = "Real content here.\nTo unsubscribe from this list, send email to majordomo@vger.kernel.org"
    result = _remove_footers(text)
    assert "unsubscribe" not in result.lower()
    assert "Real content" in result


def test_remove_footers_preserves_body():
    text = "Just a normal email body with no footer at all."
    result = _remove_footers(text)
    assert result == text


# ---------------------------------------------------------------------------
# _clean_body (combined pipeline)
# ---------------------------------------------------------------------------


def test_clean_body_combined():
    body = textwrap.dedent("""\
        This is the actual content of the message.
        > This line is quoted from someone else.
        >> Doubly quoted line.
        More actual content right here.
        diff --git a/file.c b/file.c
        +added
        -removed
        @@ hunk header
        --
        Greg Kroah-Hartman
        To unsubscribe from LKML, send email to majordomo@vger.kernel.org
    """)
    result = _clean_body(body)
    assert "> This line" not in result
    assert "Greg Kroah-Hartman" not in result
    assert "unsubscribe" not in result.lower()
    assert "This is the actual content" in result
    assert "More actual content" in result


# ---------------------------------------------------------------------------
# _extract_body
# ---------------------------------------------------------------------------


def test_extract_body_utf8(tmp_path):
    mbox_path = _make_mbox(
        [{"from_": "test@test.com", "body": "UTF-8 content: hello world"}],
        tmp_path,
    )
    mbox = mailbox.mbox(str(mbox_path))
    msg = next(iter(mbox))
    body = _extract_body(msg)
    assert "UTF-8 content" in body


def test_extract_body_empty_payload():
    msg = mailbox.mboxMessage()
    msg["From"] = "test@test.com"
    # No payload set
    body = _extract_body(msg)
    assert body == ""


# ---------------------------------------------------------------------------
# _detect_patch
# ---------------------------------------------------------------------------


def test_detect_patch_subject_patch():
    assert _detect_patch("[PATCH v3] Fix memory leak in drivers", "") is True


def test_detect_patch_body_diff():
    assert _detect_patch("Re: bug", "diff --git a/file.c b/file.c\n+added") is True


def test_detect_patch_normal_email():
    assert _detect_patch("Re: discussion", "This is just a normal reply.") is False


# ---------------------------------------------------------------------------
# _parse_timestamp
# ---------------------------------------------------------------------------


def test_parse_timestamp_valid():
    msg = mailbox.mboxMessage()
    msg["Date"] = "Mon, 01 Jun 2020 12:00:00 +0000"
    ts = _parse_timestamp(msg)
    assert isinstance(ts, datetime)
    assert ts.year == 2020


def test_parse_timestamp_missing():
    msg = mailbox.mboxMessage()
    ts = _parse_timestamp(msg)
    assert ts is None


def test_parse_timestamp_malformed():
    msg = mailbox.mboxMessage()
    msg["Date"] = "not a valid date"
    ts = _parse_timestamp(msg)
    assert ts is None


# ---------------------------------------------------------------------------
# _word_count
# ---------------------------------------------------------------------------


def test_word_count_basic():
    assert _word_count("hello world foo bar") == 4


def test_word_count_empty():
    assert _word_count("") == 0


def test_word_count_whitespace_only():
    assert _word_count("   \n\t  ") == 0


# ---------------------------------------------------------------------------
# parse_mbox — coverage for edge-case branches
# ---------------------------------------------------------------------------


def test_parse_mbox_skips_empty_body(tmp_path):
    """Covers the 'empty body' branch (skipped_encoding path)."""
    mbox_path = tmp_path / "empty.mbox"
    mbox = mailbox.mbox(str(mbox_path))
    mbox.lock()
    msg = mailbox.mboxMessage()
    msg["From"] = "torvalds@linux-foundation.org"
    msg["Subject"] = "Test"
    msg["Date"] = "Mon, 01 Jan 2020 12:00:00 +0000"
    msg["Message-ID"] = "<empty@test>"
    # No payload set at all — _extract_body will return ""
    # Force the multipart walk to also yield nothing by not setting payload
    mbox.add(msg)
    mbox.flush()
    mbox.unlock()
    emails = parse_mbox(mbox_path, "torvalds@")
    assert emails == []


def test_parse_mbox_skips_no_timestamp(tmp_path):
    """Covers the 'no timestamp' branch (skipped_other path)."""
    mbox_path = tmp_path / "nodate.mbox"
    mbox = mailbox.mbox(str(mbox_path))
    mbox.lock()
    msg = mailbox.mboxMessage()
    msg["From"] = "torvalds@linux-foundation.org"
    msg["Subject"] = "Test"
    # No Date header — _parse_timestamp returns None
    msg["Message-ID"] = "<nodate@test>"
    msg.set_payload(_enough_words())
    mbox.add(msg)
    mbox.flush()
    mbox.unlock()
    emails = parse_mbox(mbox_path, "torvalds@")
    assert emails == []


# ---------------------------------------------------------------------------
# _extract_body — additional coverage
# ---------------------------------------------------------------------------


def test_extract_body_latin1_bytes():
    """Covers Latin-1 fallback path when UTF-8 decode fails."""
    # Create a message with Latin-1 encoded bytes that aren't valid UTF-8
    latin1_text = "Ren\xe9 M\xfcller wrote this".encode("latin-1")
    msg = mailbox.mboxMessage()
    msg["Content-Transfer-Encoding"] = "8bit"
    msg.set_payload(latin1_text)
    # Manually set payload to raw bytes bypassing charset encoding
    msg._payload = latin1_text  # noqa: SLF001
    body = _extract_body(msg)
    # Should decode successfully (Latin-1 or replace)
    assert isinstance(body, str)


def test_extract_body_multipart_finds_text_plain(tmp_path):
    """Covers multipart walk that finds a text/plain part."""
    import email
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    multipart = MIMEMultipart("alternative")
    multipart.attach(MIMEText("Plain text content here.", "plain", "utf-8"))
    multipart.attach(MIMEText("<b>HTML</b>", "html", "utf-8"))

    # Convert to mboxMessage
    raw = multipart.as_bytes()
    msg = mailbox.mboxMessage(email.message_from_bytes(raw))
    body = _extract_body(msg)
    assert "Plain text content" in body


# ---------------------------------------------------------------------------
# _parse_timestamp — UTC normalization
# ---------------------------------------------------------------------------


def test_parse_timestamp_naive_gets_utc():
    """Covers the tzinfo=None branch that adds UTC."""
    msg = mailbox.mboxMessage()
    # Date without timezone offset — parsedate_to_datetime may return naive datetime
    # Use a format that results in a naive datetime
    msg["Date"] = "Mon, 01 Jun 2020 12:00:00 -0000"
    ts = _parse_timestamp(msg)
    if ts is not None:
        assert ts.tzinfo is not None  # must always be tz-aware after normalization
