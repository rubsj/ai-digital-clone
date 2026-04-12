"""LKML mbox parsing and email cleaning pipeline.

Cleaning order (PRD Decision 5):
  parse → filter by sender → strip quoted text → remove signatures →
  remove patches/diffs → remove footers → filter by word count → validate

Run standalone to check a file:
  uv run python -m src.style.email_parser data/emails/torvalds.mbox torvalds@
"""

from __future__ import annotations

import mailbox
import re
import sys
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Optional

from rich.progress import Progress, SpinnerColumn, TextColumn

from src.schemas import EmailMessage

# Patterns compiled once at module load
_QUOTE_LINE = re.compile(r"^\s*>+.*$", re.MULTILINE)
_SIG_DELIM = re.compile(r"\n-- \n", re.DOTALL)
_PATCH_BLOCK = re.compile(
    r"(^diff --git.*$|^--- .*$|^\+\+\+ .*$|^@@.*@@.*$|^[+\-][^+\-].*$){3,}",
    re.MULTILINE,
)
_DIFF_MARKER = re.compile(r"^diff --git ", re.MULTILINE)
_FOOTER_PATTERNS = [
    re.compile(r"^To unsubscribe.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^--\s*$", re.MULTILINE),
    re.compile(r"^___+\s*$", re.MULTILINE),
    re.compile(r"LKML Archive.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"https?://\S+\s*$", re.MULTILINE),
    re.compile(r"^\s*Unsubscribe\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"Linux Kernel Mailing List.*$", re.MULTILINE | re.IGNORECASE),
]
_CONTIGUOUS_PATCH = re.compile(
    r"(?:^(?:[+\-]{1}[^+\-\n]|@@|diff --git).*$\n?){3,}",
    re.MULTILINE,
)


def _compute_quote_ratio(raw_body: str) -> float:
    """Fraction of non-empty lines starting with '>' in the raw (pre-cleaned) body."""
    lines = [line for line in raw_body.splitlines() if line.strip()]
    if not lines:
        return 0.0
    quoted = sum(1 for line in lines if line.lstrip().startswith(">"))
    return quoted / len(lines)


def parse_mbox(mbox_path: Path | str, sender_filter: str) -> list[EmailMessage]:
    """Parse an mbox file and return cleaned EmailMessage objects for one sender.

    Applies the full cleaning pipeline: quote stripping, signature removal,
    patch removal, footer removal, and word-count filtering (min 20 words).
    Malformed individual messages are skipped with a log line rather than crashing.
    """
    mbox_path = Path(mbox_path)
    if not mbox_path.exists():
        raise FileNotFoundError(f"mbox file not found: {mbox_path}")

    results: list[EmailMessage] = []
    skipped_encoding = 0
    skipped_short = 0
    skipped_other = 0

    mbox = mailbox.mbox(str(mbox_path))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task(
            f"Parsing {mbox_path.name} for '{sender_filter}'...", total=None
        )

        for msg in mbox:
            progress.advance(task)
            try:
                sender = msg.get("From", "")
                if sender_filter.lower() not in sender.lower():
                    continue

                body = _extract_body(msg)
                if not body:
                    skipped_encoding += 1
                    continue

                # Compute quote ratio BEFORE cleaning strips the quoted lines
                quote_ratio = _compute_quote_ratio(body)

                cleaned = _clean_body(body)
                if _word_count(cleaned) < 20:
                    skipped_short += 1
                    continue

                timestamp = _parse_timestamp(msg)
                if timestamp is None:
                    skipped_other += 1
                    continue

                subject = msg.get("Subject", "")
                message_id = msg.get("Message-ID", "")
                to_header = msg.get("To", "")
                recipients = [r.strip() for r in to_header.split(",") if r.strip()]

                results.append(
                    EmailMessage(
                        sender=sender,
                        recipients=recipients,
                        subject=subject,
                        body=cleaned,
                        timestamp=timestamp,
                        message_id=message_id,
                        is_patch=_detect_patch(subject, body),
                        quote_ratio=quote_ratio,
                    )
                )

            except Exception as exc:  # noqa: BLE001
                skipped_other += 1
                print(f"  [warn] skipped message: {exc}", file=sys.stderr)

    print(
        f"  {mbox_path.name}: {len(results)} clean emails "
        f"(skipped: {skipped_short} short, {skipped_encoding} encoding errors, {skipped_other} other)"
    )
    return results


def _extract_body(msg: mailbox.mboxMessage) -> str:
    """Extract plain text body with encoding fallback: UTF-8 → Latin-1 → replace.

    For multipart messages, uses the first text/plain part.
    Returns empty string if no payload can be extracted.
    """
    payload = msg.get_payload(decode=True)

    if payload is None:
        # Multipart — walk to find the first text/plain part
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                if payload is not None:
                    break

    if payload is None:
        # No Content-Transfer-Encoding set (e.g. plain ASCII messages) —
        # get_payload(decode=False) returns the raw string directly
        raw = msg.get_payload(decode=False)
        if isinstance(raw, str):
            return raw
        return ""

    for encoding in ("utf-8", "latin-1"):
        try:
            return payload.decode(encoding)
        except (UnicodeDecodeError, AttributeError):
            continue

    # Final fallback — replace unmappable bytes with the replacement character
    return payload.decode("utf-8", errors="replace")


def _clean_body(body: str) -> str:
    """Apply all cleaning steps to a raw email body.

    Order matters: strip quotes first so signature/footer detection
    isn't confused by quoted signatures from other people.
    """
    body = _strip_quoted_text(body)
    body = _remove_signatures(body)
    body = _remove_patches(body)
    body = _remove_footers(body)
    # Collapse runs of blank lines left by removal steps
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


def _strip_quoted_text(text: str) -> str:
    """Remove lines starting with > (quoted replies from other people).

    We want only the leader's original words, not content they're replying to.
    Strips all levels of quoting (>, >>, >>>).
    """
    return _QUOTE_LINE.sub("", text)


def _remove_signatures(text: str) -> str:
    """Remove email signatures — text after the standard '-- \\n' delimiter.

    Also strips common sign-off line patterns at the end of the message.
    """
    # Standard RFC 3676 signature delimiter
    parts = _SIG_DELIM.split(text, maxsplit=1)
    text = parts[0]

    # Common sign-off patterns at end of message (last 3 lines)
    lines = text.rstrip().splitlines()
    # Find where the sign-off begins
    sign_off_patterns = re.compile(
        r"^\s*(Thanks[,.]?|Regards[,.]?|Cheers[,.]?|Best[,.]?|"
        r"Linus|Greg|--\s*Greg|gregh|linus|Signed-off-by:)",
        re.IGNORECASE,
    )
    cutoff = len(lines)
    for i in range(max(0, len(lines) - 5), len(lines)):
        if sign_off_patterns.match(lines[i]):
            cutoff = i
            break
    return "\n".join(lines[:cutoff])


def _remove_patches(text: str) -> str:
    """Remove contiguous diff/patch content blocks.

    Targets: lines starting with '+'/'-' in diff context, '@@' hunk headers,
    'diff --git' headers, and '---'/'+++' file markers in contiguous runs.
    A run of 3+ such lines is treated as a patch block and removed entirely.
    """
    # Remove diff --git blocks first (they anchor the rest of the patch)
    text = _DIFF_MARKER.sub("PATCH_REMOVED\n", text)
    # Remove contiguous blocks of +/-/@@ lines (3+ consecutive)
    text = _CONTIGUOUS_PATCH.sub("", text)
    # Clean up the placeholder
    text = re.sub(r"^PATCH_REMOVED\n?", "", text, flags=re.MULTILINE)
    return text


def _remove_footers(text: str) -> str:
    """Remove mailing list footers and auto-generated content.

    Removes from the first footer match to end-of-text so that
    unsubscribe links and list footers don't contaminate style features.
    """
    earliest_match = len(text)
    for pattern in _FOOTER_PATTERNS:
        m = pattern.search(text)
        if m and m.start() < earliest_match:
            earliest_match = m.start()

    return text[:earliest_match] if earliest_match < len(text) else text


def _word_count(text: str) -> int:
    """Count whitespace-delimited words in cleaned text."""
    return len(text.split())


def _parse_timestamp(msg: mailbox.mboxMessage) -> Optional[datetime]:
    """Parse the Date header; returns None for malformed or missing dates."""
    date_str = msg.get("Date", "")
    if not date_str:
        return None
    try:
        dt = parsedate_to_datetime(date_str)
        # Normalize to UTC-aware datetime
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:  # noqa: BLE001
        return None


def _detect_patch(subject: str, body: str) -> bool:
    """Heuristic: email is primarily a patch submission, not a discussion message.

    [PATCH] in subject or 3+ consecutive diff lines in body indicates a patch.
    """
    if re.search(r"\[PATCH", subject, re.IGNORECASE):
        return True
    if _DIFF_MARKER.search(body):
        return True
    return False


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m src.style.email_parser <mbox_path> <sender_filter>")
        sys.exit(1)
    emails = parse_mbox(sys.argv[1], sys.argv[2])
    print(f"\nTotal clean emails: {len(emails)}")
    for e in emails[:3]:
        print(f"\n--- Sample ---")
        print(f"  From: {e.sender}")
        print(f"  Date: {e.timestamp}")
        print(f"  Subject: {e.subject[:80]}")
        print(f"  Body ({_word_count(e.body)} words): {e.body[:200]}...")
