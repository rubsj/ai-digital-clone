"""Validate pre-downloaded LKML mbox files and report clean email counts.

Run: uv run python scripts/validate_emails.py
Expected: >= 200 clean emails per leader, sample output for spot-check.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path so src imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.style.email_parser import _word_count, parse_mbox


def main() -> None:
    config = load_config()
    all_ok = True

    for key, leader in config.leaders.items():
        mbox_path = Path(leader.mbox_path)
        print(f"\n{'=' * 55}")
        print(f"Leader: {leader.name} ({key})")
        print(f"mbox:   {mbox_path}")

        if not mbox_path.exists():
            print(f"  ERROR: file not found at {mbox_path}")
            print(f"  Download the mbox manually from lore.kernel.org and place it at this path.")
            all_ok = False
            continue

        print(f"  File size: {mbox_path.stat().st_size / 1024 / 1024:.1f} MB")

        emails = parse_mbox(mbox_path, leader.email_filter)
        count = len(emails)
        target = 200

        status = "OK" if count >= target else "FAIL"
        print(f"\n  Result: {count} clean emails [{status}] (target: >= {target})")

        if count < target:
            all_ok = False
            print(
                f"  Tip: if cleaning is too aggressive, lower style.min_email_words "
                f"in configs/default.yaml (currently 20)"
            )

        # Spot-check: print 3 samples
        print(f"\n  Sample emails (first 3):")
        for i, email in enumerate(emails[:3], 1):
            word_cnt = _word_count(email.body)
            print(f"\n  [{i}] {email.timestamp.date()} | {email.subject[:60]}")
            print(f"       From: {email.sender[:60]}")
            print(f"       Words: {word_cnt}")
            preview = email.body[:200].replace("\n", " ")
            print(f"       Body: {preview}...")

    print(f"\n{'=' * 55}")
    if all_ok:
        print("VALIDATION PASSED — both leaders have >= 200 clean emails")
        sys.exit(0)
    else:
        print("VALIDATION FAILED — see errors above")
        sys.exit(1)


if __name__ == "__main__":
    main()
