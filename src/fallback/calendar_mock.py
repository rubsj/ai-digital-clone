"""Calendar mock: generate N available meeting slots without any external API.

Slots are on the next N business days (Mon–Fri), at random times within
business hours (9 AM–5 PM PT). A seed parameter makes output deterministic
for tests.

Format: "Tuesday, April 16, 2026 at 10:30 AM PT"
"""

from __future__ import annotations

import random
from datetime import date, datetime, timedelta


def _next_business_days(start: date, n: int) -> list[date]:
    """Return the next n weekdays after (not including) start."""
    days: list[date] = []
    current = start
    while len(days) < n:
        current += timedelta(days=1)
        if current.weekday() < 5:  # Mon=0 … Fri=4
            days.append(current)
    return days


def _random_slot_time(rng: random.Random) -> tuple[int, int]:
    """Return (hour, minute) within 9:00–16:30 so slot ends by 5 PM."""
    hour = rng.randint(9, 16)
    minute = rng.choice([0, 30])
    # Avoid 4:30 PM slot running to 5:00 PM edge — allow it; meetings can be 30 min.
    return hour, minute


def _format_slot(d: date, hour: int, minute: int) -> str:
    """Format a date + time as the canonical human-readable slot string."""
    dt = datetime(d.year, d.month, d.day, hour, minute)
    # e.g. "Tuesday, April 16, 2026 at 10:30 AM PT"
    time_str = dt.strftime("%-I:%M %p")  # "10:30 AM" (no leading zero)
    date_str = dt.strftime("%A, %B %-d, %Y")  # "Tuesday, April 16, 2026"
    return f"{date_str} at {time_str} PT"


def generate_available_slots(
    n: int = 3,
    seed: int | None = None,
    _today: date | None = None,
) -> list[str]:
    """Generate n human-readable available meeting slots on the next n business days.

    Args:
        n:      Number of slots to generate (default 3).
        seed:   RNG seed for deterministic output in tests.
        _today: Override today's date (for testing). Defaults to date.today().

    Returns:
        List of slot strings, one per business day, in chronological order.
    """
    rng = random.Random(seed)
    today = _today or date.today()
    days = _next_business_days(today, n)
    return [_format_slot(d, *_random_slot_time(rng)) for d in days]
