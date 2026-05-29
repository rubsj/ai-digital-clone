"""StyleProfileBuilder Component: mbox → cleaned emails → 15 features → profile.

LLM-free. Wraps the v1 src/style/ pipeline, all of which is frozen by ADR-013:
the email cleaning pipeline (parse_mbox) and the 15-feature extractor and the
aggregation/EMA math (profile_builder) produce identical outputs.

One sanctioned addition (approved this session): populate the new
StyleProfile.sample_emails field with 3-5 already-cleaned email bodies for
CloneAgent to use as in-context style examples. Sampling carries cleaned
text forward only — it touches neither the cleaning pipeline nor the features,
so the ADR-013 freeze holds. The sampling lives here, not in the frozen
profile_builder, so that module stays untouched.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.config import AppConfig, load_config
from src.schemas import EmailMessage, StyleProfile
from src.style.email_parser import parse_mbox
from src.style.feature_extractor import extract_features
from src.style.profile_builder import build_profile_batch

logger = logging.getLogger(__name__)

_MAX_SAMPLE_EMAILS = 5


class StyleProfileBuilder:
    """Build a leader StyleProfile from an mbox archive."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self._config = config or load_config()

    def run(
        self,
        mbox_path: Path | str,
        sender_filter: str,
        leader_name: str,
    ) -> StyleProfile:
        """Parse → extract features → aggregate → attach sample emails."""
        emails = parse_mbox(mbox_path, sender_filter)
        if not emails:
            raise ValueError(
                f"No emails matched sender filter {sender_filter!r} in {mbox_path}"
            )

        features = [extract_features(e) for e in emails]
        profile = build_profile_batch(
            leader_name=leader_name,
            features_list=features,
            alpha=self._config.style.alpha,
        )
        return profile.model_copy(
            update={"sample_emails": self._sample_emails(emails)}
        )

    @staticmethod
    def _sample_emails(emails: list[EmailMessage]) -> list[str]:
        """Up to 5 cleaned bodies, evenly spaced across the set (deterministic)."""
        n = len(emails)
        if n <= _MAX_SAMPLE_EMAILS:
            return [e.body for e in emails]
        step = n / _MAX_SAMPLE_EMAILS
        return [emails[int(i * step)].body for i in range(_MAX_SAMPLE_EMAILS)]
