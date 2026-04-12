"""15-feature style extractor for LKML emails.

Public API:
    extract_features(email: EmailMessage) -> StyleFeatures

All 15 features produce values in [0, 1], matching the order expected by
StyleFeatures.to_vector() in src/schemas.py.

Patterns are compiled once at module load (same convention as email_parser.py).
No new dependencies — only re, collections.Counter, and numpy.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

import numpy as np

from src.schemas import EmailMessage, StyleFeatures

# ---------------------------------------------------------------------------
# Compiled patterns (module-level, compiled once)
# ---------------------------------------------------------------------------

_WORDS = re.compile(r"\b[a-zA-Z']+\b")
_SENTENCES = re.compile(r"[^.!?]*[.!?]+")
_QUESTION_SENT = re.compile(r"[^.!?]*\?")

# Punctuation counts
_EXCLAMATION = re.compile(r"!")
_ELLIPSIS = re.compile(r"\.\.\.")
_DASH = re.compile(r"--|—")
_SEMICOLON = re.compile(r";")
_COLON = re.compile(r":")

# Greeting detection (first non-empty line)
_GREET_HI = re.compile(r"^\s*hi\b", re.IGNORECASE)
_GREET_HELLO = re.compile(r"^\s*hello\b", re.IGNORECASE)
_GREET_HEY = re.compile(r"^\s*hey\b", re.IGNORECASE)
_GREET_DEAR = re.compile(r"^\s*dear\b", re.IGNORECASE)

# Reasoning connectors
_RE_BECAUSE = re.compile(r"\bbecause\b", re.IGNORECASE)
_RE_THEREFORE = re.compile(r"\btherefore\b", re.IGNORECASE)
_RE_HOWEVER = re.compile(r"\bhowever\b", re.IGNORECASE)
_RE_BUT = re.compile(r"\bbut\b", re.IGNORECASE)
_RE_SO = re.compile(r"\bso\b", re.IGNORECASE)
_RE_IF_THEN = re.compile(r"\bif\b.{0,60}\bthen\b", re.IGNORECASE)
_RE_THE_THING = re.compile(r"\bthe\s+thing\s+is\b", re.IGNORECASE)

# Technical kernel/systems terms (~50)
_TECH_TERMS = frozenset([
    "kernel", "syscall", "mutex", "spinlock", "semaphore", "interrupt", "irq",
    "scheduler", "preempt", "context", "switch", "memory", "allocation", "heap",
    "stack", "pointer", "buffer", "overflow", "race", "condition", "deadlock",
    "livelock", "thread", "process", "fork", "exec", "socket", "tcp", "udp",
    "filesystem", "inode", "block", "device", "driver", "module", "namespace",
    "cgroup", "container", "virtualization", "hypervisor", "paravirt", "mmap",
    "pagefault", "tlb", "cache", "pipeline", "branch", "prediction", "register",
    "instruction", "assembly", "abi", "syscall", "ioctl", "dma",
])

# Code pattern for lines
_CODE_LINE = re.compile(
    r"(^\s{4,}|^\t|\{|\}|->|::|#include|#define|sizeof\(|return\s+[^;]+;|[a-z_]+\([^)]*\);)",
    re.MULTILINE,
)

# Technical depth signals
_FUNC_REF = re.compile(r"\b[a-z_][a-z0-9_]+\(\)", re.IGNORECASE)
_FILE_PATH = re.compile(r"\b\w+\.[ch]\b")
_CONFIG_REF = re.compile(r"\bCONFIG_\w+")
_COMMIT_SHA = re.compile(r"\b[0-9a-f]{7,40}\b")

# Patch language keywords
_PATCH_APPLIED = re.compile(r"\bapplied\b", re.IGNORECASE)
_PATCH_NAK = re.compile(r"\bnak\b|\bnack\b", re.IGNORECASE)
_PATCH_ACKED = re.compile(r"\backed[-_]?by\b", re.IGNORECASE)
_PATCH_REVIEWED = re.compile(r"\breviewed[-_]?by\b", re.IGNORECASE)
_PATCH_LOOKS_GOOD = re.compile(r"\blooks\s+good\b", re.IGNORECASE)
_PATCH_PLEASE_FIX = re.compile(r"\bplease\s+fix\b", re.IGNORECASE)
_PATCH_RESUBMIT = re.compile(r"\bresubmit", re.IGNORECASE)

# Formality: contractions
_CONTRACTIONS = re.compile(
    r"\b(don't|can't|won't|isn't|doesn't|it's|I'm|you're|they're|we're|"
    r"that's|there's|what's|couldn't|shouldn't|wouldn't|haven't|hadn't|"
    r"didn't|weren't|aren't|he's|she's|let's|I've|I'd|I'll)\b",
    re.IGNORECASE,
)
# Formality: formal words
_FORMAL_WORDS = frozenset([
    "regarding", "furthermore", "consequently", "nevertheless", "therefore",
    "specifically", "respectively", "additionally", "subsequently", "hereby",
    "notwithstanding", "pursuant", "wherein", "thereof", "henceforth",
])
# Formality: profanity
_PROFANITY = frozenset([
    "crap", "stupid", "idiot", "damn", "hell", "bullshit", "shit", "ass",
    "clueless", "braindead",
])
# Formality: first-person
_FIRST_PERSON = re.compile(r"\b(I|me|my|I'm|I've|I'd|I'll)\b")

# Sentiment keywords (LKML-adapted)
_NEUTRAL_TECH = frozenset([
    "patch", "commit", "merge", "config", "driver", "kernel", "module",
    "revert", "bisect", "rebase", "build", "test", "debug", "fix", "bug",
    "nak", "ack", "kill", "abort", "fatal",
])
_POSITIVE_WORDS = frozenset([
    "good", "great", "nice", "excellent", "works", "correct", "right",
    "perfect", "fine", "clean", "proper", "reasonable",
])
_NEGATIVE_WORDS = frozenset([
    "broken", "wrong", "stupid", "crap", "clueless", "horrible", "buggy",
    "idiotic", "insane", "braindead", "garbage", "useless", "terrible",
    "disgusting", "nonsense",
])
_POSITIVE_PHRASES = ["looks good", "well done"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_features(email: EmailMessage) -> StyleFeatures:
    """Extract 15 style features from a single cleaned EmailMessage.

    All returned values are in [0, 1] and match the field order expected by
    StyleFeatures.to_vector() in src/schemas.py.
    """
    body = email.body
    return StyleFeatures(
        avg_message_length=_avg_message_length(body),
        greeting_patterns=_greeting_patterns(body),
        punctuation_patterns=_punctuation_patterns(body),
        capitalization_ratio=_capitalization_ratio(body),
        question_frequency=_question_frequency(body),
        vocabulary_richness=_vocabulary_richness(body),
        common_phrases=_common_phrases(body),
        reasoning_patterns=_reasoning_patterns(body),
        sentiment_distribution=_sentiment_distribution(body),
        formality_level=_formality_level(body),
        technical_terminology=_technical_terminology(body),
        code_snippet_freq=_code_snippet_freq(body),
        quote_reply_ratio=email.quote_ratio,  # pre-computed before cleaning
        patch_language=_patch_language(body),
        technical_depth=_technical_depth(body),
    )


# ---------------------------------------------------------------------------
# Private feature helpers
# ---------------------------------------------------------------------------


def _avg_message_length(body: str) -> float:
    """Idx 0: word count normalized to [0, 1] with 500 words saturating at 1.0."""
    word_count = len(body.split())
    return min(word_count / 500.0, 1.0)


def _greeting_patterns(body: str) -> dict[str, float]:
    """Idx 1: binary presence of greeting types in the first non-empty line.

    Keys: hi, hello, hey, dear, none (no greeting detected).
    Values: 0.0 or 1.0 — only one key is 1.0 per email.
    """
    first_line = ""
    for line in body.splitlines():
        if line.strip():
            first_line = line
            break

    if _GREET_HI.match(first_line):
        return {"hi": 1.0, "hello": 0.0, "hey": 0.0, "dear": 0.0, "none": 0.0}
    if _GREET_HELLO.match(first_line):
        return {"hi": 0.0, "hello": 1.0, "hey": 0.0, "dear": 0.0, "none": 0.0}
    if _GREET_HEY.match(first_line):
        return {"hi": 0.0, "hello": 0.0, "hey": 1.0, "dear": 0.0, "none": 0.0}
    if _GREET_DEAR.match(first_line):
        return {"hi": 0.0, "hello": 0.0, "hey": 0.0, "dear": 1.0, "none": 0.0}
    return {"hi": 0.0, "hello": 0.0, "hey": 0.0, "dear": 0.0, "none": 1.0}


def _punctuation_patterns(body: str) -> dict[str, float]:
    """Idx 2: per-character frequencies of 6 punctuation types, capped at 1.0.

    Uses min(count / len(body) * 50, 1.0) so ~1 per 50 chars saturates.
    """
    if not body:
        return {"exclamation": 0.0, "question": 0.0, "ellipsis": 0.0,
                "dash": 0.0, "semicolon": 0.0, "colon": 0.0}
    n = len(body)

    def _rate(pattern: re.Pattern[str]) -> float:
        return min(len(pattern.findall(body)) / n * 50, 1.0)

    return {
        "exclamation": _rate(_EXCLAMATION),
        "question": _rate(re.compile(r"\?")),
        "ellipsis": _rate(_ELLIPSIS),
        "dash": _rate(_DASH),
        "semicolon": _rate(_SEMICOLON),
        "colon": _rate(_COLON),
    }


def _capitalization_ratio(body: str) -> float:
    """Idx 3: word-level ALL-CAPS ratio.

    Counts words where len >= 2 and word.isupper() — captures emphatic
    usage like NEVER/WRONG without inflating on sentence-start capitals.
    """
    words = _WORDS.findall(body)
    if not words:
        return 0.0
    allcaps = sum(1 for w in words if len(w) >= 2 and w.isupper())
    return allcaps / len(words)


def _question_frequency(body: str) -> float:
    """Idx 4: question sentences / total sentences."""
    sentences = _SENTENCES.findall(body)
    if not sentences:
        return 0.0
    questions = sum(1 for s in sentences if s.rstrip().endswith("?"))
    return questions / len(sentences)


def _vocabulary_richness(body: str) -> float:
    """Idx 5: type-token ratio — unique words / total words."""
    words = [w.lower() for w in _WORDS.findall(body)]
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def _common_phrases(body: str) -> list[str]:
    """Idx 6: top-20 bigrams and trigrams appearing >= 2 times.

    Returns a list[str]; to_vector() converts via min(len/20, 1.0).
    """
    tokens = [w.lower() for w in _WORDS.findall(body)]
    if len(tokens) < 2:
        return []

    bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]
    trigrams = [
        f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}"
        for i in range(len(tokens) - 2)
    ]

    counts = Counter(bigrams + trigrams)
    # Keep only phrases appearing >= 2 times, top 20 by frequency
    return [phrase for phrase, cnt in counts.most_common(20) if cnt >= 2]


def _reasoning_patterns(body: str) -> dict[str, float]:
    """Idx 7: reasoning connector frequencies per sentence, capped at 1.0.

    count / sentence_count normalized with * 5 so ~1 per 5 sentences saturates.
    """
    sentences = _SENTENCES.findall(body)
    n_sent = max(len(sentences), 1)

    def _norm(pattern: re.Pattern[str]) -> float:
        return min(len(pattern.findall(body)) / n_sent * 5, 1.0)

    return {
        "because": _norm(_RE_BECAUSE),
        "therefore": _norm(_RE_THEREFORE),
        "however": _norm(_RE_HOWEVER),
        "but": _norm(_RE_BUT),
        "so": _norm(_RE_SO),
        "if_then": _norm(_RE_IF_THEN),
        "the_thing_is": _norm(_RE_THE_THING),
    }


def _sentiment_distribution(body: str) -> dict[str, float]:
    """Idx 8: LKML-adapted positive/negative/neutral fractions.

    Skips words in the neutral-technical list to avoid LKML domain noise.
    Multi-word positive phrases ("looks good") checked before tokenization.
    """
    body_lower = body.lower()

    # Check multi-word positive phrases first
    pos_phrase_hits = sum(1 for p in _POSITIVE_PHRASES if p in body_lower)

    tokens = [w.lower() for w in _WORDS.findall(body)]
    # Filter out neutral-technical tokens
    filtered = [t for t in tokens if t not in _NEUTRAL_TECH]

    pos_hits = pos_phrase_hits + sum(1 for t in filtered if t in _POSITIVE_WORDS)
    neg_hits = sum(1 for t in filtered if t in _NEGATIVE_WORDS)
    total = pos_hits + neg_hits

    if total == 0:
        return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}

    positive_frac = pos_hits / total
    negative_frac = neg_hits / total
    neutral_frac = max(1.0 - positive_frac - negative_frac, 0.0)
    return {
        "positive": positive_frac,
        "negative": negative_frac,
        "neutral": neutral_frac,
    }


def _formality_level(body: str) -> float:
    """Idx 9: weighted mean of 5 sub-signals, each in [0, 1].

    formality = 0.25 * formal_word_rate
              + 0.20 * (1 - contraction_rate)
              + 0.20 * avg_sent_len_norm
              + 0.20 * (1 - profanity_rate)
              + 0.15 * (1 - first_person_rate)
    """
    words = [w.lower() for w in _WORDS.findall(body)]
    n_words = max(len(words), 1)
    sentences = _SENTENCES.findall(body)
    n_sent = max(len(sentences), 1)

    # 1. Formal word rate
    formal_count = sum(1 for w in words if w in _FORMAL_WORDS)
    formal_word_rate = min(formal_count / n_words * 50, 1.0)

    # 2. Contraction rate (inverted)
    contraction_count = len(_CONTRACTIONS.findall(body))
    contraction_rate = min(contraction_count / n_words * 20, 1.0)

    # 3. Average sentence length (longer = more formal)
    avg_sent_len_norm = min((n_words / n_sent) / 25.0, 1.0)

    # 4. Profanity rate (inverted)
    profanity_count = sum(1 for w in words if w in _PROFANITY)
    profanity_rate = min(profanity_count / n_words * 100, 1.0)

    # 5. First-person rate (inverted)
    fp_count = len(_FIRST_PERSON.findall(body))
    first_person_rate = min(fp_count / n_words * 20, 1.0)

    formality = (
        0.25 * formal_word_rate
        + 0.20 * (1.0 - contraction_rate)
        + 0.20 * avg_sent_len_norm
        + 0.20 * (1.0 - profanity_rate)
        + 0.15 * (1.0 - first_person_rate)
    )
    return float(np.clip(formality, 0.0, 1.0))


def _technical_terminology(body: str) -> float:
    """Idx 10: fraction of words matching ~50 kernel/systems terms.

    Normalized with * 20 so ~1 matching word per 20 saturates at 1.0.
    """
    words = [w.lower() for w in _WORDS.findall(body)]
    if not words:
        return 0.0
    tech_count = sum(1 for w in words if w in _TECH_TERMS)
    return min(tech_count / len(words) * 20, 1.0)


def _code_snippet_freq(body: str) -> float:
    """Idx 11: fraction of lines matching code patterns, capped at 1.0.

    Normalized with * 5 so 20% code lines saturates at 1.0.
    """
    lines = body.splitlines()
    if not lines:
        return 0.0
    code_lines = sum(1 for line in lines if _CODE_LINE.search(line))
    return min(code_lines / len(lines) * 5, 1.0)


# quote_reply_ratio (Idx 12) is read directly from email.quote_ratio in extract_features()


def _patch_language(body: str) -> dict[str, float]:
    """Idx 13: binary presence (0.0 / 1.0) of 8 LKML patch-review terms."""
    return {
        "applied": 1.0 if _PATCH_APPLIED.search(body) else 0.0,
        "nak": 1.0 if _PATCH_NAK.search(body) else 0.0,
        "acked_by": 1.0 if _PATCH_ACKED.search(body) else 0.0,
        "reviewed_by": 1.0 if _PATCH_REVIEWED.search(body) else 0.0,
        "looks_good": 1.0 if _PATCH_LOOKS_GOOD.search(body) else 0.0,
        "please_fix": 1.0 if _PATCH_PLEASE_FIX.search(body) else 0.0,
        "resubmit": 1.0 if _PATCH_RESUBMIT.search(body) else 0.0,
    }


def _technical_depth(body: str) -> float:
    """Idx 14: composite technical depth score, capped at 1.0.

    Weighted sum of four signals:
      0.35 * tech_term_density   (same as technical_terminology but unscaled)
      0.30 * function_ref_density  (b[a-z_]+() patterns)
      0.20 * file_path_density    (*.c / *.h paths)
      0.15 * config_sha_density   (CONFIG_ refs + commit SHAs)
    Each sub-signal normalized independently then combined.
    """
    words = _WORDS.findall(body)
    n_words = max(len(words), 1)
    lines = body.splitlines()
    n_lines = max(len(lines), 1)

    # Sub-signal 1: tech term density (raw ratio * 20 capped)
    tech_count = sum(1 for w in words if w.lower() in _TECH_TERMS)
    tech_density = min(tech_count / n_words * 20, 1.0)

    # Sub-signal 2: function reference density (count per line * 3 capped)
    func_refs = len(_FUNC_REF.findall(body))
    func_density = min(func_refs / n_lines * 3, 1.0)

    # Sub-signal 3: file path density (count per line * 5 capped)
    file_paths = len(_FILE_PATH.findall(body))
    file_density = min(file_paths / n_lines * 5, 1.0)

    # Sub-signal 4: CONFIG_ + commit SHA density (count per line * 5 capped)
    config_refs = len(_CONFIG_REF.findall(body))
    sha_refs = len(_COMMIT_SHA.findall(body))
    config_sha_density = min((config_refs + sha_refs) / n_lines * 5, 1.0)

    depth = (
        0.35 * tech_density
        + 0.30 * func_density
        + 0.20 * file_density
        + 0.15 * config_sha_density
    )
    return float(np.clip(depth, 0.0, 1.0))
