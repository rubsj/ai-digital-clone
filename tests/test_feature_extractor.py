"""Tests for src/style/feature_extractor.py.

Coverage target: >= 90% of feature_extractor.py.
Each of the 15 features has at least one "present" and one "zero/minimal" test.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from src.schemas import EmailMessage, StyleFeatures
from src.style.feature_extractor import (
    _avg_message_length,
    _capitalization_ratio,
    _code_snippet_freq,
    _common_phrases,
    _formality_level,
    _greeting_patterns,
    _patch_language,
    _punctuation_patterns,
    _question_frequency,
    _reasoning_patterns,
    _sentiment_distribution,
    _technical_depth,
    _technical_terminology,
    _vocabulary_richness,
    extract_features,
)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _make_email(body: str = "", quote_ratio: float = 0.0) -> EmailMessage:
    return EmailMessage(
        sender="torvalds@linux-foundation.org",
        subject="Test",
        body=body,
        timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
        message_id="<test@example.com>",
        quote_ratio=quote_ratio,
    )


_REALISTIC_BODY = (
    "Look, the problem here is that the kernel scheduler is fundamentally broken. "
    "You cannot just add a mutex around the spinlock and call it done. "
    "That's the WRONG approach and it will cause a race condition every single time. "
    "If you want to fix this properly, you need to think about the memory allocation path. "
    "The buffer overflow happens because nobody checks the pointer before dereferencing it. "
    "I've seen this pattern before — it's clueless engineering. "
    "Please fix this before the next merge window. "
    "The kernel cannot ship with a known deadlock in the interrupt handler."
)


# ---------------------------------------------------------------------------
# Feature 0: avg_message_length
# ---------------------------------------------------------------------------


def test_avg_message_length_zero_words():
    assert _avg_message_length("") == 0.0


def test_avg_message_length_short():
    result = _avg_message_length("hello world")
    assert 0.0 < result < 1.0


def test_avg_message_length_saturates_at_one():
    long_body = "word " * 600
    assert _avg_message_length(long_body) == 1.0


def test_avg_message_length_500_words():
    body = "word " * 500
    assert _avg_message_length(body) == 1.0


def test_avg_message_length_250_words():
    body = "word " * 250
    assert abs(_avg_message_length(body) - 0.5) < 0.01


# ---------------------------------------------------------------------------
# Feature 1: greeting_patterns
# ---------------------------------------------------------------------------


def test_greeting_hi():
    result = _greeting_patterns("Hi Linus,\nThe patch looks good.")
    assert result["hi"] == 1.0
    assert len(result) == 1


def test_greeting_hello():
    result = _greeting_patterns("Hello everyone,\nThis is broken.")
    assert result["hello"] == 1.0
    assert len(result) == 1


def test_greeting_hey():
    result = _greeting_patterns("Hey,\nStop doing that.")
    assert result["hey"] == 1.0
    assert len(result) == 1


def test_greeting_dear():
    result = _greeting_patterns("Dear maintainer,\nRegarding this patch...")
    assert result["dear"] == 1.0
    assert len(result) == 1


def test_greeting_none():
    # No greeting → empty dict so dict_mean = 0.0 (discriminative)
    result = _greeting_patterns("The scheduler is broken.\nFix it now.")
    assert result == {}


def test_greeting_returns_only_found_key():
    result = _greeting_patterns("Hi there,\nSomething.")
    assert set(result.keys()) == {"hi"}


def test_greeting_empty_body():
    # Empty body → no greeting → empty dict
    result = _greeting_patterns("")
    assert result == {}


# ---------------------------------------------------------------------------
# Feature 2: punctuation_patterns
# ---------------------------------------------------------------------------


def test_punctuation_empty():
    result = _punctuation_patterns("")
    assert all(v == 0.0 for v in result.values())


def test_punctuation_exclamation():
    result = _punctuation_patterns("Wrong! This is wrong! Fix it now!")
    assert result["exclamation"] > 0.0


def test_punctuation_question():
    result = _punctuation_patterns("What is this? Why did you do that?")
    assert result["question"] > 0.0


def test_punctuation_colon():
    result = _punctuation_patterns("The issue is: the buffer overflows here: done.")
    assert result["colon"] > 0.0


def test_punctuation_all_in_range():
    result = _punctuation_patterns("Hello world... this is fine; or is it? Maybe -- not!")
    assert all(0.0 <= v <= 1.0 for v in result.values())


def test_punctuation_returns_all_keys():
    result = _punctuation_patterns("test")
    assert set(result.keys()) == {"exclamation", "question", "ellipsis", "dash", "semicolon", "colon"}


# ---------------------------------------------------------------------------
# Feature 3: capitalization_ratio
# ---------------------------------------------------------------------------


def test_capitalization_no_allcaps():
    assert _capitalization_ratio("hello world this is a sentence.") == 0.0


def test_capitalization_with_allcaps():
    result = _capitalization_ratio("This is WRONG and you should NEVER do it.")
    assert result > 0.0


def test_capitalization_empty():
    assert _capitalization_ratio("") == 0.0


def test_capitalization_sentence_start_not_counted():
    # "Hello" starts the sentence — word-level: not isupper() on the whole word
    result = _capitalization_ratio("Hello World This Is Capitalized.")
    assert result == 0.0


def test_capitalization_in_range():
    result = _capitalization_ratio("NEVER do THIS WRONG thing in the KERNEL.")
    assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# Feature 4: question_frequency
# ---------------------------------------------------------------------------


def test_question_frequency_no_questions():
    assert _question_frequency("This is a statement. Another one here.") == 0.0


def test_question_frequency_all_questions():
    result = _question_frequency("What is this? Why is it broken? How do we fix it?")
    assert result == 1.0


def test_question_frequency_mixed():
    result = _question_frequency("This is broken. What happened? Fix it now.")
    assert 0.0 < result < 1.0


def test_question_frequency_empty():
    assert _question_frequency("") == 0.0


# ---------------------------------------------------------------------------
# Feature 5: vocabulary_richness
# ---------------------------------------------------------------------------


def test_vocabulary_richness_all_unique():
    result = _vocabulary_richness("alpha beta gamma delta epsilon")
    assert result == 1.0


def test_vocabulary_richness_all_same():
    result = _vocabulary_richness("word word word word word")
    assert result == pytest.approx(1 / 5)


def test_vocabulary_richness_empty():
    assert _vocabulary_richness("") == 0.0


def test_vocabulary_richness_in_range():
    result = _vocabulary_richness(_REALISTIC_BODY)
    assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# Feature 6: common_phrases
# ---------------------------------------------------------------------------


def test_common_phrases_empty():
    assert _common_phrases("") == []


def test_common_phrases_short():
    assert _common_phrases("hello world") == []


def test_common_phrases_repeated_bigram():
    body = "memory leak memory leak memory leak here now please"
    result = _common_phrases(body)
    assert "memory leak" in result


def test_common_phrases_max_20():
    # Feed lots of unique bigrams — result must be <= 20
    words = [f"word{i}" for i in range(100)]
    body = " ".join(words * 3)  # repeat 3x so bigrams appear >= 2 times
    result = _common_phrases(body)
    assert len(result) <= 20


def test_common_phrases_returns_list_of_strings():
    body = "the kernel the kernel the kernel is broken here now"
    result = _common_phrases(body)
    assert isinstance(result, list)
    assert all(isinstance(p, str) for p in result)


# ---------------------------------------------------------------------------
# Feature 7: reasoning_patterns
# ---------------------------------------------------------------------------


def test_reasoning_because_present():
    result = _reasoning_patterns("The fix fails because the pointer is null.")
    assert result["because"] > 0.0


def test_reasoning_all_zero():
    result = _reasoning_patterns("This code works fine now.")
    for v in result.values():
        assert v == 0.0


def test_reasoning_however_present():
    result = _reasoning_patterns("The code compiles. However, it crashes at runtime.")
    assert result["however"] > 0.0


def test_reasoning_the_thing_is():
    result = _reasoning_patterns("The thing is, you need to check the return value.")
    assert result["the_thing_is"] > 0.0


def test_reasoning_all_keys_present():
    result = _reasoning_patterns("test")
    assert set(result.keys()) == {"because", "therefore", "however", "but", "so", "if_then", "the_thing_is"}


def test_reasoning_all_in_range():
    result = _reasoning_patterns(_REALISTIC_BODY)
    assert all(0.0 <= v <= 1.0 for v in result.values())


# ---------------------------------------------------------------------------
# Feature 8: sentiment_distribution
# ---------------------------------------------------------------------------


def test_sentiment_positive_present():
    result = _sentiment_distribution("This looks great! The code is clean and correct. Perfect work.")
    assert "positive" in result
    assert result["positive"] > 0.0


def test_sentiment_negative_present():
    result = _sentiment_distribution("This is broken and wrong. Horrible buggy garbage code.")
    assert "negative" in result
    assert result["negative"] > 0.0


def test_sentiment_neutral_only():
    # Only LKML neutral-technical words → no emotional content → empty dict
    result = _sentiment_distribution("The kernel patch commit module driver config.")
    assert result == {}


def test_sentiment_looks_good_phrase():
    result = _sentiment_distribution("This looks good to me. I'll apply it.")
    assert result.get("positive", 0.0) > 0.0


def test_sentiment_no_emotional_content_empty_dict():
    # "test" has no positive or negative words
    result = _sentiment_distribution("test")
    assert result == {}


def test_sentiment_word_rates_vary():
    # High positive density → rate closer to 1.0
    positive_body = ("good " * 10) + "word " * 10  # 50% positive → rate = min(10/20*10, 1.0)
    result = _sentiment_distribution(positive_body)
    assert result.get("positive", 0.0) > 0.0


def test_sentiment_all_in_range():
    result = _sentiment_distribution(_REALISTIC_BODY)
    assert all(0.0 <= v <= 1.0 for v in result.values())


# ---------------------------------------------------------------------------
# Feature 9: formality_level
# ---------------------------------------------------------------------------


def test_formality_low_profanity_contractions():
    # Lots of contractions and profanity → low formality
    result = _formality_level(
        "I can't believe this crap. It's totally wrong. Don't do it. "
        "This is bullshit and I won't accept it. I'm done."
    )
    assert result < 0.5


def test_formality_high_formal_words():
    result = _formality_level(
        "Furthermore, regarding the implementation, consequently the system fails. "
        "Therefore we must subsequently address this. Additionally, pursuant to "
        "the specification, the module should be fixed."
    )
    assert result > 0.0


def test_formality_empty():
    result = _formality_level("")
    assert 0.0 <= result <= 1.0


def test_formality_in_range():
    result = _formality_level(_REALISTIC_BODY)
    assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# Feature 10: technical_terminology
# ---------------------------------------------------------------------------


def test_technical_terminology_present():
    result = _technical_terminology(
        "The kernel scheduler uses a mutex to protect the spinlock. "
        "A race condition can occur during interrupt handling."
    )
    assert result > 0.0


def test_technical_terminology_absent():
    result = _technical_terminology("The cat sat on the mat. Nice day today.")
    assert result == 0.0


def test_technical_terminology_saturates():
    # Lots of tech terms
    tech_body = " ".join(["kernel mutex spinlock scheduler interrupt"] * 50)
    assert _technical_terminology(tech_body) == 1.0


def test_technical_terminology_in_range():
    result = _technical_terminology(_REALISTIC_BODY)
    assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# Feature 11: code_snippet_freq
# ---------------------------------------------------------------------------


def test_code_snippet_no_code():
    result = _code_snippet_freq("This is plain prose without any code at all here.")
    assert result == 0.0


def test_code_snippet_with_code():
    code_body = (
        "The fix is straightforward:\n"
        "    if (ptr == NULL) {\n"
        "        return -ENOMEM;\n"
        "    }\n"
        "Apply this patch to the driver."
    )
    result = _code_snippet_freq(code_body)
    assert result > 0.0


def test_code_snippet_in_range():
    result = _code_snippet_freq(_REALISTIC_BODY)
    assert 0.0 <= result <= 1.0


def test_code_snippet_empty():
    assert _code_snippet_freq("") == 0.0


# ---------------------------------------------------------------------------
# Feature 12: quote_reply_ratio (via extract_features)
# ---------------------------------------------------------------------------


def test_quote_reply_ratio_from_email():
    email = _make_email("Some body text here about the kernel.", quote_ratio=0.4)
    features = extract_features(email)
    assert features.quote_reply_ratio == 0.4


def test_quote_reply_ratio_zero():
    email = _make_email("No quotes in this email at all.", quote_ratio=0.0)
    features = extract_features(email)
    assert features.quote_reply_ratio == 0.0


# ---------------------------------------------------------------------------
# Feature 13: patch_language
# ---------------------------------------------------------------------------


def test_patch_language_applied():
    result = _patch_language("I applied your patch to the tree.")
    assert result["applied"] == 1.0


def test_patch_language_nak():
    result = _patch_language("NAK — this breaks the ABI.")
    assert result["nak"] == 1.0


def test_patch_language_acked_by():
    result = _patch_language("Acked-by: Greg Kroah-Hartman <gregkh@linux.org>")
    assert result["acked_by"] == 1.0


def test_patch_language_reviewed_by():
    result = _patch_language("Reviewed-by: someone@kernel.org")
    assert result["reviewed_by"] == 1.0


def test_patch_language_looks_good():
    result = _patch_language("Looks good to me, applying.")
    assert result["looks_good"] == 1.0


def test_patch_language_please_fix():
    result = _patch_language("Please fix the commit message before resubmitting.")
    assert result["please_fix"] == 1.0
    assert result["resubmit"] == 1.0


def test_patch_language_none_present():
    result = _patch_language("The scheduler is broken and needs fixing.")
    assert all(v == 0.0 for v in result.values())


def test_patch_language_all_keys():
    result = _patch_language("test")
    assert set(result.keys()) == {
        "applied", "nak", "acked_by", "reviewed_by",
        "looks_good", "please_fix", "resubmit"
    }


# ---------------------------------------------------------------------------
# Feature 14: technical_depth
# ---------------------------------------------------------------------------


def test_technical_depth_high():
    deep_body = (
        "The kernel scheduler deadlock in interrupt() happens because "
        "schedule_timeout() calls do_fork() which touches mm->mmap. "
        "See drivers/base/core.c and include/linux/mutex.h. "
        "CONFIG_PREEMPT must be enabled. Commit abc1234 introduced this. "
        "The race between spin_lock() and mutex_lock() is a known issue."
    )
    result = _technical_depth(deep_body)
    assert result > 0.0


def test_technical_depth_zero():
    result = _technical_depth("The cat sat on the mat today.")
    assert result == 0.0


def test_technical_depth_in_range():
    result = _technical_depth(_REALISTIC_BODY)
    assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# Integration: extract_features
# ---------------------------------------------------------------------------


def test_extract_features_returns_style_features():
    email = _make_email(_REALISTIC_BODY)
    result = extract_features(email)
    assert isinstance(result, StyleFeatures)


def test_extract_features_to_vector_shape():
    email = _make_email(_REALISTIC_BODY)
    vec = extract_features(email).to_vector()
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (15,)


def test_extract_features_to_vector_all_in_range():
    email = _make_email(_REALISTIC_BODY)
    vec = extract_features(email).to_vector()
    assert np.all(vec >= 0.0)
    assert np.all(vec <= 1.0)


def test_extract_features_empty_body():
    email = _make_email("")
    result = extract_features(email)
    vec = result.to_vector()
    assert vec.shape == (15,)
    assert np.all(vec >= 0.0)
    assert np.all(vec <= 1.0)


def test_extract_features_all_15_populated():
    email = _make_email(_REALISTIC_BODY, quote_ratio=0.25)
    result = extract_features(email)
    # Greeting may be absent (empty dict is valid — most emails have no greeting)
    assert isinstance(result.greeting_patterns, dict)
    assert len(result.punctuation_patterns) > 0
    assert len(result.reasoning_patterns) > 0
    # REALISTIC_BODY has "clueless" → negative sentiment present
    assert len(result.sentiment_distribution) > 0
    assert len(result.patch_language) > 0
    # quote_reply_ratio wired through
    assert result.quote_reply_ratio == 0.25


def test_extract_features_very_long_body_caps_at_one():
    long_body = (_REALISTIC_BODY + " ") * 20
    email = _make_email(long_body)
    result = extract_features(email)
    assert result.avg_message_length == 1.0
    vec = result.to_vector()
    assert np.all(vec <= 1.0)
