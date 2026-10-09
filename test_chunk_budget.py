"""Offline tests for the chunk token budget helpers and the audio quality guard.

Run with:  python test_chunk_budget.py

No GPU or model download required — only the pure planning functions from
audiobook_converter are exercised (importing the module pulls in torch &
friends, but nothing is loaded or run).
"""

import numpy as np

from audiobook_converter import (
    ChunkBudgetError,
    estimate_prefill_tokens,
    estimate_spoken_words,
    plan_chunk_generation,
    check_chunk_audio,
    check_chunk_audio_quality,
    FASTER_QWEN_MAX_SEQ_LEN,
)


def make_text(words: int) -> str:
    return " ".join(["word"] * words)


def expect_no_error(fn, label):
    try:
        return fn()
    except Exception as e:
        raise AssertionError(f"{label}: unexpected error: {e}")


def expect_budget_error(fn, label, deterministic=True):
    try:
        fn()
    except ChunkBudgetError as e:
        assert e.deterministic is deterministic, (
            f"{label}: expected deterministic={deterministic}, got {e.deterministic}"
        )
        return e
    raise AssertionError(f"{label}: expected ChunkBudgetError, none raised")


SR = 24000  # Qwen3-TTS codec sample rate


def make_clean(seconds: int) -> np.ndarray:
    """Speech-like audio: two tones with a slow amplitude envelope."""
    t = np.arange(seconds * SR) / SR
    env = 0.6 + 0.4 * np.sin(2 * np.pi * 0.3 * t)
    return (0.1 * np.sin(2 * np.pi * 200 * t) + 0.05 * np.sin(2 * np.pi * 600 * t)) * env


def make_noise(seconds: int, amp: float = 0.05) -> np.ndarray:
    return amp * np.random.default_rng(42).standard_normal(seconds * SR)


def main():
    max_seq_len = FASTER_QWEN_MAX_SEQ_LEN
    print(f"FASTER_QWEN_MAX_SEQ_LEN = {max_seq_len}")

    # 1. Default chunk (150 words) fits in the default budget
    #    (calibrated rate: 85 wpm -> 105.9s expected -> 1589 tokens at 12.5 tokens/s with 1.2x safety)
    mnt, expected = expect_no_error(
        lambda: plan_chunk_generation(make_text(150), max_seq_len, "custom_voice"),
        "150 words @ default max_seq_len",
    )
    assert mnt == 1589, f"expected 1589, got {mnt}"
    assert abs(expected - 150 / 85 * 60) < 1e-9, f"unexpected expected value: {expected}"
    print(f"  OK: 150 words -> max_new_tokens={mnt}, expected ~{expected:.0f}s")

    # 2. Very short chunk hits the 256-token floor
    mnt, _ = expect_no_error(
        lambda: plan_chunk_generation(make_text(10), max_seq_len, "custom_voice"),
        "10 words (floor)",
    )
    assert mnt == 256, f"expected 256 floor, got {mnt}"
    print(f"  OK: 10 words -> max_new_tokens={mnt} (floor)")

    # 3. 1000 words does NOT fit in 8192 -> deterministic error with a remedy
    err = expect_budget_error(
        lambda: plan_chunk_generation(make_text(1000), 8192, "custom_voice"),
        "1000 words @ 8192",
    )
    assert "Raise FASTER_QWEN_MAX_SEQ_LEN" in str(err)
    print(f"  OK: 1000 words @ 8192 rejected: {err}")

    # 4. 300 words fits in 4096 (would be ~212s of audio: beyond the quality
    #    safe zone, but still budget-feasible)
    mnt, expected = expect_no_error(
        lambda: plan_chunk_generation(make_text(300), 4096, "custom_voice"),
        "300 words @ 4096",
    )
    assert mnt == 3177, f"expected 3177, got {mnt}"
    assert abs(expected - 300 / 85 * 60) < 1e-9, f"unexpected expected value: {expected}"
    print(f"  OK: 300 words @ 4096 -> max_new_tokens={mnt}")

    # 5. 400 words does NOT fit in 4096 at the calibrated rate -> error
    expect_budget_error(
        lambda: plan_chunk_generation(make_text(400), 4096, "custom_voice"),
        "400 words @ 4096",
    )
    print("  OK: 400 words @ 4096 rejected (needs 8192)")

    # 6. Voice-clone mode adds the reference margin (audio budget unchanged)
    mnt_clone, _ = expect_no_error(
        lambda: plan_chunk_generation(make_text(400), 8192, "voice_clone"),
        "400 words @ 8192 voice_clone",
    )
    mnt_custom, _ = plan_chunk_generation(make_text(400), 8192, "custom_voice")
    assert mnt_clone == mnt_custom, "voice mode must not change the audio budget"
    prefill_margin = estimate_prefill_tokens(make_text(400), "voice_clone") - \
        estimate_prefill_tokens(make_text(400), "custom_voice")
    assert prefill_margin == 512, f"expected 512 ref margin, got {prefill_margin}"
    print(f"  OK: voice_clone prefill margin = {prefill_margin}")

    # 7. Empty text -> deterministic error
    expect_budget_error(
        lambda: plan_chunk_generation("", 8192, "custom_voice"),
        "empty text",
    )
    print("  OK: empty text rejected")

    # 7b. Numbers count as one spoken word per digit ("1923" = nineteen twenty-three)
    assert estimate_spoken_words("In 1923 he wrote") == 7, estimate_spoken_words("In 1923 he wrote")
    assert estimate_spoken_words("pp. 124-57") == 6
    mnt_plain, exp_plain = plan_chunk_generation("In the year he wrote", max_seq_len, "custom_voice")
    mnt_digits, exp_digits = plan_chunk_generation("In 1923 he wrote", max_seq_len, "custom_voice")
    assert exp_digits > exp_plain, "a year must be expected to take longer than a short word"
    print(f"  OK: spoken-length estimate counts digits ({exp_plain:.1f}s -> {exp_digits:.1f}s)")

    # 8. Post-generation checks (700-word chunk: 6048-token budget = 504s cap)
    # Cap hit: generation ran into the 504s cap instead of emitting EOS. At
    # real narration speed that means a runaway, so it is retryable.
    err = expect_budget_error(
        lambda: check_chunk_audio(502.0, 6048, 420.0),
        "cap hit",
        deterministic=False,
    )
    assert "token cap" in str(err)
    print(f"  OK: cap hit detected (retryable): {err}")

    # Runaway that stopped on its own below the cap (2026-09-11, chunk 36:
    # 140 words, 113.9s on attempt 1 vs 48.2s on the retry) -> retryable
    mnt_140, expected_140 = plan_chunk_generation(make_text(140), max_seq_len, "custom_voice")
    err = expect_budget_error(
        lambda: check_chunk_audio(113.9, mnt_140, expected_140),
        "runaway below cap",
        deterministic=False,
    )
    assert "longer than expected" in str(err)
    print(f"  OK: runaway below the cap flagged (retryable): {err}")
    expect_no_error(lambda: check_chunk_audio(48.2, mnt_140, expected_140), "chunk 36 retry")
    print("  OK: chunk 36's good retry (48.2s) passes")

    # Slow but complete narration (the slowest observed body chunk, 136 wpm,
    # and a deliberately slower 110 wpm) must not be flagged as runaway
    expect_no_error(lambda: check_chunk_audio(150 / 136 * 60, 1589, 105.9), "136 wpm")
    expect_no_error(lambda: check_chunk_audio(150 / 110 * 60, 1589, 105.9), "110 wpm")
    print("  OK: slow narration (136 and 110 wpm) passes")

    # Very short chunk (a heading): the absolute slack keeps it from being flagged
    expect_no_error(lambda: check_chunk_audio(2.5, 256, 2 / 85 * 60), "short heading")
    print("  OK: 2-word heading at 2.5s passes")

    # Regression (2026-09-11 incident): the model narrates 700 words in
    # 403.2s (~104 wpm) and emits EOS after the FULL text. That is 16.8s
    # below the cap and must NOT be flagged as truncation.
    expect_no_error(
        lambda: check_chunk_audio(403.2, 6048, 420.0),
        "full text at 104 wpm (regression)",
    )
    print("  OK: 403.2s full-text audio (104 wpm) passes checks")

    # Too short: 50s for ~420s expected -> retryable (non-deterministic)
    err = expect_budget_error(
        lambda: check_chunk_audio(50.0, 6048, 420.0),
        "too short",
        deterministic=False,
    )
    assert "retrying" in str(err)
    print(f"  OK: too-short audio flagged (retryable): {err}")

    # Fast narrator: 280s for ~420s expected (ratio 0.67, well above the floor)
    expect_no_error(lambda: check_chunk_audio(280.0, 6048, 420.0), "fast narrator")
    print("  OK: fast narrator (280s vs 420s expected) passes")

    # Regression (2026-09-11 run, 150-word chunks): the model narrates short
    # chunks at ~2x the 85-wpm budgeting rate, so COMPLETE audio measures
    # 0.34-0.53 of the estimate. The old 0.40 floor dropped these good chunks
    # (13 of 177). The fastest complete chunk observed (ratio 0.34) must pass:
    expect_no_error(
        lambda: check_chunk_audio(36.0, 1589, 105.9),
        "fast-but-complete 150-word chunk (ratio 0.34)",
    )
    print("  OK: fast-but-complete audio (36s vs 106s expected) passes")

    # The old threshold boundary (ratio 0.40) also passes now:
    expect_no_error(
        lambda: check_chunk_audio(42.4, 1589, 105.9),
        "old 0.40 boundary",
    )
    print("  OK: old 0.40 boundary (42.4s vs 106s expected) passes")

    # Catastrophic truncation (audio missing >70% of the text) is still
    # flagged as retryable:
    err = expect_budget_error(
        lambda: check_chunk_audio(20.0, 1589, 105.9),
        "catastrophic truncation",
        deterministic=False,
    )
    assert "retrying" in str(err)
    print(f"  OK: catastrophic truncation (20s vs 106s expected) flagged: {err}")

    # ------------------------------------------------------------------
    # Audio quality guard (long-context degradation detection)
    # ------------------------------------------------------------------

    # 9. Clean speech-like audio passes
    expect_no_error(
        lambda: check_chunk_audio_quality(make_clean(30), SR),
        "clean audio",
    )
    print("  OK: clean 30s audio passes quality guard")

    # 10. Noise tail (20s clean + 10s noise) -> retryable error with onset
    err = expect_budget_error(
        lambda: check_chunk_audio_quality(
            np.concatenate([make_clean(20), make_noise(10)]), SR
        ),
        "noise tail",
        deterministic=False,
    )
    assert "degraded" in str(err)
    assert "from ~20s" in str(err), f"onset not reported correctly: {err}"
    print(f"  OK: noise tail detected with onset: {err}")

    # 11. Whole-chunk noise -> retryable error
    expect_budget_error(
        lambda: check_chunk_audio_quality(make_noise(30), SR),
        "whole noise",
        deterministic=False,
    )
    print("  OK: whole-chunk noise detected")

    # 12. Silence must NOT trigger the guard (flatness of silence is ~1.0)
    expect_no_error(
        lambda: check_chunk_audio_quality(np.zeros(30 * SR), SR),
        "silence",
    )
    print("  OK: silence passes quality guard")

    # 13. Clean audio with a trailing pause passes (silence excluded)
    expect_no_error(
        lambda: check_chunk_audio_quality(
            np.concatenate([make_clean(25), np.zeros(5 * SR)]), SR
        ),
        "clean + trailing pause",
    )
    print("  OK: clean audio with trailing pause passes")

    # 14. Very short audio (< 20s) is skipped, not flagged
    expect_no_error(
        lambda: check_chunk_audio_quality(make_noise(10), SR),
        "short audio skipped",
    )
    print("  OK: <20s audio skipped by quality guard")

    print("\nAll chunk budget and quality guard tests passed.")


if __name__ == "__main__":
    main()