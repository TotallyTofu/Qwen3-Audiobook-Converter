#!/usr/bin/env python3
"""
Qwen-Based Audiobook Converter
Converts PDFs, EPUBs, DOCX, DOC, TXT files into audiobooks using faster-qwen3-tts

Uses faster-qwen3-tts with CUDA graph optimization for 5-10x speedup vs baseline.
Requires: NVIDIA GPU with CUDA, PyTorch >= 2.5.1
"""

import os
import math
import shutil
import logging
import hashlib
import argparse
import posixpath
import statistics
import subprocess
import tempfile
import warnings
import wave
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from urllib.parse import unquote
import time
import sys
import zipfile
import xml.etree.ElementTree as ET
from html import unescape
import re
from datetime import datetime

# faster-qwen3-tts imports
import numpy as np
import torch
from faster_qwen3_tts import FasterQwen3TTS

# Audio processing
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from pydub.silence import detect_leading_silence

# Format-specific imports (loaded on demand)
import PyPDF2
import ebooklib
from ebooklib import epub

# Fix Windows console encoding for emoji/unicode
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# =============================================================================
# LOAD CONFIG FROM config.py (with fallback defaults)
# =============================================================================

try:
    import config as user_config
except ImportError:
    user_config = None


def _cfg(attr, default):
    """Get a config value from config.py, falling back to default if not found."""
    if user_config is not None and hasattr(user_config, attr):
        return getattr(user_config, attr)
    return default


# =============================================================================
# HARDCODED CONFIGURATION (defaults used when config.py is missing)
# =============================================================================

MAX_RETRIES = 3
# When a chunk still fails after MAX_RETRIES, it is split in half and each half
# is generated on its own (recursively, up to this depth) instead of being
# dropped from the audiobook.
MAX_SPLIT_DEPTH = 2

# faster-qwen3-tts Model Configuration
CUSTOM_VOICE_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
VOICE_CLONE_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
VOICE_DESIGN_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
MODEL_IDS = {
    "custom_voice": CUSTOM_VOICE_MODEL_ID,
    "voice_clone": VOICE_CLONE_MODEL_ID,
    "voice_design": VOICE_DESIGN_MODEL_ID,
}
DEVICE = _cfg("FASTER_QWEN_DEVICE", "cuda")
DTYPE = torch.bfloat16 if _cfg("FASTER_QWEN_DTYPE", "bfloat16") == "bfloat16" else torch.float16

# Hardcoded Voice Settings (CustomVoice mode)
CUSTOM_VOICE_SPEAKER = _cfg("CUSTOM_VOICE_SPEAKER", "Ryan")
CUSTOM_VOICE_LANGUAGE = _cfg("CUSTOM_VOICE_LANGUAGE", "English")
CUSTOM_VOICE_INSTRUCT = _cfg(
    "CUSTOM_VOICE_INSTRUCT",
    "Narrate like a professional audiobook reader: calm and clear, at a slow, relaxed pace, "
    "with natural pauses between sentences.",
)
CUSTOM_VOICE_SEED = _cfg("CUSTOM_VOICE_SEED", -1)

# Voice Clone Settings (loaded from config.py with fallbacks)
VOICE_CLONE_LANGUAGE = _cfg("VOICE_CLONE_LANGUAGE", "Auto")
VOICE_CLONE_USE_XVECTOR_ONLY = _cfg("VOICE_CLONE_USE_XVECTOR_ONLY", False)
VOICE_CLONE_SEED = _cfg("VOICE_CLONE_SEED", -1)
VOICE_CLONE_APPEND_SILENCE = _cfg("VOICE_CLONE_APPEND_SILENCE", True)

# Voice Design Settings
VOICE_DESIGN_LANGUAGE = _cfg("VOICE_DESIGN_LANGUAGE", "Auto")
VOICE_DESIGN_DESCRIPTION = _cfg(
    "VOICE_DESIGN_DESCRIPTION",
    "A clear, calm, professional audiobook narrator reading at a slow, relaxed pace.",
)
VOICE_DESIGN_SEED = _cfg("VOICE_DESIGN_SEED", -1)

# Processing Settings (loaded from config.py with fallbacks)
BOOKS_FOLDER = _cfg("BOOKS_FOLDER", "book_to_convert")
AUDIOBOOKS_FOLDER = _cfg("AUDIOBOOKS_FOLDER", "audiobooks")
CHUNK_SIZE_WORDS = _cfg("CHUNK_SIZE_WORDS", 150)
AUDIO_FORMAT = _cfg("AUDIO_FORMAT", "mp3")
AUDIO_BITRATE = _cfg("AUDIO_BITRATE", "128k")

# Silence inserted after each chunk in the final audiobook, by the kind of
# boundary the chunk ends on. The model itself leaves only ~120 ms of silence
# at the end of a chunk, versus ~340 ms between sentences inside a chunk.
SENTENCE_PAUSE_MS = _cfg("SENTENCE_PAUSE_MS", 250)
PARAGRAPH_PAUSE_MS = _cfg("PARAGRAPH_PAUSE_MS", 1000)
CHAPTER_PAUSE_MS = _cfg("CHAPTER_PAUSE_MS", 2500)

# Streaming settings (loaded from config.py with fallbacks). Streaming only
# lowers time-to-first-audio, which an offline converter does not use.
STREAMING_ENABLED = _cfg("STREAMING_ENABLED", False)
STREAMING_CHUNK_SIZE = _cfg("STREAMING_CHUNK_SIZE", 8)  # Steps per chunk (~640ms audio per chunk)

# TTS token budget settings (chunk truncation prevention)
# faster-qwen3-tts caps each generation at max_seq_len TOTAL tokens (prompt +
# audio). If the chunk's speech needs more audio tokens than that, the tail of
# the chunk is silently dropped. These knobs size the per-chunk budget and
# detect truncation after generation. See config.py for details.
FASTER_QWEN_MAX_SEQ_LEN = _cfg("FASTER_QWEN_MAX_SEQ_LEN", 4096)
TTS_WORDS_PER_MIN = _cfg("TTS_WORDS_PER_MIN", 85)
TTS_CODEC_TOKENS_PER_SEC = _cfg("TTS_CODEC_TOKENS_PER_SEC", 12.5)  # 1920 samples per token at 24 kHz
TTS_BUDGET_SAFETY_FACTOR = _cfg("TTS_BUDGET_SAFETY_FACTOR", 1.2)
TTS_MIN_MAX_NEW_TOKENS = _cfg("TTS_MIN_MAX_NEW_TOKENS", 256)
TTS_VOICE_CLONE_REF_MARGIN = _cfg("TTS_VOICE_CLONE_REF_MARGIN", 512)
TTS_TOO_SHORT_RATIO = _cfg("TTS_TOO_SHORT_RATIO", 0.30)
TTS_TOO_LONG_RATIO = _cfg("TTS_TOO_LONG_RATIO", 1.0)
TTS_TOO_LONG_SLACK_SEC = _cfg("TTS_TOO_LONG_SLACK_SEC", 3.0)

# Text, cache and logging options
CLEAN_PAGE_NUMBERS = _cfg("CLEAN_PAGE_NUMBERS", True)
ENABLE_CACHING = _cfg("ENABLE_CACHING", True)
LOG_LEVEL = _cfg("LOG_LEVEL", "INFO")
LOG_TO_FILE = _cfg("LOG_TO_FILE", True)
LOG_TO_CONSOLE = _cfg("LOG_TO_CONSOLE", True)

# =============================================================================
# OPTIONAL IMPORTS WITH FALLBACKS
# =============================================================================

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    import docx2txt
    DOC_AVAILABLE = True
except ImportError:
    DOC_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False


# =============================================================================
# TTS TOKEN BUDGET HELPERS (chunk truncation prevention)
# =============================================================================

class ChunkBudgetError(RuntimeError):
    """A chunk cannot be fully generated within the token budget.

    `deterministic` is True when retrying with the same settings cannot
    succeed (token cap hit, or text too long for the configured
    max_seq_len). When False, a retry with a fresh seed may succeed.
    """

    def __init__(self, message: str, deterministic: bool = True):
        super().__init__(message)
        self.deterministic = deterministic


def estimate_prefill_tokens(text: str, voice_mode: str = "custom_voice") -> int:
    """Conservative estimate of the prompt (prefill) token count for a chunk.

    The talker prefill holds the entire chunk text (~1.3-1.5 tokens per
    English word) plus system/speaker/instruct tokens. Voice-clone ICL mode
    additionally places the reference audio's codec tokens in context.
    """
    words = len(text.split())
    prefill = int(words * 1.5) + 300
    if voice_mode == "voice_clone":
        prefill += TTS_VOICE_CLONE_REF_MARGIN
    return prefill


def estimate_spoken_words(text: str) -> int:
    """Word count adjusted for numbers, which take several words to say.

    "1923" is spoken as "nineteen twenty-three", so a token counts as one
    word per digit it contains (at least one). Bibliographies and
    chronologies otherwise look far shorter than they sound.
    """
    return sum(max(1, sum(ch.isdigit() for ch in token)) for token in text.split())


def plan_chunk_generation(text: str, max_seq_len: int, voice_mode: str = "custom_voice") -> Tuple[int, float]:
    """Compute the max_new_tokens budget for one chunk.

    Returns (max_new_tokens, expected_seconds). Raises ChunkBudgetError
    (deterministic) when the chunk cannot fit in `max_seq_len`, with a
    message that states the exact numbers and the two remedies.
    """
    words = len(text.split())
    if words == 0:
        raise ChunkBudgetError("Chunk text is empty; nothing to generate.", deterministic=True)

    expected_sec = estimate_spoken_words(text) / TTS_WORDS_PER_MIN * 60.0
    needed = math.ceil(expected_sec * TTS_CODEC_TOKENS_PER_SEC * TTS_BUDGET_SAFETY_FACTOR)
    needed = max(needed, TTS_MIN_MAX_NEW_TOKENS)

    prefill_est = estimate_prefill_tokens(text, voice_mode)
    budget = max_seq_len - 1 - prefill_est

    if budget < TTS_MIN_MAX_NEW_TOKENS:
        raise ChunkBudgetError(
            f"Chunk too long: ~{words} words need ~{prefill_est} prompt tokens, leaving only "
            f"{budget} audio tokens in max_seq_len={max_seq_len} (need >= {TTS_MIN_MAX_NEW_TOKENS}). "
            f"Lower CHUNK_SIZE_WORDS or raise FASTER_QWEN_MAX_SEQ_LEN.",
            deterministic=True,
        )

    if needed > budget:
        raise ChunkBudgetError(
            f"Chunk would be truncated: ~{words} words need ~{needed} audio tokens "
            f"(expected speech ~{expected_sec:.0f}s x {TTS_BUDGET_SAFETY_FACTOR} safety), but only "
            f"{budget} fit in max_seq_len={max_seq_len} after ~{prefill_est} prompt tokens. "
            f"Raise FASTER_QWEN_MAX_SEQ_LEN to >= {prefill_est + needed + 1} or lower CHUNK_SIZE_WORDS.",
            deterministic=True,
        )

    return needed, expected_sec


def check_chunk_audio(actual_sec: float, max_new_tokens: int, expected_sec: float) -> None:
    """Detect truncated, runaway, or prematurely ended generations.

    - Cap hit: generation stopped within 3s of the max_new_tokens cap
      (max_new_tokens/12 s) instead of emitting EOS. The budget is sized at
      85 wpm x 1.2 while the model narrates at ~150-250 wpm, so reaching
      the cap almost always means the model failed to stop (runaway
      babble), not that the text needed more time. That is a sampling
      accident, so it is retryable.
    - Too long: audio well past the expected length at the slow budgeting
      rate (TTS_TOO_LONG_RATIO x expected + TTS_TOO_LONG_SLACK_SEC) but
      still under the cap, e.g. a runaway that eventually stopped
      (2026-09-11: chunk 36 ran 113.9s on its first attempt, 48.2s on the
      retry). Retryable.
    - Too short: audio is far below the expected speech length, likely a
      premature EOS (retryable). The floor is TTS_TOO_SHORT_RATIO (0.30),
      calibrated 2026-09-11: the model narrates 150-word chunks at ~2x the
      85-wpm budgeting rate, so COMPLETE audio measures 0.34-0.53 of the
      estimate. A 0.40 floor sat inside that band and dropped good chunks;
      0.30 passes all observed complete audio while still flagging audio
      missing >70% of the text.
    """
    cap_sec = max_new_tokens / TTS_CODEC_TOKENS_PER_SEC
    if actual_sec >= cap_sec - 3.0:
        raise ChunkBudgetError(
            f"Generation hit the token cap: {actual_sec:.1f}s of audio at the "
            f"{max_new_tokens}-token budget (~{cap_sec:.0f}s) for ~{expected_sec:.0f}s of speech. "
            f"The model most likely failed to stop; retrying. If this happens on most chunks, "
            f"the narration is slower than TTS_WORDS_PER_MIN assumes: lower it or raise "
            f"FASTER_QWEN_MAX_SEQ_LEN.",
            deterministic=False,
        )
    too_long_sec = expected_sec * TTS_TOO_LONG_RATIO + TTS_TOO_LONG_SLACK_SEC
    if actual_sec > too_long_sec:
        raise ChunkBudgetError(
            f"Audio is far longer than expected: {actual_sec:.1f}s for ~{expected_sec:.0f}s of speech "
            f"(limit {too_long_sec:.0f}s). Likely a runaway generation (babble or repetition); retrying.",
            deterministic=False,
        )
    if actual_sec < expected_sec * TTS_TOO_SHORT_RATIO:
        raise ChunkBudgetError(
            f"Audio is far shorter than expected: {actual_sec:.1f}s for ~{expected_sec:.0f}s of speech "
            f"(floor {TTS_TOO_SHORT_RATIO:.0%}). Likely premature EOS; retrying.",
            deterministic=False,
        )


def check_chunk_audio_quality(audio: np.ndarray, sample_rate: int) -> None:
    """Detect long-context audio degradation (garbled, noise-like output).

    The Qwen3-TTS talker is trained on short utterances. When a single
    generation runs too long, it drifts out of distribution and starts
    emitting noise-like codec tokens. That shows up as a sharp rise in
    spectral flatness (noise-likeness): clean speech measures ~0.05-0.15
    per second, degraded audio sits at ~0.3-0.7. This check would have
    caught the 2026-09-11 Labyrinths chunk that garbled from ~184s on.

    Raises a retryable ChunkBudgetError when the tail is clearly more
    noise-like than the head, or when the whole chunk is noise-like.
    Silent frames are excluded so trailing pauses don't trigger it.
    """
    n_frames = len(audio) // sample_rate
    if n_frames < 20:
        return  # too short to be meaningful

    window = np.hanning(sample_rate)
    seconds = []
    for i in range(n_frames):
        seg = audio[i * sample_rate:(i + 1) * sample_rate]
        rms = float(np.sqrt(np.mean(seg ** 2)))
        if rms < 0.01:
            continue  # silence: flatness is meaningless there
        spec = np.abs(np.fft.rfft(seg * window))[1:] + 1e-12
        flat = float(np.exp(np.mean(np.log(spec))) / np.mean(spec))
        seconds.append((i, flat))

    if len(seconds) < 10:
        return  # mostly silence; nothing to check

    flat = np.array([f for _, f in seconds])
    idx = np.array([i for i, _ in seconds])
    k = max(2, len(flat) // 5)
    head = float(np.median(flat[:k]))
    tail = float(np.median(flat[-k:]))
    whole = float(np.median(flat))

    if whole > 0.45:
        raise ChunkBudgetError(
            f"Audio quality degraded across the whole chunk (median spectral "
            f"flatness {whole:.2f} is noise-like). Lower CHUNK_SIZE_WORDS so "
            f"each generation stays short.",
            deterministic=False,
        )

    if tail > 0.25 and tail > 2.0 * max(head, 0.05):
        # Locate the approximate onset: first run of 3+ consecutive
        # degraded seconds (clean speech never sustains flatness > 0.3).
        onset = None
        run_start = None
        for i, f in zip(idx, flat):
            if f > 0.3:
                if run_start is None:
                    run_start = int(i)
                if int(i) - run_start >= 2 and onset is None:
                    onset = run_start
            else:
                run_start = None
        onset_txt = f" from ~{onset}s" if onset is not None else ""
        raise ChunkBudgetError(
            f"Audio quality degraded{onset_txt}: tail spectral flatness "
            f"{tail:.2f} vs head {head:.2f}. Likely long-context drift; "
            f"retrying with a fresh seed. If it persists, lower CHUNK_SIZE_WORDS.",
            deterministic=False,
        )


# =============================================================================
# TEXT STRUCTURE HELPERS (paragraphs, sentences, chunks)
# =============================================================================
#
# Extracted text keeps its structure as blank lines so the chunker can place
# pauses: paragraphs are separated by "\n\n" and chapters/sections (EPUB spine
# documents, or 2+ blank lines in plain text) by "\n\n\n".

PARAGRAPH_SEP = "\n\n"
SECTION_SEP = "\n\n\n"

# Block-level HTML elements: each one starts a new paragraph. Without this,
# get_text() fused adjacent blocks ("Pliny.On the fourteenth...", which the
# model read as "Pliny on February 14th. I February received...").
HTML_BLOCK_TAGS = [
    "address", "article", "aside", "blockquote", "dd", "div", "dl", "dt", "figcaption",
    "figure", "footer", "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr", "li", "nav",
    "ol", "p", "pre", "section", "table", "td", "th", "tr", "ul",
]

# Sentence end: terminal punctuation, optional closing quotes/brackets, whitespace.
_SENTENCE_END_RE = re.compile(r'[.!?…]+["\'”’)\]]*\s+')
_TRAILING_PUNCT_RE = re.compile(r'[.!?…"\'”’)\]]+$')
_CLAUSE_BREAK_RE = re.compile(r'(?<=[,;:—])\s+')
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "st", "jr", "sr", "prof", "rev", "gen", "col", "capt", "lt",
    "sgt", "gov", "sen", "vs", "etc", "vol", "pp", "ch", "fig", "cf", "mt", "e.g", "i.e",
}


@dataclass
class Chunk:
    """One TTS generation unit and the silence that follows it in the audiobook."""
    text: str
    pause_ms: int


def normalize_paragraphs(text: str) -> str:
    """Collapse whitespace inside paragraphs; keep paragraphs separated by one blank line."""
    text = text.replace("­", "").replace("​", "").replace("﻿", "")
    paragraphs = (" ".join(p.split()) for p in re.split(r"\n\s*\n", text))
    return PARAGRAPH_SEP.join(p for p in paragraphs if p)


def merge_broken_paragraphs(paragraphs: List[str]) -> List[str]:
    """Re-join sentences that a hard line break split into two "paragraphs".

    Some EPUBs (copyright pages, scanned text) put each visual line in its
    own block element. A block that ends without punctuation followed by
    one starting in lowercase is the same sentence ("may be" / "reproduced
    in any form"), so it must not get a period and a pause in between.
    """
    merged: List[str] = []
    for paragraph in paragraphs:
        if merged and merged[-1][-1].isalnum() and paragraph[0].islower():
            merged[-1] += " " + paragraph
        else:
            merged.append(paragraph)
    return merged


def ensure_terminal_punctuation(paragraph: str) -> str:
    """End headings, titles and list items with a period so the model pauses after them."""
    if paragraph and paragraph[-1].isalnum():
        return paragraph + "."
    return paragraph


def split_sentences(text: str) -> List[str]:
    """Split on sentence-ending punctuation, including endings inside closing quotes.

    The old splitter required whitespace right after the punctuation, so
    every `."` / `.”` ending was missed. Common abbreviations ("Mr.") and
    initials ("J. E.") are not treated as sentence ends.
    """
    sentences = []
    start = 0
    for m in _SENTENCE_END_RE.finditer(text):
        candidate = text[start:m.end()].strip()
        if m.group().rstrip() == ".":
            last_word = _TRAILING_PUNCT_RE.sub("", candidate.split()[-1]).lower()
            if last_word in _ABBREVIATIONS or (len(last_word) == 1 and last_word.isalpha()):
                continue
        sentences.append(candidate)
        start = m.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def split_long_sentence(sentence: str, max_words: int) -> List[str]:
    """Split a sentence longer than max_words at clause punctuation, keeping the punctuation.

    A clause that is itself longer than max_words is cut at word boundaries,
    so no piece can exceed the token budget.
    """
    if len(sentence.split()) <= max_words:
        return [sentence]
    pieces: List[str] = []
    cur: List[str] = []
    for clause in _CLAUSE_BREAK_RE.split(sentence):
        words = clause.split()
        while len(words) > max_words:
            if cur:
                pieces.append(" ".join(cur))
                cur = []
            pieces.append(" ".join(words[:max_words]))
            words = words[max_words:]
        if cur and len(cur) + len(words) > max_words:
            pieces.append(" ".join(cur))
            cur = []
        cur += words
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def split_text_in_half(text: str) -> List[str]:
    """Split text into two pieces of similar length at the best available boundary.

    Prefers a sentence boundary, then a clause boundary, then any word, and
    only accepts a boundary that leaves each side at least a quarter of the
    words. Returns [text] when the text is a single word.
    """
    total = len(text.split())
    if total < 2:
        return [text]
    levels = [split_sentences(text), _CLAUSE_BREAK_RE.split(text), text.split()]
    for level, units in enumerate(levels):
        units = [u for u in units if u.strip()]
        if len(units) < 2:
            continue
        best_cut, best_diff, left_words = 1, None, 0
        for i in range(1, len(units)):
            left_words += len(units[i - 1].split())
            diff = abs(total - 2 * left_words)
            if best_diff is None or diff < best_diff:
                best_cut, best_diff = i, diff
        left, right = " ".join(units[:best_cut]), " ".join(units[best_cut:])
        if min(len(left.split()), len(right.split())) >= total / 4 or level == len(levels) - 1:
            return [left, right]
    return [text]


# =============================================================================
# FASTER QWEN BACKEND
# =============================================================================

class FasterQwenBackend:
    """Backend using faster-qwen3-tts with CUDA graph optimization"""

    def __init__(self, device: str = DEVICE, dtype: torch.dtype = DTYPE,
                 speaker: str = CUSTOM_VOICE_SPEAKER, language: str = CUSTOM_VOICE_LANGUAGE,
                 instruct: str = CUSTOM_VOICE_INSTRUCT, voice_description: str = VOICE_DESIGN_DESCRIPTION):
        self.device = device
        self.dtype = dtype
        self.model = None
        self.speaker = speaker
        self.language = language
        self.instruct = instruct
        self.voice_description = voice_description
        self.voice_clone_ref_audio: Optional[str] = None
        self.voice_clone_ref_text: str = ""
        self.speaker_embedding_path: Optional[str] = None
        self.speaker_embedding: Optional[torch.Tensor] = None
        self.sample_rate: int = 24000

    def initialize(self, voice_mode: str, voice_clone_ref_audio: Optional[str] = None, voice_clone_ref_text: str = "") -> None:
        """Load the TTS model and set up voice configuration"""
        print("[INFO] Loading faster-qwen3-tts model...")
        print("[INFO] This may take a few minutes on first run (downloads model from HuggingFace)")

        model_id = MODEL_IDS[voice_mode]

        try:
            self.model = FasterQwen3TTS.from_pretrained(
                model_id,
                device=self.device,
                dtype=self.dtype,
                max_seq_len=FASTER_QWEN_MAX_SEQ_LEN,
            )
            self.sample_rate = self.model.sample_rate
            print(f"[OK] Model loaded successfully on {self.device}: {model_id}")
            print(f"[OK] Sample rate: {self.sample_rate} Hz")
            print(f"[OK] Token budget: max_seq_len={FASTER_QWEN_MAX_SEQ_LEN} "
                  f"(CHUNK_SIZE_WORDS={CHUNK_SIZE_WORDS})")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            print("Make sure you have:")
            print("1. NVIDIA GPU with CUDA")
            print("2. PyTorch >= 2.5.1 with CUDA support")
            print("3. Sufficient VRAM (at least 6GB recommended)")
            sys.exit(1)

        if voice_mode == "voice_clone" and voice_clone_ref_audio:
            self.voice_clone_ref_audio = voice_clone_ref_audio
            if not Path(voice_clone_ref_audio).exists():
                print(f"[ERROR] Reference audio not found: {voice_clone_ref_audio}")
                sys.exit(1)

        if voice_mode == "voice_clone" and voice_clone_ref_text:
            self.voice_clone_ref_text = voice_clone_ref_text

    @staticmethod
    def _first_waveform(result) -> np.ndarray:
        """Non-streaming calls return (list_of_waveforms, sample_rate); take the first waveform."""
        audio_list, _sr = result
        return audio_list[0]

    @staticmethod
    def _collect_stream(result) -> np.ndarray:
        """Streaming calls yield (audio_chunk, sample_rate, timing); join the chunks."""
        chunks = [audio_chunk for audio_chunk, _sr, _timing in result]
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks)

    def _voice_clone_kwargs(self) -> Dict[str, Any]:
        """Reference arguments for voice cloning: ICL when ref_text is set, else x-vector.

        No style instruction is sent: the Base model is not trained to follow
        one, and in a 2026-10-09 A/B test (same voice, Labyrinths chunks 6-15)
        it doubled chunk-to-chunk timbre variation while slowing the pace only
        ~4%. The clone takes its pace from the reference clip instead.
        """
        if not self.voice_clone_ref_audio:
            raise ValueError("Reference audio not set for voice cloning")
        if self.voice_clone_ref_text:
            kwargs = {
                "ref_audio": self.voice_clone_ref_audio,
                "ref_text": self.voice_clone_ref_text,
                "xvec_only": False,
                "append_silence": VOICE_CLONE_APPEND_SILENCE,
            }
        elif self.speaker_embedding is not None:
            kwargs = {
                "ref_audio": self.voice_clone_ref_audio,
                "ref_text": "",
                "xvec_only": True,
                "voice_clone_prompt": {"ref_spk_embedding": [self.speaker_embedding]},
            }
        elif VOICE_CLONE_USE_XVECTOR_ONLY:
            kwargs = {"ref_audio": self.voice_clone_ref_audio, "ref_text": "", "xvec_only": True}
        else:
            raise ValueError("Voice clone requires either ref_text (ICL mode) or speaker embedding (xvector mode)")
        return kwargs

    def generate_custom_voice(self, text: str, max_new_tokens: Optional[int] = None) -> np.ndarray:
        """Generate audio using CustomVoice mode with pre-built speaker"""
        gen_kwargs = {"max_new_tokens": max_new_tokens} if max_new_tokens is not None else {}
        return self._first_waveform(self.model.generate_custom_voice(
            text=text,
            language=self.language,
            speaker=self.speaker,
            instruct=self.instruct,
            **gen_kwargs,
        ))

    def generate_custom_voice_streaming(self, text: str, max_new_tokens: Optional[int] = None) -> np.ndarray:
        """Generate audio using CustomVoice mode with streaming for faster TTFB"""
        gen_kwargs = {"max_new_tokens": max_new_tokens} if max_new_tokens is not None else {}
        return self._collect_stream(self.model.generate_custom_voice_streaming(
            text=text,
            language=self.language,
            speaker=self.speaker,
            instruct=self.instruct,
            chunk_size=STREAMING_CHUNK_SIZE,
            **gen_kwargs,
        ))

    def generate_voice_clone(self, text: str, max_new_tokens: Optional[int] = None) -> np.ndarray:
        """Generate audio using Voice Clone mode with ICL or xvector"""
        gen_kwargs = {"max_new_tokens": max_new_tokens} if max_new_tokens is not None else {}
        return self._first_waveform(self.model.generate_voice_clone(
            text=text,
            language=self.language,
            **self._voice_clone_kwargs(),
            **gen_kwargs,
        ))

    def generate_voice_clone_streaming(self, text: str, max_new_tokens: Optional[int] = None) -> np.ndarray:
        """Generate audio using Voice Clone mode with streaming"""
        gen_kwargs = {"max_new_tokens": max_new_tokens} if max_new_tokens is not None else {}
        return self._collect_stream(self.model.generate_voice_clone_streaming(
            text=text,
            language=self.language,
            chunk_size=STREAMING_CHUNK_SIZE,
            **self._voice_clone_kwargs(),
            **gen_kwargs,
        ))

    def generate_voice_design(self, text: str, max_new_tokens: Optional[int] = None) -> np.ndarray:
        """Generate audio using VoiceDesign mode (instruction-based)"""
        gen_kwargs = {"max_new_tokens": max_new_tokens} if max_new_tokens is not None else {}
        return self._first_waveform(self.model.generate_voice_design(
            text=text,
            language=self.language,
            instruct=self.voice_description,
            **gen_kwargs,
        ))

    def generate_voice_design_streaming(self, text: str, max_new_tokens: Optional[int] = None) -> np.ndarray:
        """Generate audio using VoiceDesign mode with streaming"""
        gen_kwargs = {"max_new_tokens": max_new_tokens} if max_new_tokens is not None else {}
        return self._collect_stream(self.model.generate_voice_design_streaming(
            text=text,
            language=self.language,
            instruct=self.voice_description,
            chunk_size=STREAMING_CHUNK_SIZE,
            **gen_kwargs,
        ))

    def generate(self, voice_mode: str, text: str, max_new_tokens: Optional[int] = None) -> np.ndarray:
        """Generate one waveform for `text` in the given voice mode."""
        generators = {
            "custom_voice": (self.generate_custom_voice, self.generate_custom_voice_streaming),
            "voice_clone": (self.generate_voice_clone, self.generate_voice_clone_streaming),
            "voice_design": (self.generate_voice_design, self.generate_voice_design_streaming),
        }
        if voice_mode not in generators:
            raise ValueError(f"Unknown voice mode: {voice_mode}")
        generate_full, generate_streaming = generators[voice_mode]
        fn = generate_streaming if STREAMING_ENABLED else generate_full
        return np.asarray(fn(text, max_new_tokens), dtype=np.float32)

    def extract_speaker_embedding(self, audio_path: str) -> torch.Tensor:
        """Extract speaker embedding from reference audio for faster reuse"""
        print(f"[INFO] Extracting speaker embedding from {audio_path}...")
        try:
            prompt_items = self.model.model.create_voice_clone_prompt(
                ref_audio=audio_path,
                ref_text="",
                x_vector_only_mode=True,
            )
            spk_emb = prompt_items[0].ref_spk_embedding
            print("[OK] Speaker embedding extracted (4KB, 2048-dim)")
            return spk_emb
        except Exception as e:
            print(f"[WARNING] Could not extract speaker embedding: {e}")
            return None


# =============================================================================
# AUDIOBOOK CONVERTER
# =============================================================================

class QwenAudiobookConverter:
    """Audiobook converter using faster-qwen3-tts backend"""

    def __init__(self, voice_mode: str = "custom_voice", voice_clone_ref_audio: Optional[str] = None,
                 voice_clone_ref_text: Optional[str] = None, force_xvector: bool = False,
                 speaker: Optional[str] = None, language: Optional[str] = None,
                 instruct: Optional[str] = None, voice_description: Optional[str] = None):
        self.voice_mode = voice_mode
        self.voice_clone_ref_audio = voice_clone_ref_audio
        # Use passed ref_text first, then config.py fallback
        self.voice_clone_ref_text = voice_clone_ref_text if voice_clone_ref_text else ""
        # Voice settings: explicit arguments (web UI) override config.py
        self.speaker = speaker or CUSTOM_VOICE_SPEAKER
        self.language = language or {
            "custom_voice": CUSTOM_VOICE_LANGUAGE,
            "voice_clone": VOICE_CLONE_LANGUAGE,
            "voice_design": VOICE_DESIGN_LANGUAGE,
        }.get(voice_mode, "Auto")
        self.instruct = instruct if instruct is not None else CUSTOM_VOICE_INSTRUCT  # custom voice only
        self.voice_description = voice_description or VOICE_DESIGN_DESCRIPTION
        self.last_output_path: Optional[Path] = None
        self.last_missing_chunks: List[int] = []
        self.pace_wpm: List[float] = []  # measured narration pace per generated chunk
        self.setup_logging()
        self.setup_directories()
        self.validate_configuration()

        # Initialize faster-qwen3-tts backend
        self.backend = FasterQwenBackend(
            device=DEVICE, dtype=DTYPE, speaker=self.speaker, language=self.language,
            instruct=self.instruct, voice_description=self.voice_description,
        )
        try:
            init_ref_text = self.voice_clone_ref_text or _cfg("VOICE_CLONE_REF_TEXT", "")
            if init_ref_text and voice_mode == "voice_clone":
                self.backend.initialize(voice_mode, voice_clone_ref_audio, init_ref_text)
            else:
                self.backend.initialize(voice_mode, voice_clone_ref_audio)
            print("[OK] Backend initialized (faster-qwen3-tts)")
        except Exception as e:
            print(f"[FATAL] Failed to initialize backend: {e}")
            sys.exit(1)

        # If voice clone, determine mode and set up ref_text or xvector
        if voice_mode == "voice_clone" and voice_clone_ref_audio:
            # Step 1: Try to find ref_text from multiple sources
            caller_ref_text = voice_clone_ref_text or ""
            config_ref_text = _cfg("VOICE_CLONE_REF_TEXT", "")

            # Priority: caller text > config file
            available_ref_text = caller_ref_text if caller_ref_text else config_ref_text

            # Determine which mode we should use
            # ICL mode takes priority if ref_text is available
            has_ref_text = bool(available_ref_text)
            force_xvec_mode = force_xvector and not has_ref_text
            use_xvect_config = VOICE_CLONE_USE_XVECTOR_ONLY and not has_ref_text

            if has_ref_text:
                # ICL mode - ref_text provided, use it (overrides xvector checkbox)
                self.voice_clone_ref_text = available_ref_text
                self.backend.voice_clone_ref_text = available_ref_text
                print(f"[INFO] Voice clone ICL mode active (ref_text from {'caller' if caller_ref_text else 'config.py'})")
                self.logger.info(f"Voice clone: ICL mode, reference {Path(voice_clone_ref_audio).name}")
                self._log_reference_pace()
            elif force_xvec_mode or use_xvect_config:
                # XVector-only mode - no ref_text available or forced without ref_text
                self.voice_clone_info = Path(voice_clone_ref_audio).name
                emb = self.backend.extract_speaker_embedding(voice_clone_ref_audio)
                if emb is not None:
                    self.backend.speaker_embedding = emb
                print("[INFO] Using xvector-only voice clone mode (speaker embedding extracted)")
            else:
                # No ref_text and no force_xvector - fallback to xvector-only
                print("[WARNING] No ref_text provided for voice clone.")
                print("  Falling back to xvector-only mode (no transcription needed).")
                self.voice_clone_info = Path(voice_clone_ref_audio).name
                emb = self.backend.extract_speaker_embedding(voice_clone_ref_audio)
                if emb is not None:
                    self.backend.speaker_embedding = emb
                print("[INFO] Using xvector-only voice clone mode (speaker embedding extracted)")

    def _log_reference_pace(self) -> None:
        """Log the voice-clone reference clip's speaking rate.

        In ICL mode the model continues on from the reference clip, so the
        clone tends to narrate at a similar pace: a calmly read clip gives
        calmer narration. Compare with the "Narration pace" log lines.
        """
        try:
            clip = AudioSegment.from_file(self.voice_clone_ref_audio)
            lead = detect_leading_silence(clip, silence_threshold=-45.0)
            trail = detect_leading_silence(clip.reverse(), silence_threshold=-45.0)
            speech_sec = max(len(clip) - lead - trail, 1) / 1000
            words = len(self.voice_clone_ref_text.split())
            message = (f"Reference clip pace: {words} words in {speech_sec:.1f}s of speech = "
                       f"{words / speech_sec * 60:.0f} wpm")
            self.logger.info(message)
            print(f"[INFO] {message}")
        except Exception as e:
            self.logger.warning(f"Could not measure the reference clip's pace: {e}")

    def setup_logging(self):
        """Setup logging configuration"""
        handlers: List[logging.Handler] = []
        if LOG_TO_FILE:
            Path("logs").mkdir(exist_ok=True)
            handlers.append(logging.FileHandler(
                f"logs/audiobook_{datetime.now().strftime('%Y%m%d')}.log", encoding="utf-8"
            ))
        if LOG_TO_CONSOLE:
            handlers.append(logging.StreamHandler(sys.stdout))
        logging.basicConfig(
            level=getattr(logging, str(LOG_LEVEL).upper(), logging.INFO),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=handlers or [logging.NullHandler()],
        )
        self.logger = logging.getLogger(__name__)

    def setup_directories(self):
        """Create necessary directories"""
        directories = [BOOKS_FOLDER, AUDIOBOOKS_FOLDER, "chunks", "cache/audio_chunks", "logs"]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def validate_configuration(self):
        """Validate configuration settings"""
        if self.voice_mode == "voice_clone":
            if not self.voice_clone_ref_audio:
                print("[ERROR] Configuration Error!")
                print("Voice Clone mode requires a reference audio file.")
                print("Use --voice-sample <path> to specify the reference audio.")
                sys.exit(1)

            if not Path(self.voice_clone_ref_audio).exists():
                print("[ERROR] Configuration Error!")
                print(f"Reference audio file not found: {self.voice_clone_ref_audio}")
                sys.exit(1)

    def get_cache_path(self, text: str) -> Path:
        """Get cache path for text chunk

        Everything that changes the generated audio is part of the key (voice,
        language, style instruction, model, budget), so changing a setting
        never replays audio generated under the old one.
        """
        if self.voice_mode == "custom_voice":
            voice = f"{self.speaker}|{self.language}|{self.instruct}"
        elif self.voice_mode == "voice_clone":
            # The trailing empty field once held a style instruction; kept so cached chunks stay valid
            voice = f"{self._ref_audio_fingerprint()}|{self.language}|{self.voice_clone_ref_text}|"
        else:
            voice = f"{self.language}|{self.voice_description}"
        content = (
            f"{text}|{self.voice_mode}|{MODEL_IDS.get(self.voice_mode)}|{voice}|"
            f"v3_{FASTER_QWEN_MAX_SEQ_LEN}_{TTS_WORDS_PER_MIN}_{TTS_BUDGET_SAFETY_FACTOR}"
        )
        hash_obj = hashlib.md5(content.encode())
        return Path("cache/audio_chunks") / f"{hash_obj.hexdigest()}.wav"

    def _ref_audio_fingerprint(self) -> str:
        """Content hash of the voice-clone reference audio (computed once)."""
        if not hasattr(self, "_ref_audio_hash"):
            self._ref_audio_hash = ""
            if self.voice_clone_ref_audio and Path(self.voice_clone_ref_audio).exists():
                self._ref_audio_hash = hashlib.md5(Path(self.voice_clone_ref_audio).read_bytes()).hexdigest()
        return self._ref_audio_hash

    def _seed_generation(self, text: str, attempt: int) -> None:
        """Apply the configured seed: reproducible per chunk, different on each retry."""
        seed = {
            "custom_voice": CUSTOM_VOICE_SEED,
            "voice_clone": VOICE_CLONE_SEED,
            "voice_design": VOICE_DESIGN_SEED,
        }.get(self.voice_mode, -1)
        if seed is not None and seed >= 0:
            torch.manual_seed((seed + zlib.crc32(text.encode("utf-8")) + attempt) % (2 ** 31))

    def synthesize_text(self, text: str, label: str, attempt: int = 0) -> np.ndarray:
        """Run one TTS generation for `text` and verify the result.

        Raises ChunkBudgetError when the text cannot fit the token budget or
        the audio fails a post-generation check.
        """
        # Size the token budget so the chunk cannot be silently truncated
        max_new_tokens, expected_sec = plan_chunk_generation(
            text, FASTER_QWEN_MAX_SEQ_LEN, self.voice_mode
        )
        words = len(text.split())
        self.logger.info(
            f"Chunk {label}: {words} words, max_new_tokens={max_new_tokens} (sized for a deliberately "
            f"slow {TTS_WORDS_PER_MIN} wpm read, ~{expected_sec:.0f}s; max_seq_len={FASTER_QWEN_MAX_SEQ_LEN})"
        )

        self._seed_generation(text, attempt)
        start_time = time.time()
        audio = self.backend.generate(self.voice_mode, text, max_new_tokens)

        elapsed = time.time() - start_time
        actual_sec = len(audio) / self.backend.sample_rate if len(audio) > 0 else 0
        rtf = elapsed / actual_sec if actual_sec > 0 else 0
        wpm = words / actual_sec * 60 if actual_sec > 0 else 0
        self.logger.info(
            f"Chunk {label}: {elapsed:.1f}s, RTF: {rtf:.2f}, "
            f"Audio length: {actual_sec:.1f}s, pace {wpm:.0f} wpm"
        )

        # Detect truncation, runaway generation, and premature EOS
        check_chunk_audio(actual_sec, max_new_tokens, expected_sec)

        # Detect long-context quality degradation (garbled tail)
        check_chunk_audio_quality(audio, self.backend.sample_rate)

        if words >= 30:  # short headings/fragments make noisy pace samples
            self.pace_wpm.append(wpm)
        return audio

    def log_pace(self) -> None:
        """Log the median narration pace of the chunks generated so far."""
        if self.pace_wpm:
            message = (f"Narration pace: median {statistics.median(self.pace_wpm):.0f} wpm "
                       f"over {len(self.pace_wpm)} chunks (audiobook narrators: ~150-160 wpm)")
            self.logger.info(message)
            print(f"[INFO] {message}")

    def synthesize_with_retry(self, text: str, label: str) -> Tuple[Optional[np.ndarray], bool]:
        """Generate `text` with up to MAX_RETRIES attempts.

        Returns (audio, worth_splitting). audio is None if every attempt
        failed; worth_splitting is True when the failures came from the
        text itself (budget or audio checks), where shorter text can help,
        rather than from errors such as running out of GPU memory.
        """
        worth_splitting = False
        for attempt in range(MAX_RETRIES):
            try:
                return self.synthesize_text(text, label, attempt), False
            except ChunkBudgetError as e:
                self.logger.error(f"Chunk {label} attempt {attempt + 1}: {e}")
                worth_splitting = True
                if e.deterministic:
                    break  # retrying the same text cannot help
            except Exception as e:
                self.logger.warning(f"Chunk {label} attempt {attempt + 1} error: {e}")
                worth_splitting = False
        return None, worth_splitting

    def synthesize_chunk(self, text: str, label: str, depth: int = 0) -> Optional[np.ndarray]:
        """Generate a chunk; if it keeps failing, generate its halves separately.

        Shorter generations are more reliable, so a chunk that fails all
        retries is split (up to MAX_SPLIT_DEPTH levels) rather than dropped.
        The pieces are joined with a sentence pause.
        """
        audio, worth_splitting = self.synthesize_with_retry(text, label)
        if audio is not None or not worth_splitting or depth >= MAX_SPLIT_DEPTH:
            return audio

        pieces = split_text_in_half(text)
        if len(pieces) < 2:
            return None
        self.logger.warning(f"Chunk {label}: retrying as {len(pieces)} shorter pieces")
        parts = []
        for i, piece in enumerate(pieces, 1):
            part = self.synthesize_chunk(piece, f"{label}.{i}", depth + 1)
            if part is None:
                return None
            parts.append(part)

        gap = np.zeros(int(self.backend.sample_rate * SENTENCE_PAUSE_MS / 1000), dtype=np.float32)
        joined = [parts[0]]
        for part in parts[1:]:
            joined += [gap, part]
        return np.concatenate(joined)

    def _write_wav(self, audio: np.ndarray, path: Path) -> None:
        """Write float audio in [-1, 1] as 16-bit mono WAV (clipped, so peaks cannot wrap around)."""
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.backend.sample_rate)
            wav.writeframes(pcm.tobytes())

    def process_chunk(self, chunk_num: int, text: str) -> bool:
        """Produce chunks/chunk_NNNN.wav for one chunk (from cache when available)."""
        output_path = Path("chunks") / f"chunk_{chunk_num:04d}.wav"
        cache_path = self.get_cache_path(text)
        if ENABLE_CACHING and cache_path.exists():
            shutil.copy2(cache_path, output_path)
            self.logger.debug(f"Using cached audio for chunk {chunk_num}")
            return True

        audio = self.synthesize_chunk(text, str(chunk_num))
        if audio is None:
            self.logger.error(f"Chunk {chunk_num} failed after all retries and splits")
            return False

        self._write_wav(audio, output_path)
        if ENABLE_CACHING:
            shutil.copy2(output_path, cache_path)
        self.logger.debug(f"Chunk {chunk_num} generated successfully")
        return True

    def extract_text_from_epub(self, file_path: Path) -> str:
        """Extract text from EPUB with fallback methods"""
        methods = [
            self._extract_epub_ebooklib,
            self._extract_epub_zipfile,
            self._extract_epub_manual
        ]

        for method in methods:
            try:
                text = method(file_path)
                if text and text.strip():
                    self.logger.info(f"EPUB extraction successful ({method.__name__}): {len(text)} characters")
                    return text
                self.logger.warning(f"EPUB method {method.__name__} found no text")
            except Exception as e:
                self.logger.warning(f"EPUB method {method.__name__} failed: {e}")
                continue

        raise RuntimeError("All EPUB extraction methods failed")

    def _extract_epub_ebooklib(self, file_path: Path) -> str:
        """Extract using ebooklib, in spine (reading) order. Each spine document is a section."""
        book = epub.read_epub(str(file_path))
        sections = []

        for item_id, _linear in book.spine:
            item = book.get_item_with_id(item_id)
            if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT:
                continue
            clean_text = self._clean_html(item.get_content())
            if clean_text.strip():
                sections.append(clean_text)

        return SECTION_SEP.join(sections)

    def _epub_document_paths(self, epub_zip: zipfile.ZipFile) -> List[str]:
        """HTML documents in spine (reading) order from the OPF, or archive order if unreadable."""
        names = epub_zip.namelist()
        try:
            container = ET.fromstring(epub_zip.read("META-INF/container.xml"))
            opf_path = container.find(".//{*}rootfile").get("full-path")
            opf = ET.fromstring(epub_zip.read(opf_path))
            base = posixpath.dirname(opf_path)
            manifest = {
                item.get("id"): posixpath.normpath(posixpath.join(base, unquote(item.get("href"))))
                for item in opf.find("{*}manifest")
            }
            spine = [manifest[ref.get("idref")] for ref in opf.find("{*}spine") if ref.get("idref") in manifest]
            ordered = [path for path in spine if path in names]
            if ordered:
                return ordered
        except Exception as e:
            self.logger.warning(f"Could not read the EPUB spine, using archive order: {e}")
        return [n for n in names if n.lower().endswith(('.html', '.xhtml', '.htm'))]

    def _extract_epub_zipfile(self, file_path: Path) -> str:
        """Extract using zipfile parsing, in spine order"""
        sections = []
        with zipfile.ZipFile(file_path, 'r') as epub_zip:
            for file_name in self._epub_document_paths(epub_zip):
                try:
                    clean_text = self._clean_html(epub_zip.read(file_name))
                    if clean_text.strip():
                        sections.append(clean_text)
                except Exception as e:
                    self.logger.warning(f"Skipping unreadable EPUB document {file_name}: {e}")
        return SECTION_SEP.join(sections)

    def _extract_epub_manual(self, file_path: Path) -> str:
        """Manual extraction fallback"""
        text_parts = []
        with zipfile.ZipFile(file_path, 'r') as epub_zip:
            for file_name in epub_zip.namelist():
                if not any(file_name.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.css', '.js']):
                    try:
                        content = epub_zip.read(file_name).decode('utf-8', errors='ignore')
                        if '<' in content and len(content.strip()) > 100:
                            clean_text = self._clean_html(content)
                            if clean_text:
                                text_parts.append(clean_text)
                    except Exception:
                        continue
        return SECTION_SEP.join(text_parts)

    def _clean_html(self, html_content) -> str:
        """Convert (X)HTML to text with one paragraph per block element.

        Accepts bytes (lets BeautifulSoup detect the declared encoding) or str.
        """
        if not html_content:
            return ""

        if BS4_AVAILABLE:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # XHTML parsed as HTML is fine for text extraction
                    soup = BeautifulSoup(html_content, 'html.parser')
                for tag in soup(["script", "style", "head"]):
                    tag.decompose()
                for br in soup.find_all("br"):
                    br.replace_with(" ")
                for tag in soup.find_all(HTML_BLOCK_TAGS):
                    tag.insert_before(PARAGRAPH_SEP)
                    tag.insert_after(PARAGRAPH_SEP)
                return normalize_paragraphs(soup.get_text())
            except Exception:
                pass

        # Fallback regex cleaning
        if isinstance(html_content, bytes):
            html_content = html_content.decode('utf-8', errors='ignore')
        html_content = re.sub(r'<(script|style|head)\b[^>]*>.*?</\1>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        html_content = re.sub(r'<br\b[^>]*>', ' ', html_content, flags=re.IGNORECASE)
        html_content = re.sub(r'</?(?:%s)\b[^>]*>' % '|'.join(HTML_BLOCK_TAGS), PARAGRAPH_SEP,
                              html_content, flags=re.IGNORECASE)
        html_content = re.sub(r'<[^>]+>', '', html_content)
        return normalize_paragraphs(unescape(html_content))

    def extract_text_from_file(self, file_path: Path) -> str:
        """Extract text from various file formats"""
        extension = file_path.suffix.lower()

        if extension == '.txt':
            return self._extract_txt(file_path)
        elif extension == '.pdf':
            return self._extract_pdf(file_path)
        elif extension == '.epub':
            return self.extract_text_from_epub(file_path)
        elif extension == '.docx' and DOCX_AVAILABLE:
            return self._extract_docx(file_path)
        elif extension == '.doc' and DOC_AVAILABLE:
            return self._extract_doc(file_path)
        else:
            raise ValueError(f"Unsupported file format: {extension}")

    def _extract_txt(self, file_path: Path) -> str:
        """Extract from TXT with encoding detection"""
        raw = Path(file_path).read_bytes()
        # UTF-16 only with a BOM: decoding arbitrary 8-bit text as UTF-16
        # "succeeds" and produces garbage. latin-1 accepts any byte, so it
        # is the last resort.
        encodings = ['utf-16'] if raw[:2] in (b'\xff\xfe', b'\xfe\xff') else []
        encodings += ['utf-8-sig', 'cp1252', 'latin-1']
        for encoding in encodings:
            try:
                return self._clean_text(raw.decode(encoding))
            except UnicodeDecodeError:
                continue
        raise ValueError("Could not decode text file")

    def _extract_pdf(self, file_path: Path) -> str:
        """Extract from PDF"""
        pages = []
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            self.logger.info(f"PDF has {total_pages} pages")

            for page_num, page in enumerate(pdf_reader.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        pages.append(page_text)
                    if page_num % 10 == 0:
                        self.logger.debug(f"Extracted {page_num}/{total_pages} pages")
                except Exception as e:
                    self.logger.warning(f"Failed to extract page {page_num}: {e}")
                    continue

        # Sentences run across page breaks, so pages are joined as plain lines
        text = "\n".join(pages)
        self.logger.info(f"Extracted text from {total_pages} pages, {len(text)} characters total")
        return self._clean_text(text)

    def _extract_docx(self, file_path: Path) -> str:
        """Extract from DOCX"""
        doc = Document(file_path)
        return normalize_paragraphs(PARAGRAPH_SEP.join(para.text for para in doc.paragraphs))

    def _extract_doc(self, file_path: Path) -> str:
        """Extract from DOC"""
        text = docx2txt.process(str(file_path))
        return self._clean_text(text) if text else ""

    def _clean_text(self, text: str) -> str:
        """Clean plain text, keeping paragraph (blank line) and section (2+ blank lines) breaks."""
        if not text:
            return ""
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        if CLEAN_PAGE_NUMBERS:
            # Drop lines holding nothing but a page number. A number standing
            # alone between blank lines leaves a single paragraph break.
            text = re.sub(r'\n[ \t]*\n[ \t]*\d{1,4}[ \t]*\n[ \t]*\n', '\n\n', text)
            text = re.sub(r'(?m)^[ \t]*\d{1,4}[ \t]*(?:\n|$)', '', text)
        # Re-join words hyphenated across a line break ("conti-\nnue")
        text = re.sub(r'([a-z])-\n[ \t]*([a-z])', r'\1\2', text)
        sections = re.split(r'\n[ \t]*\n(?:[ \t]*\n)+', text)
        return SECTION_SEP.join(s for s in (normalize_paragraphs(sec) for sec in sections) if s)

    def split_into_chunks(self, text: str) -> List[Chunk]:
        """Split text into TTS chunks of at most CHUNK_SIZE_WORDS words.

        Chunks never span a section (chapter) boundary, and end at a
        paragraph boundary once they are at least half full. Each chunk
        records the pause that follows it: chapter, paragraph or sentence.
        """
        chunks: List[Chunk] = []
        for section in re.split(r'\n\s*\n\s*\n', text):
            section_chunks: List[Chunk] = []
            cur: List[str] = []
            cur_words = 0
            at_paragraph_end = False

            def flush():
                nonlocal cur, cur_words
                pause = PARAGRAPH_PAUSE_MS if at_paragraph_end else SENTENCE_PAUSE_MS
                section_chunks.append(Chunk(" ".join(cur), pause))
                cur, cur_words = [], 0

            paragraphs = [" ".join(p.split()) for p in re.split(r'\n\s*\n', section)]
            for paragraph in merge_broken_paragraphs([p for p in paragraphs if p]):
                units = []
                for sentence in split_sentences(ensure_terminal_punctuation(paragraph)):
                    units.extend(split_long_sentence(sentence, CHUNK_SIZE_WORDS))
                paragraph_words = sum(len(u.split()) for u in units)

                if cur and cur_words + paragraph_words > CHUNK_SIZE_WORDS and cur_words >= CHUNK_SIZE_WORDS // 2:
                    flush()
                for unit in units:
                    unit_words = len(unit.split())
                    if cur and cur_words + unit_words > CHUNK_SIZE_WORDS:
                        flush()
                    cur.append(unit)
                    cur_words += unit_words
                    at_paragraph_end = False
                at_paragraph_end = True

            if cur:
                flush()
            if section_chunks:
                section_chunks[-1].pause_ms = CHAPTER_PAUSE_MS
            chunks.extend(section_chunks)

        return chunks

    def combine_chunks(self, chunks: List[Chunk], output_path: Path, results: Dict[int, bool]) -> bool:
        """Stream the generated chunk WAVs, with their pauses, through ffmpeg into the audiobook.

        Audio is piped chunk by chunk, so memory use stays flat for any book
        length. The output is written to a temporary name and renamed only
        on success, so a failed run never leaves a file that looks finished.
        """
        container = {"mp3": "mp3", "m4a": "mp4", "wav": "wav"}.get(AUDIO_FORMAT, AUDIO_FORMAT)
        codec = {"mp3": ["-c:a", "libmp3lame", "-b:a", AUDIO_BITRATE],
                 "m4a": ["-c:a", "aac", "-b:a", AUDIO_BITRATE],
                 "wav": ["-c:a", "pcm_s16le"]}.get(AUDIO_FORMAT, ["-b:a", AUDIO_BITRATE])
        temp_path = output_path.with_name(output_path.name + ".part")
        chunk_files = {i: Path("chunks") / f"chunk_{i:04d}.wav" for i in range(1, len(chunks) + 1)}
        included = [i for i in chunk_files if results.get(i) and chunk_files[i].exists()]
        if not included:
            self.logger.error("Failed to combine chunks: no valid chunks found")
            return False

        with wave.open(str(chunk_files[included[0]]), "rb") as first:
            sample_rate = first.getframerate()

        included_set = set(included)
        with tempfile.TemporaryFile() as ffmpeg_log:
            proc = None
            try:
                proc = subprocess.Popen(
                    [AudioSegment.converter, "-y", "-loglevel", "error",
                     "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0",
                     *codec, "-f", container, str(temp_path)],
                    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=ffmpeg_log,
                )
                combined = 0
                for i, chunk in enumerate(chunks, 1):
                    if i in included_set:
                        with wave.open(str(chunk_files[i]), "rb") as wav:
                            if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (sample_rate, 1, 2):
                                raise RuntimeError(f"Chunk {i} has an unexpected WAV format")
                            proc.stdin.write(wav.readframes(wav.getnframes()))
                        combined += 1
                        if combined % 100 == 0:
                            self.logger.info(f"Combined {combined} chunks")
                    proc.stdin.write(b"\x00\x00" * int(sample_rate * chunk.pause_ms / 1000))
                proc.stdin.close()
                if proc.wait() != 0:
                    raise RuntimeError(f"ffmpeg exited with code {proc.returncode}")
                os.replace(temp_path, output_path)
            except Exception as e:
                if proc is not None and proc.poll() is None:
                    proc.kill()
                    proc.wait()
                temp_path.unlink(missing_ok=True)
                ffmpeg_log.seek(0)
                details = ffmpeg_log.read().decode(errors="replace").strip()
                self.logger.error(f"Failed to combine chunks: {e}" + (f" | ffmpeg: {details}" if details else ""))
                return False

        self.logger.info(f"Audiobook saved: {output_path} ({len(included)}/{len(chunks)} chunks)")
        print(f"[INFO] Saved audiobook: {output_path.name} ({len(included)}/{len(chunks)} chunks)")
        return True

    def cleanup_chunks(self, cache_paths: Optional[List[Path]] = None):
        """Remove temporary chunk files, and the given cache files (a finished book's)."""
        try:
            # Clean up chunks folder
            chunk_count = 0
            for chunk_file in Path("chunks").glob("chunk_*.wav"):
                try:
                    chunk_file.unlink()
                    chunk_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to delete {chunk_file}: {e}")

            # Clean up this book's cache entries; other books' (and an
            # incomplete book's) cache is kept so conversions can resume
            cache_count = 0
            for cache_file in set(cache_paths or []):
                try:
                    if cache_file.exists():
                        cache_file.unlink()
                        cache_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to delete cache file {cache_file}: {e}")

            if chunk_count > 0 or cache_count > 0:
                self.logger.info(f"Cleaned up {chunk_count} chunk files and {cache_count} cache files")
                print(f"[INFO] Cleaned up {chunk_count} chunk files and {cache_count} cache files")
        except Exception as e:
            self.logger.warning(f"Cleanup failed: {e}")

    def _write_missing_report(self, report_path: Path, book_name: str, chunks: List[Chunk], missing: List[int]) -> None:
        """List the chunks that could not be generated, with their text."""
        lines = [f"{len(missing)} of {len(chunks)} chunks of {book_name} could not be generated.",
                 "Run the converter again to retry only these chunks; the rest are cached." if ENABLE_CACHING
                 else "Caching is disabled, so a new run regenerates every chunk.", ""]
        for i in missing:
            lines += [f"--- chunk {i} ---", chunks[i - 1].text, ""]
        report_path.write_text("\n".join(lines), encoding="utf-8")

    def convert_book(self, file_path: Path) -> bool:
        """Convert a single book to audiobook using faster-qwen3-tts.

        Returns True only when every chunk was generated. If some chunks
        fail, the audiobook is still written, as <name>.INCOMPLETE.<ext>,
        next to a <name>.MISSING.txt report, and the cache is kept so the
        next run regenerates only the missing chunks.
        """
        self.logger.info(f"Converting: {file_path.name}")
        start_time = time.time()
        self.last_output_path = None
        self.last_missing_chunks = []
        self.pace_wpm = []

        try:
            # Extract text
            self.logger.info("Extracting text...")
            text = self.extract_text_from_file(file_path)
            if not text.strip():
                self.logger.error("No text extracted")
                return False

            self.logger.info(f"Extracted {len(text)} characters ({len(text.split())} words)")

            # Split into chunks
            chunks = self.split_into_chunks(text)
            total_chunks = len(chunks)
            if total_chunks == 0:
                self.logger.error("No chunks created")
                return False

            # Log chunk info and estimate time based on streaming performance
            chunk_sizes = [len(chunk.text.split()) for chunk in chunks]
            avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
            self.logger.info(f"Split into {total_chunks} chunks (avg {avg_chunk_size:.0f} words per chunk)")

            # Estimate time from the measured RTF (~0.52-0.60 on the reference GPU)
            # plus ~15s of prefill/overhead per chunk.
            est_seconds_per_chunk = avg_chunk_size / TTS_WORDS_PER_MIN * 60.0 * 0.6 + 15
            est_time = total_chunks * est_seconds_per_chunk
            est_minutes = int(est_time // 60)
            print(f"[INFO] Processing {total_chunks} chunks via faster-qwen3-tts...")
            print(f"[INFO] Estimated time: ~{est_minutes} minutes")

            print(f"\n{'=' * 50}")
            print(f"PROCESSING {total_chunks} CHUNKS")
            print(f"{'=' * 50}")

            # Track results by chunk number
            results = {}  # chunk_num -> success (bool)

            # Process chunks sequentially to ensure correct order and naming
            for chunk_num, chunk in enumerate(chunks, 1):
                try:
                    result = self.process_chunk(chunk_num, chunk.text)
                    results[chunk_num] = result

                    if result:
                        print(f"[OK] Chunk {chunk_num:3d}/{total_chunks} completed")
                        self.logger.info(f"+ Chunk {chunk_num}/{total_chunks} completed")
                    else:
                        print(f"[FAIL] Chunk {chunk_num:3d}/{total_chunks} FAILED")
                        self.logger.error(f"- Chunk {chunk_num}/{total_chunks} failed")

                except Exception as e:
                    results[chunk_num] = False
                    print(f"[ERROR] Chunk {chunk_num:3d}/{total_chunks} ERROR: {e}")
                    self.logger.error(f"- Chunk {chunk_num}/{total_chunks} error: {e}")

                if chunk_num % 10 == 0:
                    self.log_pace()

            self.log_pace()
            missing = [i for i in range(1, total_chunks + 1) if not results.get(i)]
            successful_chunks = total_chunks - len(missing)
            print(f"\n{'=' * 50}")
            print(f"CHUNK PROCESSING COMPLETE")
            print(f"Successful: {successful_chunks}/{total_chunks}")
            print(f"{'=' * 50}")
            self.logger.info(f"Processing completed: {successful_chunks}/{total_chunks} chunks")

            if successful_chunks == 0:
                self.logger.error("No chunks were successfully processed")
                self.cleanup_chunks()
                return False

            final_path = Path(AUDIOBOOKS_FOLDER) / f"{file_path.stem}.{AUDIO_FORMAT}"
            incomplete_path = Path(AUDIOBOOKS_FOLDER) / f"{file_path.stem}.INCOMPLETE.{AUDIO_FORMAT}"
            report_path = Path(AUDIOBOOKS_FOLDER) / f"{file_path.stem}.MISSING.txt"
            output_path = incomplete_path if missing else final_path

            success = self.combine_chunks(chunks, output_path, results)
            if not success:
                self.logger.error("Failed to combine chunks into final audiobook")
                self.cleanup_chunks()  # generated audio stays in the cache for the next run
                return False

            self.last_output_path = output_path
            self.last_missing_chunks = missing
            duration = time.time() - start_time
            minutes = int(duration // 60)
            seconds = int(duration % 60)

            if missing:
                self._write_missing_report(report_path, file_path.name, chunks, missing)
                self.logger.error(
                    f"{len(missing)} chunk(s) could not be generated and are MISSING from "
                    f"{output_path}: {missing}. See {report_path}."
                )
                print(f"[WARNING] {len(missing)} chunk(s) are MISSING: {missing}")
                print(f"[WARNING] Saved as {output_path.name}; the missing text is listed in {report_path.name}")
                if ENABLE_CACHING:
                    print("[WARNING] Run the converter again to retry only the missing chunks.")
                self.cleanup_chunks()
                return False

            incomplete_path.unlink(missing_ok=True)
            report_path.unlink(missing_ok=True)
            self.logger.info(f"Conversion completed in {minutes}m {seconds}s: {output_path}")
            print(f"[SUCCESS] Conversion completed in {minutes}m {seconds}s")
            self.cleanup_chunks(cache_paths=[self.get_cache_path(chunk.text) for chunk in chunks])
            return True

        except Exception as e:
            self.logger.error(f"Conversion failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.cleanup_chunks()  # generated audio stays in the cache for the next run
            return False

    def run(self):
        """Main conversion process"""
        print("=" * 70)
        print("QWEN AUDIOBOOK CONVERTER (faster-qwen3-tts backend)")
        print("=" * 70)
        print(f"Books folder: {BOOKS_FOLDER}")
        print(f"Output folder: {AUDIOBOOKS_FOLDER}")
        print(f"Voice mode: {self.voice_mode}")
        print(f"Model: {MODEL_IDS.get(self.voice_mode)}")
        print(f"Device: {DEVICE}")
        if self.voice_mode == "custom_voice":
            print(f"Speaker: {self.speaker}")
            print(f"Language: {self.language}")
            print(f"Style: {self.instruct or '(none)'}")
        elif self.voice_mode == "voice_design":
            print(f"Voice description: {self.voice_description}")
        print(f"Output format: {AUDIO_FORMAT}")
        print(f"Streaming: {'enabled' if STREAMING_ENABLED else 'disabled'}")
        print(f"Chunk size: {CHUNK_SIZE_WORDS} words")
        print("=" * 70)

        # Check for books
        books_dir = Path(BOOKS_FOLDER)
        supported_formats = ['.txt', '.pdf', '.epub']
        if DOCX_AVAILABLE:
            supported_formats.append('.docx')
        if DOC_AVAILABLE:
            supported_formats.append('.doc')

        book_files = [f for f in books_dir.iterdir()
                      if f.is_file() and f.suffix.lower() in supported_formats]

        if not book_files:
            print(f"[INFO] No supported files found in {BOOKS_FOLDER}")
            print(f"Supported formats: {', '.join(supported_formats)}")

            # Create sample file
            sample_file = books_dir / "sample.txt"
            with open(sample_file, 'w') as f:
                f.write("This is a sample audiobook for testing the Qwen-based converter. "
                        "The system uses faster-qwen3-tts with CUDA graph optimization for "
                        "fast voice generation. You can replace this file with your own books to convert.")
            print(f"[INFO] Created sample file: {sample_file}")
            return

        # Skip books that already have a finished audiobook
        pending = []
        for book_file in book_files:
            existing = Path(AUDIOBOOKS_FOLDER) / f"{book_file.stem}.{AUDIO_FORMAT}"
            if existing.exists():
                print(f"[SKIP] {book_file.name}: {existing} already exists (delete it to convert again)")
            else:
                pending.append(book_file)

        if not pending:
            print("[INFO] Every book already has an audiobook; nothing to do.")
            return

        print(f"[INFO] Found {len(pending)} books to convert")

        # Convert each book
        results = {}
        for book_file in pending:
            try:
                success = self.convert_book(book_file)
                results[book_file.name] = success
            except KeyboardInterrupt:
                print("\n[WARNING] Conversion interrupted by user")
                break
            except Exception as e:
                self.logger.error(f"Unexpected error: {e}")
                results[book_file.name] = False

        # Print summary
        successful = sum(results.values())
        total = len(results)

        print("\n" + "=" * 70)
        print("CONVERSION SUMMARY")
        print("=" * 70)
        print(f"Total: {total} | Success: {successful} | Failed: {total - successful}")
        print("=" * 70)

        for filename, success in results.items():
            status = "[OK]" if success else "[FAIL]"
            print(f"{status} {filename}")

        if successful > 0:
            print(f"\n[INFO] Audiobooks saved to: {AUDIOBOOKS_FOLDER}/")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    """Entry point with argparse"""
    parser = argparse.ArgumentParser(
        description="Convert books to audiobooks using faster-qwen3-tts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use custom voice (default - Ryan speaker)
  python audiobook_converter.py

  # Use voice cloning with reference audio
  python audiobook_converter.py --voice-clone --voice-sample path/to/reference.wav

  # Use voice design mode
  python audiobook_converter.py --voice-design

  # Use xvector-only mode (no transcription needed for voice clone)
  python audiobook_converter.py --voice-clone --voice-sample ref.wav --xvector

        """
    )

    parser.add_argument(
        "--voice-clone",
        action="store_true",
        help="Use voice cloning mode instead of custom voice (requires --voice-sample)"
    )

    parser.add_argument(
        "--voice-sample",
        type=str,
        help="Path to reference audio file for voice cloning (WAV format)"
    )

    parser.add_argument(
        "--xvector",
        action="store_true",
        help="Use xvector-only mode for voice clone (no transcription needed)"
    )

    parser.add_argument(
        "--voice-design",
        action="store_true",
        help="Use voice design mode (instruction-based voice generation)"
    )

    args = parser.parse_args()

    # Determine voice mode: command-line flags first, then VOICE_MODE in config.py
    config_mode = _cfg("VOICE_MODE", "custom_voice")
    if args.voice_design or (not args.voice_clone and config_mode == "voice_design"):
        voice_mode = "voice_design"
        voice_clone_ref_audio = None
    elif args.voice_clone or config_mode == "voice_clone":
        voice_clone_ref_audio = args.voice_sample or _cfg("VOICE_CLONE_REF_AUDIO", "")
        if not voice_clone_ref_audio:
            print("[ERROR] Voice clone mode requires a reference audio file")
            print("Usage: python audiobook_converter.py --voice-clone --voice-sample <path>")
            print("   or: set VOICE_CLONE_REF_AUDIO in config.py")
            sys.exit(1)
        voice_mode = "voice_clone"
    else:
        voice_mode = "custom_voice"
        voice_clone_ref_audio = None

    try:
        converter = QwenAudiobookConverter(
            voice_mode=voice_mode,
            voice_clone_ref_audio=voice_clone_ref_audio,
            force_xvector=args.xvector
        )

        converter.run()
    except KeyboardInterrupt:
        print("\n[WARNING] Shutdown requested by user")
    except Exception as e:
        print(f"[FATAL] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
