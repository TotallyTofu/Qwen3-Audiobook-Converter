"""Offline tests for text extraction, chunking, the split-on-failure fallback,
combining chunks into the final audiobook, and the web UI's progress/stop hooks.

Run with:  python test_text_pipeline.py

No GPU or model download required. The converter is created without loading
a model (like inspect_chunks.py), and generation is faked where needed. The
combine test needs ffmpeg on PATH and is skipped without it.
"""

import logging
import os
import shutil
import tempfile
import wave
import zipfile
from pathlib import Path

import numpy as np

import audiobook_converter as ac
from audiobook_converter import (
    Chunk,
    split_sentences,
    split_long_sentence,
    split_text_in_half,
    CHUNK_SIZE_WORDS,
    SENTENCE_PAUSE_MS,
    PARAGRAPH_PAUSE_MS,
    CHAPTER_PAUSE_MS,
)

SR = 24000


def make_converter():
    conv = ac.QwenAudiobookConverter.__new__(ac.QwenAudiobookConverter)  # no model load
    conv.logger = logging.getLogger("test")
    conv.voice_mode = "custom_voice"
    return conv


def words(n, prefix="w"):
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_html_extraction(conv):
    # Adjacent block elements must not fuse (2026-09-11: "Pliny.On the fourteenth...")
    html = (b"<html><head><title>Running Title</title></head><body>"
            b"<h1>Chapter One</h1><p>the work by Pliny.</p><p>On the fourteenth of February</p>"
            b"<p>and<br/>translated, <i>origin</i>ally</p></body></html>")
    text = conv._clean_html(html)
    paragraphs = text.split("\n\n")
    assert paragraphs == ["Chapter One", "the work by Pliny.", "On the fourteenth of February",
                          "and translated, originally"], paragraphs
    assert "Running Title" not in text, "the <head> title must not be read"
    print("  OK: block elements become paragraphs; <br> is a space; inline tags join; <head> skipped")

    # Same result through the regex fallback (no BeautifulSoup)
    saved = ac.BS4_AVAILABLE
    ac.BS4_AVAILABLE = False
    try:
        assert conv._clean_html(html).split("\n\n") == paragraphs
    finally:
        ac.BS4_AVAILABLE = saved
    print("  OK: regex fallback gives the same paragraphs")


def test_sentences():
    text = ('“Hello there.” She said it twice. Mr. Smith left early! '
            'Translated by J. E. Irby. "Why?" he asked.')
    assert split_sentences(text) == [
        "“Hello there.”", "She said it twice.", "Mr. Smith left early!",
        "Translated by J. E. Irby.", '"Why?"', "he asked.",
    ], split_sentences(text)
    print("  OK: sentence split handles closing quotes, abbreviations and initials")

    long_sentence = ", ".join(words(10, f"c{i}_") for i in range(20)) + "."  # 200 words
    pieces = split_long_sentence(long_sentence, 50)
    assert all(len(p.split()) <= 50 for p in pieces)
    assert " ".join(pieces) == long_sentence, "splitting must not drop words or commas"
    assert all(p.endswith((",", ".")) for p in pieces)
    no_punct = words(130)
    assert [len(p.split()) for p in split_long_sentence(no_punct, 50)] == [50, 50, 30]
    print("  OK: long sentences split at clauses, keep their commas, never exceed the limit")


def test_chunking(conv):
    para_a = " ".join(f"Sentence a{i} has five words." for i in range(20))  # 100 words
    para_b = " ".join(f"Sentence b{i} has five words." for i in range(10))  # 50 words
    para_c = " ".join(f"Sentence c{i} has five words." for i in range(6))   # 30 words
    text = "\n\n\n".join([
        "CHAPTER ONE\n\n" + para_a + "\n\n" + para_b + "\n\n" + para_c,
        "CHAPTER TWO\n\n" + para_c,
    ])
    chunks = conv.split_into_chunks(text)

    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(len(c.text.split()) <= CHUNK_SIZE_WORDS for c in chunks)
    assert chunks[0].text.startswith("CHAPTER ONE. Sentence a0"), "headings get a period"
    # Chapter one: heading + para_a (102 words) is over half full and para_b
    # does not fit, so the chunk ends at the paragraph break.
    assert chunks[0].text.endswith("a19 has five words."), chunks[0].text[-40:]
    assert chunks[0].pause_ms == PARAGRAPH_PAUSE_MS
    # The last chunk of each chapter ends with a chapter pause, and no chunk
    # spans the chapter boundary.
    chapter_ends = [c for c in chunks if c.pause_ms == CHAPTER_PAUSE_MS]
    assert len(chapter_ends) == 2
    assert chunks[-1].text.startswith("CHAPTER TWO.")
    assert not any("CHAPTER TWO" in c.text for c in chunks[:-1])
    # No words lost (only heading periods added)
    assert sum(len(c.text.split()) for c in chunks) == len(text.split())
    print(f"  OK: {len(chunks)} chunks; paragraph/chapter boundaries respected with their pauses")

    # A sentence broken across two blocks (copyright pages) is re-joined, not given a period
    chunks = conv.split_into_chunks("All rights reserved. No part of this book may be\n\n"
                                    "reproduced in any form.\n\nCHAPTER ONE\n\nIt begins.")
    assert chunks[0].text == ("All rights reserved. No part of this book may be reproduced in any form. "
                              "CHAPTER ONE. It begins."), chunks[0].text
    print("  OK: sentence split across blocks is re-joined; headings still get a period")

    # A paragraph longer than a chunk is split by sentences with sentence pauses
    long_para = " ".join(f"Sentence d{i} has five words." for i in range(70))  # 350 words
    chunks = conv.split_into_chunks(long_para)
    assert [c.pause_ms for c in chunks][:-1] == [SENTENCE_PAUSE_MS] * (len(chunks) - 1)
    assert chunks[-1].pause_ms == CHAPTER_PAUSE_MS
    assert all(c.text.endswith(".") for c in chunks), "mid-paragraph chunks end on a sentence"
    print("  OK: long paragraph splits on sentences with sentence pauses")


def test_plain_text(conv):
    raw = ("He was 25 years old and owned 300 books.\n"
           "42\n"
           "The story conti-\nnues on the next page.\n\n"
           "A new paragraph.\n\n\n\n"
           "CHAPTER 2\n\n"
           "Next chapter text.")
    text = conv._clean_text(raw)
    assert "25 years" in text and "300 books" in text, "numbers inside sentences must survive"
    assert "42" not in text, "a page-number line is removed"
    assert "continues" in text, "line-break hyphenation is re-joined"
    sections = text.split("\n\n\n")
    assert len(sections) == 2, sections
    assert sections[0].split("\n\n") == [
        "He was 25 years old and owned 300 books. The story continues on the next page.",
        "A new paragraph.",
    ], sections[0]
    assert sections[1] == "CHAPTER 2\n\nNext chapter text."
    print("  OK: numbers kept, page numbers dropped, hyphenation joined, paragraphs/sections kept")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cp1252.txt"
        path.write_bytes("“Quoted,” she said — twice.".encode("cp1252"))
        assert conv._extract_txt(path) == "“Quoted,” she said — twice."
    print("  OK: cp1252 text decodes correctly (not as UTF-16 garbage)")


def build_epub(path: Path):
    """Minimal EPUB whose archive order differs from its spine (reading) order."""
    docs = {"chapter2.xhtml": "Second chapter.", "chapter1.xhtml": "First chapter.",
            "about.xhtml": "About the author."}
    spine = ["chapter1.xhtml", "chapter2.xhtml", "about.xhtml"]
    opf = ('<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="2.0" '
           'unique-identifier="id"><metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
           '<dc:title>T</dc:title><dc:identifier id="id">x</dc:identifier><dc:language>en</dc:language>'
           '</metadata><manifest><item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>'
           + "".join(f'<item id="{n[:-6]}" href="{n}" media-type="application/xhtml+xml"/>' for n in docs)
           + '</manifest><spine toc="ncx">'
           + "".join(f'<itemref idref="{n[:-6]}"/>' for n in spine)
           + "</spine></package>")
    ncx = ('<?xml version="1.0"?><ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">'
           '<head/><docTitle><text>T</text></docTitle><navMap/></ncx>')
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("mimetype", "application/epub+zip")
        z.writestr("META-INF/container.xml",
                   '<?xml version="1.0"?><container version="1.0" '
                   'xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
                   '<rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>'
                   '</rootfiles></container>')
        z.writestr("OEBPS/content.opf", opf)
        z.writestr("OEBPS/toc.ncx", ncx)
        for name, body in docs.items():  # archive order: chapter2, chapter1, about
            z.writestr(f"OEBPS/{name}", f'<html xmlns="http://www.w3.org/1999/xhtml"><head>'
                                        f'<title>{name}</title></head><body><p>{body}</p></body></html>')


def test_epub_order(conv):
    expected = "First chapter.\n\n\nSecond chapter.\n\n\nAbout the author."
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "book.epub"
        build_epub(path)
        assert conv._extract_epub_ebooklib(path) == expected, conv._extract_epub_ebooklib(path)
        assert conv._extract_epub_zipfile(path) == expected, conv._extract_epub_zipfile(path)
        assert conv.extract_text_from_epub(path) == expected
    print("  OK: both EPUB methods read in spine order, one section per document")


class FakeBackend:
    """Fails (runs into the token cap) on any text longer than max_ok_words."""

    sample_rate = SR

    def __init__(self, max_ok_words, raise_error=False):
        self.max_ok_words = max_ok_words
        self.raise_error = raise_error
        self.calls = []

    def generate(self, voice_mode, text, max_new_tokens):
        self.calls.append(text)
        if self.raise_error:
            raise RuntimeError("CUDA out of memory")
        n = len(text.split())
        seconds = n / 150 * 60 if n <= self.max_ok_words else max_new_tokens / 12.5
        t = np.arange(int(seconds * SR)) / SR
        return (0.1 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)


def test_split_fallback(conv):
    text = " ".join(f"Sentence {i} has five words." for i in range(8))  # 40 words
    halves = split_text_in_half(text)
    assert len(halves) == 2 and " ".join(halves) == text
    assert [len(h.split()) for h in halves] == [20, 20]
    assert split_text_in_half("single") == ["single"]
    print("  OK: split_text_in_half cuts at a balanced sentence boundary")

    conv.backend = FakeBackend(max_ok_words=12)
    audio = conv.synthesize_chunk(text, "1")
    assert audio is not None, "chunk should succeed once split into short pieces"
    # 40 words fail 3x, each 20-word half fails 3x, the four 10-word quarters succeed
    assert len(conv.backend.calls) == 3 + 2 * 3 + 4
    expected = 40 / 150 * 60 + 3 * SENTENCE_PAUSE_MS / 1000
    assert abs(len(audio) / SR - expected) < 0.01, len(audio) / SR
    print(f"  OK: failing chunk recovered by splitting ({len(conv.backend.calls)} generations, pauses inserted)")

    conv.backend = FakeBackend(max_ok_words=0, raise_error=True)
    assert conv.synthesize_chunk(text, "2") is None
    assert len(conv.backend.calls) == ac.MAX_RETRIES, "non-text errors are retried, not split"
    print("  OK: errors unrelated to the text (e.g. OOM) are not split")


def test_voice_clone_reference(conv):
    backend = ac.FasterQwenBackend()
    backend.voice_clone_ref_audio, backend.voice_clone_ref_text = "ref.wav", "Reference transcript."
    kwargs = backend._voice_clone_kwargs()
    assert kwargs["xvec_only"] is False and "instruct" not in kwargs, "voice clone sends no style instruction"
    print("  OK: voice clone uses ICL with the transcript and no style instruction")

    # Reference clip pace: 0.5s silence + 4s speech + 0.5s silence, 10 words -> 150 wpm
    records = []
    handler = logging.Handler()
    handler.emit = lambda record: records.append(record.getMessage())
    conv.logger.addHandler(handler)
    saved_level = conv.logger.level
    conv.logger.setLevel(logging.INFO)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            clip = Path(tmp) / "ref.wav"
            t = np.arange(4 * SR) / SR
            speech = (8000 * np.sin(2 * np.pi * 180 * t)).astype(np.int16)
            silence = np.zeros(SR // 2, dtype=np.int16)
            with wave.open(str(clip), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SR)
                w.writeframes(np.concatenate([silence, speech, silence]).tobytes())
            conv.voice_clone_ref_audio = str(clip)
            conv.voice_clone_ref_text = "one two three four five six seven eight nine ten"
            conv._log_reference_pace()
    finally:
        conv.logger.removeHandler(handler)
        conv.logger.setLevel(saved_level)
    assert any("Reference clip pace: 10 words in 4.0s of speech = 150 wpm" in r for r in records), records
    print("  OK: reference clip pace measured on speech only (150 wpm)")


def write_wav(path, seconds, value=1000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(np.full(int(seconds * SR), value, dtype=np.int16).tobytes())


def test_combine(conv):
    if shutil.which(ac.AudioSegment.converter) is None:
        print("  SKIP: ffmpeg not found; combine test not run")
        return
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            Path("chunks").mkdir()
            chunks = [Chunk("a", 250), Chunk("b", 700), Chunk("c", 2000)]
            for i, sec in enumerate([1.0, 2.0, 1.5], 1):
                write_wav(Path("chunks") / f"chunk_{i:04d}.wav", sec)
            saved = ac.AUDIO_FORMAT
            ac.AUDIO_FORMAT = "wav"
            try:
                out = Path("book.wav")
                assert conv.combine_chunks(chunks, out, {1: True, 2: False, 3: True})
            finally:
                ac.AUDIO_FORMAT = saved
            with wave.open(str(out), "rb") as w:
                duration = w.getnframes() / w.getframerate()
            # chunk 1 + pause, (chunk 2 missing) + its pause, chunk 3 + pause
            assert abs(duration - (1.0 + 0.25 + 0.7 + 1.5 + 2.0)) < 0.01, duration
            assert not Path("book.wav.part").exists()
        finally:
            os.chdir(cwd)
    print(f"  OK: combined {duration:.2f}s = audio + pauses; missing chunk skipped; no .part left")


def test_progress_and_stop():
    """The web UI's hooks, on the real convert_book: a progress report per chunk,
    and a stop request that ends the run without writing a partial audiobook."""
    if shutil.which(ac.AudioSegment.converter) is None:
        print("  SKIP: ffmpeg not found; progress/stop test not run")
        return
    conv = make_converter()  # a fresh one: this test sets voice attributes on it
    conv.speaker, conv.language, conv.instruct = "Ryan", "English", "calm"
    conv.backend = FakeBackend(max_ok_words=1000)

    cwd = os.getcwd()
    saved_format = ac.AUDIO_FORMAT
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        ac.AUDIO_FORMAT = "wav"
        try:
            for folder in ("chunks", "cache/audio_chunks", "audiobooks"):
                Path(folder).mkdir(parents=True)
            book = Path("book.txt")
            # 3 chunks of 100 words. Plain words: the length estimate counts every digit as a word
            book.write_text("\n\n".join(" ".join([w] * 100) + "." for w in ("alpha", "bravo", "charlie")),
                            encoding="utf-8")

            events = []
            conv.progress_callback = lambda stage, done, total: events.append((stage, done, total))
            assert conv.convert_book(book)
            assert events[0][0] == "reading" and events[-1][0] == "combining", events
            generating = [(done, total) for stage, done, total in events if stage == "generating"]
            assert generating == [(0, 3), (1, 3), (2, 3), (3, 3)], generating
            print("  OK: progress reports reading, each chunk (0..3 of 3), then combining")

            # Stop once the first chunk is done: nothing is written, that chunk stays cached
            Path("audiobooks", "book.wav").unlink()
            events.clear()
            conv.should_stop = lambda: any(stage == "generating" and done >= 1 for stage, done, _ in events)
            assert not conv.convert_book(book)
            assert conv.last_output_path is None and not list(Path("audiobooks").iterdir())
            assert max(done for stage, done, _ in events if stage == "generating") == 1, events
            assert all(stage != "combining" for stage, _, _ in events)
            assert len(list(Path("cache/audio_chunks").glob("*.wav"))) == (1 if ac.ENABLE_CACHING else 0)
            assert not list(Path("chunks").glob("chunk_*.wav")), "temporary chunk files are cleaned up"
        finally:
            ac.AUDIO_FORMAT = saved_format
            os.chdir(cwd)
    print("  OK: stop request ends the run after chunk 1 with no audiobook written; finished chunk stays cached")


def main():
    conv = make_converter()
    test_html_extraction(conv)
    test_sentences()
    test_chunking(conv)
    test_plain_text(conv)
    test_epub_order(conv)
    test_split_fallback(conv)
    test_voice_clone_reference(conv)
    test_combine(conv)
    test_progress_and_stop()
    print("\nAll text pipeline tests passed.")


if __name__ == "__main__":
    main()
