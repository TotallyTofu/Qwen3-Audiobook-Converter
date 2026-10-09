"""Dump full text of specific chunks to diagnose premature-EOS / fast-speech failures.

Usage: python inspect_chunks.py 52 163 164 172 177 77 81
       python inspect_chunks.py --book "book_to_convert/My Book.epub" 1 2 3
No GPU needed — bypasses __init__ (no model load), reuses the real
extraction/splitting code via unbound methods.
"""
import argparse
import logging
from pathlib import Path

import audiobook_converter as ac

conv = ac.QwenAudiobookConverter.__new__(ac.QwenAudiobookConverter)  # no model load
conv.logger = logging.getLogger("inspect")

PAUSE_NAMES = {
    ac.SENTENCE_PAUSE_MS: "sentence",
    ac.PARAGRAPH_PAUSE_MS: "paragraph",
    ac.CHAPTER_PAUSE_MS: "chapter",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--book", default=str(Path("book_to_convert") / "Labyrinths -- Borges Jorge Luis -- 1964.epub"))
    parser.add_argument("chunks", nargs="*", type=int, default=[52])
    args = parser.parse_args()

    text = conv.extract_text_from_file(Path(args.book))
    chunks = conv.split_into_chunks(text)
    print(f"total chunks: {len(chunks)}\n")
    for n in args.chunks:
        c = chunks[n - 1]
        pause = PAUSE_NAMES.get(c.pause_ms, "custom")
        print(f"===== chunk {n} ({len(c.text.split())} words, then {c.pause_ms} ms {pause} pause) =====")
        print(c.text)
        print()


if __name__ == "__main__":
    main()
