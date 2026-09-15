"""Dump full text of specific chunks to diagnose premature-EOS / fast-speech failures.

Usage: python inspect_chunks.py 52 163 164 172 177 77 81
No GPU needed — bypasses __init__ (no model load), reuses the real
extraction/splitting code via unbound methods.
"""
import logging
import sys
from pathlib import Path

import audiobook_converter as ac

conv = ac.QwenAudiobookConverter.__new__(ac.QwenAudiobookConverter)  # no model load
conv.logger = logging.getLogger("inspect")
split = ac.QwenAudiobookConverter.split_into_chunks.__get__(conv)
extract_epub = ac.QwenAudiobookConverter.extract_text_from_epub.__get__(conv)


def main():
    nums = [int(x) for x in sys.argv[1:]] or [52]
    path = Path("book_to_convert") / "Labyrinths -- Borges Jorge Luis -- 1964.epub"
    text = extract_epub(path)
    chunks = split(text)
    print(f"total chunks: {len(chunks)}\n")
    for n in nums:
        c = chunks[n - 1]
        words = c.split()
        print(f"===== chunk {n} ({len(words)} words) =====")
        print(c)
        print()


if __name__ == "__main__":
    main()