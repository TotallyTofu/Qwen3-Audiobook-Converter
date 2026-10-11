# 🎧 Qwen Audiobook Converter with CUDA Graph Optimization for Speedup 

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Qwen](https://img.shields.io/badge/Powered%20by-Qwen3%2DTTS-orange.svg)](https://github.com/QwenLM/Qwen3-TTS)

Convert PDFs, EPUBs, DOCX, DOC, and TXT files into high-quality audiobooks using **faster-qwen3-tts** with CUDA graph optimization for 5-10x speedup over baseline models.

## ✨ Features

- 🚀 **5-10x Faster**: CUDA graph optimization reduces processing time from ~4 minutes to ~30 seconds per chunk
- 🎤 **Three Voice Modes**
  - **Custom Voice**: Pre-built high-quality speakers (Ryan, Serena, Aiden, etc.) optimized for audiobook narration
  - **Voice Clone**: Clone any voice from a reference audio sample with ICL or xvector modes
  - **Voice Design**: Describe your desired voice tone and style using natural language
- 📚 **Multi-Format Support**: TXT, PDF, EPUB, DOCX, DOC
- 🤖 **1.7B Model Quality**: Uses the highest quality 1.7B model throughout
- 🔄 **Smart Chunking**: Intelligent text splitting with sentence boundary detection, sized to stay inside the model's quality zone
- 🛡️ **Truncation & Quality Detection**: Per-chunk token budget planning, silent-truncation detection, and a spectral guard against long-context garble
- 💾 **Intelligent Caching**: Avoids re-processing identical chunks
- 🔁 **Robust Error Handling**: Automatic retries with fresh seeds and graceful failure recovery
- 📊 **Progress Tracking**: Real-time conversion progress with RTF metrics
- ⚡ **Streaming Support**: Configurable streaming for lower time-to-first-audio

## 🔊 Audio Demo

🎧 **Sample Output**  
<figure>
  <figcaption>Listen to the T-Rex:</figcaption>
  <audio controls src="https://github.com/TotallyTofu/Qwen3-Audiobook-Converter/blob/main/sample/test_audio.mp3"></audio>
  <a href="https://github.com/TotallyTofu/Qwen3-Audiobook-Converter/blob/main/sample/test_audio.mp3"> Download audio </a>
</figure>

No it's not broken, it's a raw mp3 file download it and play it, you can't embedded audio in a readme.md GitHub whenthe sample is on GitHub

## 🧠 Performance Benchmarks (1.7B Model)

**RTF > 1.0 means faster than real-time.** On an RTX 4090, a 150-word chunk (~40s of audio) takes ~17 seconds to generate (RTF ~0.46).

## 🚀 Quick Start

### Prerequisites

1. **NVIDIA GPU** with CUDA support (RTX 3060 or better recommended, minimum 6GB VRAM)
2. **Python 3.10+** with pip
3. **PyTorch 2.5.1+** with CUDA support (automatically installed via pip)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/TotallyTofu/Qwen3-Audiobook-Converter.git
   cd Qwen3-Audiobook-Converter
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your books**:
   ```bash
   # Place your books in the book_to_convert folder
   cp your_book.pdf book_to_convert/
   ```

5. **Run the converter**:
    ```bash
    # Default: Custom Voice mode (Ryan speaker, English)
    python audiobook_converter.py

    # Voice Clone mode with xvector (no transcription needed)
    python audiobook_converter.py --voice-clone --voice-sample path/to/reference.wav --xvector
    ```

### Web UI (Recommended for Beginners)

Launch the interactive Gradio web interface:

```bash
# Start web server (opens in browser automatically)
python app.py

# Custom host and port
python app.py --host 0.0.0.0 --port 7861
```

Then open `http://localhost:7861` in your browser. It is one page with three steps:
1. **Choose a book**: drop in a PDF, EPUB, DOCX or TXT file
2. **Choose a voice**: a preset narrator, a clone of a voice sample, or a voice you describe in words
3. **Convert**: shows live progress (chunks done, time elapsed, time left), a **Stop** button, and, when it finishes, a player and download link for the audiobook

Other things to know:
- **Light / dark**: the sun/moon button in the corner switches theme. The first visit follows your operating system's setting; after that the page remembers your choice.
- **Close the tab any time.** The conversion runs on the server, so reopening the page picks up its progress (or shows the finished audiobook).
- **Stop** ends the run after the current chunk. With caching on (the default in `config.py`), chunks already generated stay cached, so converting the same book with the same settings again carries on from there.
- **Settings** (collapsed at the bottom) has the **Run conversions on** menu to pick the GPU for the next conversion, and lists the values read from `config.py`.
- Finished audiobooks are served straight from the `audiobooks/` folder, so playing or downloading a large file does not copy it.

## 📋 System Requirements

### Hardware

- **GPU**: NVIDIA GPU with CUDA support (RTX 3060+ recommended, minimum 6GB VRAM)
- **RAM**: 8GB+ system RAM recommended
- **Storage**: ~5GB for models + ~100MB per hour of audiobook output

### Software

- **Python**: 3.10 or higher
- **PyTorch**: 2.5.1+ with CUDA support (auto-installs via requirements.txt)
- **FFmpeg**: Required for MP3 output (`pip install pydub` will prompt, install separately)

## ⚙️ Configuration

### Main Settings

All settings can be configured in `config.py` or by editing the hardcoded values at the top of `audiobook_converter.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `CUSTOM_VOICE_SPEAKER` | Ryan | Speaker name (Ryan, Serena, Aiden, Dylan, Eric, etc.) |
| `CUSTOM_VOICE_LANGUAGE` | English | Target language |
| `CUSTOM_VOICE_INSTRUCT` | calm, slow, relaxed narration | Style instruction; also sets the pace (without a pace cue the model reads at ~190 wpm) |
| `CHUNK_SIZE_WORDS` | 150 | Words per processing chunk (see Chunk Sizing below) |
| `STREAMING_ENABLED` | False | Streaming decode only lowers time-to-first-audio, which an offline converter doesn't use |
| `FASTER_QWEN_DEVICE` | cuda:0 | GPU to run on, numbered as in `nvidia-smi` (the web UI's Settings section can pick another) |
| `AUDIO_FORMAT` | mp3 | Output format |
| `AUDIO_BITRATE` | 128k | Audio quality |

### Voice Modes

#### Custom Voice Mode (Default)

Uses pre-built speakers with audiobook-optimized narration style.

```bash
python audiobook_converter.py
```

**Available Speakers**:
| Speaker | Type | Description |
|---------|------|-------------|
| `Ryan` | Male | Clear and professional (default) |
| `Serena` | Female | Warm and friendly |
| `Aiden` | Male | Energetic and engaging |
| `Dylan` | Male | Calm and soothing |
| `Eric` | Male | Expressive and dynamic |
| `Ono_anna` | Female | Japanese accent support |
| `Sohee` | Female | Korean accent support |
| `Uncle_fu` | Male | Chinese accent support |
| `Vivian` | Female | Versatile |

#### Voice Clone Mode

Clone a voice from reference audio. Two sub-modes:

**ICL Mode (Default, Best Quality)**:
```bash
# Must set VOICE_CLONE_REF_TEXT in config.py with the transcript of your reference audio
python audiobook_converter.py --voice-clone --voice-sample path/to/reference.wav
```

**XVector Mode (Faster, No Transcription)**:
```bash
python audiobook_converter.py --voice-clone --voice-sample path/to/reference.wav --xvector
```

> **Note:** XVector mode extracts a speaker embedding once and reuses it. It produces slightly lower quality but avoids needing the reference audio transcript. ICL mode requires `VOICE_CLONE_REF_TEXT` to be set in `config.py`.

**Pace:** in ICL mode the clone continues on from the reference clip and picks
up its pace, so for slower narration use a clip that is read slowly and calmly
(~150 wpm is a typical audiobook pace). The log reports the clip's pace at
startup ("Reference clip pace: ... wpm") and the narration's as it goes
("Narration pace: median ... wpm"). Voice cloning takes no style instruction:
on the Base model one doubled chunk-to-chunk voice variation while slowing the
pace only ~4%.

#### Voice Design Mode

Describe the voice you want using natural language:

```bash
python audiobook_converter.py --voice-design
```

Configure the description in `config.py`:
```python
VOICE_DESIGN_DESCRIPTION = "Warm, confident narrator with slight British accent and slow pace"
```

### Processing Settings

| Setting | Value | Description |
|---------|-------|-------------|
| `CHUNK_SIZE_WORDS` | 150 | Words per processing chunk (quality-limited, see Chunk Sizing below) |
| `SENTENCE_PAUSE_MS` | 250 | Silence after a chunk that ends mid-paragraph |
| `PARAGRAPH_PAUSE_MS` | 1000 | Silence after a chunk that ends a paragraph |
| `CHAPTER_PAUSE_MS` | 2500 | Silence after a chunk that ends a chapter (EPUB document / text section) |
| `AUDIO_FORMAT` | mp3 | Output format (mp3, m4a, wav) |
| `AUDIO_BITRATE` | 128k | Audio quality |
| `MAX_RETRIES` | 3 | Retry attempts for failed chunks (hardcoded in `audiobook_converter.py`) |
| `MAX_SPLIT_DEPTH` | 2 | A chunk that fails every retry is split in half and retried, up to this many times (hardcoded) |
| `STREAMING_CHUNK_SIZE` | 8 | Steps per audio chunk when `STREAMING_ENABLED` is on |

Chunks never cross a chapter boundary and prefer to end at a paragraph break,
so the pauses above land where a narrator would pause.

### Chunk Sizing (Token Budget + Quality)

Chunk size is constrained by **three** independent limits:

**1. Token budget.** faster-qwen3-tts caps each generation at
`FASTER_QWEN_MAX_SEQ_LEN` **total** tokens (prompt + audio). The codec produces
12.5 audio tokens per second (1920 samples at 24 kHz, despite the "12Hz" name). If the cap is too small the tail of the chunk is
**silently dropped** — the audiobook then has missing paragraphs with no error
in the log. The converter sizes each chunk's token budget automatically and
fails loudly instead of truncating, but the two settings must stay matched:

| `CHUNK_SIZE_WORDS` | Required `FASTER_QWEN_MAX_SEQ_LEN` | Notes |
|--------------------|-------------------------------------|-------|
| 150 | 4096 | **Default.** ~106s of audio at the 85 wpm budgeting rate |
| 200 | 4096 | Aggressive: ~141s; near the quality boundary |
| 300 | 8192 | Not recommended: ~212s, beyond the safe zone |

Each +4096 of `FASTER_QWEN_MAX_SEQ_LEN` costs ~0.7 GB of VRAM. If a chunk's
budget cannot fit, the log shows the exact required value, e.g.:

```
Chunk would be truncated: ~1000 words need ~10165 audio tokens ..., but only
6391 fit in max_seq_len=8192 ... Raise FASTER_QWEN_MAX_SEQ_LEN to >= 11966
or lower CHUNK_SIZE_WORDS.
```

**2. Long-context quality.** The talker is trained on short utterances and
degrades to noise-like (garbled) audio when a single generation runs too long:
garble onset was observed at ~184s in a 483.8s single generation, and the
per-step text hints run out after ~83s. **Keep each generation well under
~2 minutes of audio** — this is why the default is 150 words, not 700. As a
safety net, a spectral-flatness guard (`check_chunk_audio_quality`) detects
degraded audio after generation and retries the chunk with a fresh seed.

**3. Short-chunk pacing.** The model narrates short chunks *faster* than long
ones: 700-word chunks measure ~87 wpm, while 150-word chunks measure
~160–236 wpm (the audio is complete and clean — only the pace differs). The
post-generation "too short" check is therefore calibrated with
`TTS_TOO_SHORT_RATIO` (0.30): audio shorter than 30% of the 85-wpm-based
estimate is flagged as premature EOS (retryable). Complete 150-word audio
measures 0.34–0.53 of the estimate, so a 0.40 floor makes good chunks fail
randomly. Note the budgeting rate (`TTS_WORDS_PER_MIN = 85`) is deliberately
the *slow* end of the observed range: an over-generous token cap is safe, an
under-sized one truncates.

**4. Runaway generations.** Sometimes the model fails to stop and keeps
babbling. Audio longer than `TTS_TOO_LONG_RATIO` (1.0) x the 85-wpm estimate
plus `TTS_TOO_LONG_SLACK_SEC` (3s), or audio that runs into the token cap, is
retried. Numbers count as one spoken word per digit when estimating length,
so bibliographies and dates aren't mistaken for runaways.

Related settings (see `config.py`): `TTS_WORDS_PER_MIN` (narration rate used
for budgeting, calibrated to 85), `TTS_CODEC_TOKENS_PER_SEC` (12.5),
`TTS_BUDGET_SAFETY_FACTOR` (1.2), `TTS_MIN_MAX_NEW_TOKENS` (256),
`TTS_VOICE_CLONE_REF_MARGIN` (512), `TTS_TOO_SHORT_RATIO` (0.30),
`TTS_TOO_LONG_RATIO` (1.0), `TTS_TOO_LONG_SLACK_SEC` (3.0).

> **Note:** Generated chunks are cached in `cache/audio_chunks/`, keyed by the
> chunk text and everything that affects the audio (voice, language, style
> instruction, model, budget settings), so changing a setting never replays old
> audio. An interrupted or incomplete conversion resumes from the cache; a
> book's cache entries are deleted once its audiobook is complete.

## 📖 Supported File Formats

| Format | Extension | Status |
|--------|-----------|--------|
| Plain Text | `.txt` | ✅ Full support |
| PDF | `.pdf` | ✅ Full support |
| EPUB | `.epub` | ✅ Full support |
| Word Document | `.docx` | ✅ Full support (requires python-docx) |
| Legacy Word | `.doc` | ✅ Full support (requires docx2txt) |

## 🎯 Usage Examples

### Basic Conversion

```bash
# Place your book in the input folder
cp "my_book.pdf" book_to_convert/

# Run the converter
python audiobook_converter.py

# Output will be in: audiobooks/my_book.mp3
```

### Batch Processing

```bash
# Add multiple books
cp *.pdf book_to_convert/
cp *.epub book_to_convert/

# Convert all at once
python audiobook_converter.py
```

### Voice Cloning With XVector

```bash
# One-command voice clone (no transcription needed)
python audiobook_converter.py \
  --voice-clone \
  --voice-sample "reference_audio.wav" \
  --xvector
```

### Custom Speaker Selection

Edit `config.py` to change speakers:
```python
CUSTOM_VOICE_SPEAKER = "Serena"
CUSTOM_VOICE_LANGUAGE = "English"
CUSTOM_VOICE_INSTRUCT = "Speak naturally and clearly, as if reading a dramatic book."
```

Or edit the hardcoded values at the top of `audiobook_converter.py`.

## 📁 Project Structure

```
Qwen3-Audiobook-Converter/
├── audiobook_converter.py    # Main conversion script
├── app.py                    # Gradio web UI (layout and the conversion job)
├── web/
│   ├── style.css             # Web UI look: colour tokens for light and dark, component styles
│   └── theme.js              # Light / dark toggle
├── config.py                 # Configuration file
├── test_chunk_budget.py      # Offline tests for budget/quality checks (no GPU needed)
├── test_text_pipeline.py     # Offline tests for extraction, chunking, retries, combining (no GPU needed)
├── inspect_chunks.py         # Offline tool: dump a chunk's text by number (no GPU needed)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── QUICKSTART.md             # Quick start guide
├── LICENSE                   # MIT License
├── book_to_convert/          # 📚 Input folder (place books here)
├── audiobooks/               # 🎧 Output folder (audiobooks saved here)
├── chunks/                   # ⚡ Temporary processing files (auto-cleaned)
├── cache/                    # 💾 Cached audio chunks
│   └── audio_chunks/
└── logs/                     # 📊 Processing logs
    └── audiobook_YYYYMMDD.log
```

## 🔍 How It Works

1. **Text Extraction**: Extracts text from various document formats (PDF, EPUB, DOCX, etc.), keeping paragraph and chapter breaks. EPUBs are read in spine (reading) order, one chapter per spine document
2. **Intelligent Chunking**: Splits text into ~150-word chunks at sentence boundaries (including sentences ending inside quotes), never across a chapter, preferably at paragraph breaks — short enough to stay inside the talker's quality zone (see Chunk Sizing below)
3. **CUDA Graph Optimization**: Uses static KV cache and CUDA graph capture for 5-10x speedup
4. **Voice Generation**: Generates audio locally using faster-qwen3-tts with the 1.7B model
5. **Streaming (Optional)**: Off by default; streaming only lowers time-to-first-audio
6. **Per-Chunk Verification**: After each generation, checks for token-cap hits and runaway (too long) audio, premature EOS ("too short" audio), and spectral degradation (garbled tails); retries with a fresh seed when needed, and splits a chunk that keeps failing into shorter pieces
7. **Progress Tracking**: Monitors chunk processing with RTF metrics in real-time
8. **Audio Assembly**: Streams the chunks through ffmpeg into the final audiobook, inserting sentence, paragraph and chapter pauses
9. **Cleanup and Resume**: Removes temporary files; the cache is kept until a book is complete, so re-running resumes an interrupted or incomplete conversion

### Faster-qwen3-tts vs Baseline Pipeline

```
Baseline (Qwen3-TTS original):
  Text → Tokenize → Dynamic Cache → Decode Step-by-step → Audio
  Each step = independent CUDA kernel launches = Python overhead per kernel

Faster-qwen3-tts:
  Text → Tokenize → Static KV Cache + CUDAGraph → Replay Single Operation → Audio
  Entire decode step fused into single GPU operation = minimal overhead
```

## 🧪 Offline Tests (No GPU Required)

The token-budget planning, truncation checks, and audio quality guard are
covered by a fast, dependency-light test suite:

```bash
python test_chunk_budget.py
python test_text_pipeline.py   # extraction, chunking, split-on-failure, combining (needs ffmpeg for the last)
```

`inspect_chunks.py` is a small companion tool for diagnosing chunk content
without loading the model:

```bash
python inspect_chunks.py 52 163 164   # print the exact text of chunks 52, 163, 164
python inspect_chunks.py --book "book_to_convert/My Book.epub" 1 2
```

## 🛠️ Troubleshooting

### CUDA / GPU Errors

```
Error: CUDA error: no kernel image is available for execution on the device
```

**Solutions**:
- Check your GPU compute capability matches PyTorch CUDA version
- Install correct PyTorch CUDA version: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
- Verify CUDA toolkit: `nvidia-smi`

### Model Download Fails

```
Error: Could not find model weights...
```

**Solutions**:
- Ensure stable internet connection (first run downloads ~4GB model)
- Try using a HuggingFace mirror or set `HF_ENDPOINT=https://hf-mirror.com`
- For restricted networks, download manually from HuggingFace Hub

### Voice Clone Mode Errors

```
[ERROR] Configuration Error! Voice Clone mode requires a reference audio file.
```

**Solutions**:
- Ensure `--voice-sample` points to a valid WAV file
- Verify the audio file exists and is readable
- For ICL mode, also set `VOICE_CLONE_REF_TEXT` in config.py
- Use `--xvector` flag to skip transcription requirement

### No Text Extracted

```
[ERROR] No text extracted from document
```

**Solutions**:
- Verify file isn't corrupted
- Check if document contains selectable text (not just scanned images)
- For image-based PDFs, use OCR first
- Try a different file format

### Processing Takes Too Long

On slower GPUs or without CUDA:
- **Solution**: Ensure PyTorch is installed with CUDA support (`torch.cuda.is_available()` should return `True`)
- CPU inference works but is significantly slower (~10-20x)
- Keep `STREAMING_ENABLED = False`: streaming re-decodes overlapping windows and gains nothing offline

### Out of GPU Memory

```
Error: CUDA out of memory
```

**Solutions**:
- Close other GPU applications
- Reduce `STREAMING_CHUNK_SIZE` to 4 or 2
- Reduce `CHUNK_SIZE_WORDS` to process smaller chunks
- On a multi-GPU machine, pick a GPU with more free memory (the web UI's Settings section, or `FASTER_QWEN_DEVICE` in config.py)

### FFmpeg Not Found

For MP3 output:
```bash
# Windows: choco install ffmpeg or download from https://ffmpeg.org/download.html
# Linux: sudo apt-get install ffmpeg
# macOS: brew install ffmpeg
```

### Chunk Fails: "Generation hit the token cap"

```
Chunk 42: Generation hit the token cap: 124.0s of audio at the 1589-token budget
(~127s) for ~106s of speech. The model most likely failed to stop; retrying.
```

**Cause**: the budget allows ~2.5x the time the model actually needs (it
narrates at ~150-250 wpm, the budget assumes 85 wpm x 1.2), so running into the
cap almost always means the model failed to emit EOS and kept babbling. That is
random, so the chunk is retried (and split into halves if it keeps failing).

**Solutions** (only if it happens on *most* chunks, i.e. your voice really is
that slow):
- Lower `TTS_WORDS_PER_MIN` or raise `FASTER_QWEN_MAX_SEQ_LEN` (e.g. 4096 → 8192; costs ~0.7 GB VRAM)
- Lower `CHUNK_SIZE_WORDS` (e.g. 150 → 100)

### Chunk Fails: "Audio is far longer than expected"

```
Chunk 36: Audio is far longer than expected: 113.9s for ~99s of speech
(limit 102s). Likely a runaway generation (babble or repetition); retrying.
```

**Cause**: a runaway generation that eventually stopped on its own. Retried
automatically. If many chunks hit this with a slow speaker or style
instruction, raise `TTS_TOO_LONG_RATIO` (e.g. 1.0 → 1.3).

### Audiobook saved as `*.INCOMPLETE.mp3`

When a chunk still fails after all retries and splits, the audiobook is saved
as `<book>.INCOMPLETE.<ext>` and the missing text is listed in
`audiobooks/<book>.MISSING.txt`. Run the converter again: every other chunk is
loaded from the cache, so only the missing chunks are regenerated, and the
final `<book>.<ext>` replaces the incomplete file. (Books whose final
audiobook already exists are skipped; delete it to convert the book again.)

### Chunk Fails: "Audio is far shorter than expected"

```
Chunk 52: Audio is far shorter than expected: 34.6s for ~89s of speech
(floor 30%). Likely premature EOS; retrying.
```

**Cause**: the model stopped speaking before finishing the text (early EOS).
Note that *complete* 150-word audio legitimately measures 0.34–0.53 of the
85-wpm-based estimate (the model paces short chunks at ~2x the budgeting rate),
so `TTS_TOO_SHORT_RATIO` is set to 0.30 — raising it makes good chunks fail
randomly. If many chunks fail this check even at 0.30, your speaker/instruct
is likely causing early stops: lower `CHUNK_SIZE_WORDS` (e.g. to 100) or try a
different `CUSTOM_VOICE_SPEAKER`.

### Chunk Fails: "Audio quality degraded"

```
Chunk 7: Audio quality degraded from ~187s: tail spectral flatness 0.62
(whole 0.11). Possible long-context drift; retrying.
```

**Cause**: long-context drift — the talker degrades to noise-like audio when a
single generation runs too long (onset observed at ~184s).

**Solutions**:
- Lower `CHUNK_SIZE_WORDS` (the default 150 keeps generations well under the
  onset; the 150-word default produced no quality failures in full-book runs)
- A chunk that keeps failing this check is split into shorter pieces
  automatically; shorter generations stay inside the talker's quality zone

## 🔧 Advanced Usage

### Speaker Embedding Reuse

For production use, extract speaker embedding once and reuse across multiple books:

```python
from faster_qwen3_tts import FasterQwen3TTS

model = FasterQwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")

# Extract embedding from reference audio (one-time, ~10s)
prompt_items = model.model.create_voice_clone_prompt(
    ref_audio="voice.wav", ref_text="", x_vector_only_mode=True
)
spk_emb = prompt_items[0].ref_spk_embedding
torch.save(spk_emb.detach().cpu(), "speaker.pt")

# Save and reuse
torch.load("speaker.pt", weights_only=True)  # Load when needed
```

### Adjusting Streaming Performance

Streaming is off by default for book conversion (it only helps interactive
playback). When enabled, it trades time-to-first-audio (TTFA) for throughput:

| `STREAMING_CHUNK_SIZE` | TTFA | RTF | Audio per Chunk |
|------------------------|------|-----|-----------------|
| 2 | ~250ms | 1.04x | 160ms |
| 8 | ~550ms | 1.38x | 640ms |
| 12 | ~750ms | 1.45x | 960ms |
| Non-streaming | N/A | 1.57x | All at once |

### Logging

Logs are saved to `logs/audiobook_YYYYMMDD.log` with detailed information about:
- Text extraction progress
- Chunk processing status (RTF, audio length per chunk)
- Caching decisions
- Errors and warnings

## 📊 Performance Reference

| Metric | Value |
|--------|-------|
| Processing Speed (RTX 4090) | ~17s per 150-word chunk (RTF ~0.46) |
| Quality | High-quality audio suitable for audiobooks |
| Memory Usage | ~6-8GB VRAM during inference |
| Storage | ~1MB per minute of audio (128kbps MP3) |
| Model Download Size | ~4GB (cached locally) |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/TotallyTofu/Qwen3-Audiobook-Converter.git
cd Qwen3-Audiobook-Converter

# Install dependencies
pip install -r requirements.txt

# Make your changes
# Test thoroughly
# Submit PR
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)** by the Qwen team - Base voice synthesis model
- **[faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts)** by andimarafioti - CUDA graph optimization for 5-10x speedup
- **[Gradio](https://gradio.app/)** - Original API interface framework
- All contributors and users of this project

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/TotallyTofu/Qwen3-Audiobook-Converter/issues)
- **Documentation**: See `config.py` for full configuration reference
- **Questions**: Open a discussion on GitHub

## 🔮 Roadmap

- [ ] GUI interface for easier configuration
- [ ] Chapter detection and automatic splitting
- [ ] Multiple output formats (M4B, OGG, FLAC)
- [ ] Real-time preview functionality
- [ ] Voice quality enhancement options
- [ ] Batch voice model switching
- [ ] Progress persistence (resume interrupted conversions)
- [ ] Whisper integration for automatic reference audio transcription

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

---

**Made with ❤️ for the audiobook community**
