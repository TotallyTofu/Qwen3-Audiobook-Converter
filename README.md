# 🎧 Qwen Audiobook Converter with CUDA graph optimization for speedup 

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
- 🔄 **Smart Chunking**: Intelligent text splitting with sentence boundary detection
- 💾 **Intelligent Caching**: Avoids re-processing identical chunks
- 🔁 **Robust Error Handling**: Automatic retries and graceful failure recovery
- 📊 **Progress Tracking**: Real-time conversion progress with RTF metrics
- ⚡ **Streaming Support**: Configurable streaming for lower time-to-first-audio

## 🔊 Audio Demo

🎧 **Sample Output**  
<figure>
  <figcaption>Listen to the T-Rex:</figcaption>
  <audio controls src="https://github.com/WhiskeyCoder/Qwen3-Audiobook-Converter/blob/main/sample/test_audio.mp3"></audio>
  <a href="https://github.com/WhiskeyCoder/Qwen3-Audiobook-Converter/blob/main/sample/test_audio.mp3"> Download audio </a>
</figure>

No it's not broken, it's a raw mp3 file download it and play it, you can't embedded audio in a readme.md GitHub whenthe sample is on GitHub

## 🧠 Performance Benchmarks (1.7B Model)

**RTF > 1.0 means faster than real-time.** On an RTX 4090, a 1200-word chunk (~1 minute audio) takes ~30 seconds to generate.

## 🚀 Quick Start

### Prerequisites

1. **NVIDIA GPU** with CUDA support (RTX 3060 or better recommended, minimum 6GB VRAM)
2. **Python 3.10+** with pip
3. **PyTorch 2.5.1+** with CUDA support (automatically installed via pip)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/WhiskeyCoder/Qwen3-Audiobook-Converter.git
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

Then open `http://localhost:7861` in your browser. The web UI provides:
- **Text to Speech**: Quick voice generation with preview
- **Book Converter**: Upload and convert book files
- **Settings**: Tune streaming, chunk size, and view speaker info

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
| `CHUNK_SIZE_WORDS` | 1200 | Words per processing chunk |
| `STREAMING_ENABLED` | True | Use streaming for faster time-to-first-audio |
| `DEVICE` | cuda | Compute device (cuda, cpu) |
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
| `CHUNK_SIZE_WORDS` | 1200 | Words per processing chunk |
| `MAX_WORKERS` | 1 | Concurrent chunks (keep at 1 to avoid GPU memory issues) |
| `AUDIO_FORMAT` | mp3 | Output format |
| `AUDIO_BITRATE` | 128k | Audio quality |
| `MAX_RETRIES` | 3 | Retry attempts for failed chunks |
| `STREAMING_CHUNK_SIZE` | 8 | Steps per audio chunk (smallest = faster start, larger = better throughput) |

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
qwen-audiobook-converter/
├── audiobook_converter.py    # Main conversion script
├── config.py                 # Configuration file
├── requirements.txt          # Python dependencies
├── README.md                 # This file
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

1. **Text Extraction**: Extracts text from various document formats (PDF, EPUB, DOCX, etc.)
2. **Intelligent Chunking**: Splits text into optimal chunks (~1200 words) while respecting sentence boundaries
3. **CUDA Graph Optimization**: Uses static KV cache and CUDA graph capture for 5-10x speedup
4. **Voice Generation**: Generates audio locally using faster-qwen3-tts with the 1.7B model
5. **Streaming (Optional)**: Configurable streaming for lower time-to-first-audio latency
6. **Progress Tracking**: Monitors chunk processing with RTF metrics in real-time
7. **Audio Assembly**: Combines processed chunks into final audiobook via pydub
8. **Cleanup**: Automatically removes temporary files, even on failure

### Faster-qwen3-tts vs Baseline Pipeline

```
Baseline (Qwen3-TTS original):
  Text → Tokenize → Dynamic Cache → Decode Step-by-step → Audio
  Each step = independent CUDA kernel launches = Python overhead per kernel

Faster-qwen3-tts:
  Text → Tokenize → Static KV Cache + CUDAGraph → Replay Single Operation → Audio
  Entire decode step fused into single GPU operation = minimal overhead
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
- Adjust `STREAMING_CHUNK_SIZE` higher for better throughput at the cost of initial latency

### Out of GPU Memory

```
Error: CUDA out of memory
```

**Solutions**:
- Close other GPU applications
- Reduce `STREAMING_CHUNK_SIZE` to 4 or 2
- Reduce `CHUNK_SIZE_WORDS` to process smaller chunks
- Use `DEVICE="cpu"` as fallback (much slower)

### FFmpeg Not Found

For MP3 output:
```bash
# Windows: choco install ffmpeg or download from https://ffmpeg.org/download.html
# Linux: sudo apt-get install ffmpeg
# macOS: brew install ffmpeg
```

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

Trade-off between time-to-first-audio (TTFA) and throughput:

| `STREAMING_CHUNK_SIZE` | TTFA | RTF | Audio per Chunk |
|------------------------|------|-----|-----------------|
| 2 | ~250ms | 1.04x | 167ms |
| 8 | ~550ms | 1.38x | 667ms |
| 12 | ~750ms | 1.45x | 1000ms |
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
| Processing Speed (RTX 4090) | ~30s per 1200-word chunk |
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
git clone https://github.com/WhiskeyCoder/qwen-audiobook-converter.git
cd qwen-audiobook-converter

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

- **Issues**: [GitHub Issues](https://github.com/WhiskeyCoder/Qwen3-Audiobook-Converter/issues)
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
