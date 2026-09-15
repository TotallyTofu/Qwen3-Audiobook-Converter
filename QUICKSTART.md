# 🚀 Quick Start Guide

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

**Important**: Also install FFmpeg separately (required for MP3 output):
- Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)
- Linux: `sudo apt-get install ffmpeg`
- Mac: `brew install ffmpeg`

You also need an **NVIDIA GPU with CUDA** (RTX 3060 or better recommended,
minimum 6 GB VRAM). The 1.7B model (~4 GB) is downloaded automatically on the
first run.

## Step 2: Configure the Converter

Edit `config.py`:

### For Custom Voice (Easiest - Recommended)

```python
VOICE_MODE = "custom_voice"
CUSTOM_VOICE_SPEAKER = "Ryan"  # Try: Ryan, Serena, Aiden, etc.
CUSTOM_VOICE_LANGUAGE = "English"
```

### For Voice Cloning

```python
VOICE_MODE = "voice_clone"
VOICE_CLONE_REF_AUDIO = r"C:/path/to/your/reference_audio.wav"
VOICE_CLONE_REF_TEXT = "The text spoken in the reference audio"
```

### For Voice Design

```python
VOICE_MODE = "voice_design"
VOICE_DESIGN_DESCRIPTION = "Speak in a clear, professional narrator voice."
```

## Step 3: Add Your Books

Place your books in the `book_to_convert/` folder:
- Supported formats: `.txt`, `.pdf`, `.epub`, `.docx`, `.doc`

## Step 4: Run the Converter

```bash
python audiobook_converter.py
```

Prefer a browser? Launch the web UI instead:

```bash
python app.py
```

Then open `http://localhost:7861`.

## Step 5: Find Your Audiobook

Your completed audiobook will be in the `audiobooks/` folder!

## Troubleshooting

**Out of GPU memory?**
- Lower `CHUNK_SIZE_WORDS` (e.g. to 100) and/or `FASTER_QWEN_MAX_SEQ_LEN` (e.g. to 4096)
- Close other GPU applications

**FFmpeg not found?**
- Install FFmpeg and add it to your system PATH
- Restart your terminal after installation

**Need help?**
- See `README.md` for detailed documentation, including chunk sizing and
  the per-chunk quality checks