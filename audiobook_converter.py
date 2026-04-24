#!/usr/bin/env python3
"""
Qwen-Based Audiobook Converter
Converts PDFs, EPUBs, DOCX, DOC, TXT files into audiobooks using faster-qwen3-tts

Uses faster-qwen3-tts with CUDA graph optimization for 5-10x speedup vs baseline.
Requires: NVIDIA GPU with CUDA, PyTorch >= 2.5.1
"""

import os
import shutil
import logging
import hashlib
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# faster-qwen3-tts Model Configuration
CUSTOM_VOICE_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
VOICE_CLONE_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DEVICE = _cfg("FASTER_QWEN_DEVICE", "cuda")
DTYPE = torch.bfloat16 if _cfg("FASTER_QWEN_DTYPE", "bfloat16") == "bfloat16" else torch.float16

# Hardcoded Voice Settings (CustomVoice mode)
CUSTOM_VOICE_SPEAKER = _cfg("CUSTOM_VOICE_SPEAKER", "Ryan")
CUSTOM_VOICE_LANGUAGE = _cfg("CUSTOM_VOICE_LANGUAGE", "English")
CUSTOM_VOICE_INSTRUCT = _cfg("CUSTOM_VOICE_INSTRUCT", "Speak naturally and clearly, as if reading a dramatic book to an adult audience.")
CUSTOM_VOICE_SEED = _cfg("CUSTOM_VOICE_SEED", -1)

# Voice Clone Settings (loaded from config.py with fallbacks)
VOICE_CLONE_LANGUAGE = _cfg("VOICE_CLONE_LANGUAGE", "Auto")
VOICE_CLONE_USE_XVECTOR_ONLY = _cfg("VOICE_CLONE_USE_XVECTOR_ONLY", False)
VOICE_CLONE_MAX_CHUNK_CHARS = _cfg("VOICE_CLONE_MAX_CHUNK_CHARS", 200)
VOICE_CLONE_CHUNK_GAP = _cfg("VOICE_CLONE_CHUNK_GAP", 0)
VOICE_CLONE_SEED = _cfg("VOICE_CLONE_SEED", -1)
VOICE_CLONE_APPEND_SILENCE = _cfg("VOICE_CLONE_APPEND_SILENCE", True)

# Voice Design Settings
VOICE_DESIGN_LANGUAGE = _cfg("VOICE_DESIGN_LANGUAGE", "Auto")
VOICE_DESIGN_DESCRIPTION = _cfg("VOICE_DESIGN_DESCRIPTION", "Speak in a clear, professional narrator voice suitable for reading audiobooks.")
VOICE_DESIGN_SEED = _cfg("VOICE_DESIGN_SEED", -1)

# Processing Settings (loaded from config.py with fallbacks)
BOOKS_FOLDER = _cfg("BOOKS_FOLDER", "book_to_convert")
AUDIOBOOKS_FOLDER = _cfg("AUDIOBOOKS_FOLDER", "audiobooks")
CHUNK_SIZE_WORDS = _cfg("CHUNK_SIZE_WORDS", 1000)
MAX_WORKERS = _cfg("MAX_WORKERS", 1)
AUDIO_FORMAT = _cfg("AUDIO_FORMAT", "mp3")
AUDIO_BITRATE = _cfg("AUDIO_BITRATE", "128k")
MIN_DELAY_BETWEEN_CHUNKS = _cfg("MIN_DELAY_BETWEEN_CHUNKS", 0.5)

# Streaming settings for faster generation (loaded from config.py with fallbacks)
STREAMING_ENABLED = _cfg("STREAMING_ENABLED", True)
STREAMING_CHUNK_SIZE = _cfg("STREAMING_CHUNK_SIZE", 8)  # Steps per chunk (~667ms audio per chunk)

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
# FASTER QWEN BACKEND
# =============================================================================

class FasterQwenBackend:
    """Backend using faster-qwen3-tts with CUDA graph optimization"""

    def __init__(self, device: str = DEVICE, dtype: torch.dtype = DTYPE):
        self.device = device
        self.dtype = dtype
        self.model = None
        self.voice_clone_ref_audio: Optional[str] = None
        self.voice_clone_ref_text: str = ""
        self.speaker_embedding_path: Optional[str] = None
        self.speaker_embedding: Optional[torch.Tensor] = None
        self.sample_rate: int = 24000

    def initialize(self, voice_mode: str, voice_clone_ref_audio: Optional[str] = None, voice_clone_ref_text: str = "") -> None:
        """Load the TTS model and set up voice configuration"""
        print("[INFO] Loading faster-qwen3-tts model...")
        print("[INFO] This may take a few minutes on first run (downloads model from HuggingFace)")

        model_id = CUSTOM_VOICE_MODEL_ID if voice_mode == "custom_voice" else VOICE_CLONE_MODEL_ID

        try:
            self.model = FasterQwen3TTS.from_pretrained(
                model_id,
                device=self.device,
                dtype=self.dtype,
            )
            self.sample_rate = self.model.sample_rate
            print(f"[OK] Model loaded successfully on {self.device}")
            print(f"[OK] Sample rate: {self.sample_rate} Hz")
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

    def generate_custom_voice(self, text: str) -> np.ndarray:
        """Generate audio using CustomVoice mode with pre-built speaker"""
        result = self.model.generate_custom_voice(
            text=text,
            language=CUSTOM_VOICE_LANGUAGE,
            speaker=CUSTOM_VOICE_SPEAKER,
            instruct=CUSTOM_VOICE_INSTRUCT,
        )
        audio_list, sr = result[0] if isinstance(result, tuple) else (result, self.sample_rate)
        # Return first audio array and sample rate
        audio = audio_list[0] if isinstance(audio_list, list) else audio_list
        return audio

    def generate_custom_voice_streaming(self, text: str) -> np.ndarray:
        """Generate audio using CustomVoice mode with streaming for faster TTFB"""
        result = self.model.generate_custom_voice_streaming(
            text=text,
            language=CUSTOM_VOICE_LANGUAGE,
            speaker=CUSTOM_VOICE_SPEAKER,
            instruct=CUSTOM_VOICE_INSTRUCT,
            chunk_size=STREAMING_CHUNK_SIZE,
        )
        # result is a generator yielding (audio_chunk, sr, timing) tuples
        all_chunks = []
        sr = self.sample_rate
        for audio_chunk, chunk_sr, timing in result:
            all_chunks.append(audio_chunk)
            sr = chunk_sr
        if len(all_chunks) == 1:
            return all_chunks[0]
        else:
            return np.concatenate(all_chunks)

    def generate_voice_clone(self, text: str) -> np.ndarray:
        """Generate audio using Voice Clone mode with ICL or xvector"""
        if not self.voice_clone_ref_audio:
            raise ValueError("Reference audio not set for voice cloning")

        # ICL takes priority if ref_text is available on the backend
        if self.voice_clone_ref_text:
            result = self.model.generate_voice_clone(
                text=text,
                language=VOICE_CLONE_LANGUAGE,
                ref_audio=self.voice_clone_ref_audio,
                ref_text=self.voice_clone_ref_text,
                xvec_only=False,
                append_silence=VOICE_CLONE_APPEND_SILENCE,
            )
        elif VOICE_CLONE_USE_XVECTOR_ONLY or self.speaker_embedding is not None:
            if self.speaker_embedding is not None:
                result = self.model.generate_voice_clone(
                    text=text,
                    language=VOICE_CLONE_LANGUAGE,
                    ref_audio=self.voice_clone_ref_audio,
                    ref_text="",
                    xvec_only=True,
                    voice_clone_prompt={
                        "ref_spk_embedding": [self.speaker_embedding],
                    },
                )
            else:
                result = self.model.generate_voice_clone(
                    text=text,
                    language=VOICE_CLONE_LANGUAGE,
                    ref_audio=self.voice_clone_ref_audio,
                    ref_text="",
                    xvec_only=True,
                )
        else:
            raise ValueError("Voice clone requires either ref_text (ICL mode) or speaker embedding (xvector mode)")

        audio_list, sr = result[0] if isinstance(result, tuple) else (result, self.sample_rate)
        audio = audio_list[0] if isinstance(audio_list, list) else audio_list
        return audio

    def generate_voice_clone_streaming(self, text: str) -> np.ndarray:
        """Generate audio using Voice Clone mode with streaming"""
        if not self.voice_clone_ref_audio:
            raise ValueError("Reference audio not set for voice cloning")

        non_streaming_mode = False  # Use step-by-step text feeding for streaming

        # ICL takes priority if ref_text is available on the backend
        if self.voice_clone_ref_text:
            result = self.model.generate_voice_clone_streaming(
                text=text,
                language=VOICE_CLONE_LANGUAGE,
                ref_audio=self.voice_clone_ref_audio,
                ref_text=self.voice_clone_ref_text,
                xvec_only=False,
                append_silence=VOICE_CLONE_APPEND_SILENCE,
                chunk_size=STREAMING_CHUNK_SIZE,
                non_streaming_mode=non_streaming_mode,
            )
        elif VOICE_CLONE_USE_XVECTOR_ONLY or self.speaker_embedding is not None:
            if self.speaker_embedding is not None:
                result = self.model.generate_voice_clone_streaming(
                    text=text,
                    language=VOICE_CLONE_LANGUAGE,
                    ref_audio=self.voice_clone_ref_audio,
                    ref_text="",
                    xvec_only=True,
                    voice_clone_prompt={"ref_spk_embedding": [self.speaker_embedding]},
                    chunk_size=STREAMING_CHUNK_SIZE,
                    non_streaming_mode=non_streaming_mode,
                )
            else:
                result = self.model.generate_voice_clone_streaming(
                    text=text,
                    language=VOICE_CLONE_LANGUAGE,
                    ref_audio=self.voice_clone_ref_audio,
                    ref_text="",
                    xvec_only=True,
                    chunk_size=STREAMING_CHUNK_SIZE,
                    non_streaming_mode=non_streaming_mode,
                )
        else:
            raise ValueError("Voice clone requires either ref_text (ICL mode) or speaker embedding (xvector mode)")

        all_chunks = []
        sr = self.sample_rate
        for audio_chunk, chunk_sr, timing in result:
            all_chunks.append(audio_chunk)
            sr = chunk_sr
        if len(all_chunks) == 1:
            return all_chunks[0]
        else:
            return np.concatenate(all_chunks)

    def generate_voice_design(self, text: str) -> np.ndarray:
        """Generate audio using VoiceDesign mode (instruction-based)"""
        result = self.model.generate_voice_design(
            text=text,
            language=VOICE_DESIGN_LANGUAGE,
            instruct=VOICE_DESIGN_DESCRIPTION,
        )
        audio_list, sr = result[0] if isinstance(result, tuple) else (result, self.sample_rate)
        audio = audio_list[0] if isinstance(audio_list, list) else audio_list
        return audio

    def generate_voice_design_streaming(self, text: str) -> np.ndarray:
        """Generate audio using VoiceDesign mode with streaming"""
        result = self.model.generate_voice_design_streaming(
            text=text,
            language=VOICE_DESIGN_LANGUAGE,
            instruct=VOICE_DESIGN_DESCRIPTION,
            chunk_size=STREAMING_CHUNK_SIZE,
        )
        all_chunks = []
        sr = self.sample_rate
        for audio_chunk, chunk_sr, timing in result:
            all_chunks.append(audio_chunk)
            sr = chunk_sr
        if len(all_chunks) == 1:
            return all_chunks[0]
        else:
            return np.concatenate(all_chunks)

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
                 voice_clone_ref_text: Optional[str] = None, force_xvector: bool = False):
        self.voice_mode = voice_mode
        self.voice_clone_ref_audio = voice_clone_ref_audio
        # Use passed ref_text first, then config.py fallback
        self.voice_clone_ref_text = voice_clone_ref_text if voice_clone_ref_text else ""
        self.setup_logging()
        self.setup_directories()
        self.validate_configuration()

        # Initialize faster-qwen3-tts backend
        self.backend = FasterQwenBackend(device=DEVICE, dtype=DTYPE)
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

    def setup_logging(self):
        """Setup logging configuration"""
        Path("logs").mkdir(exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"logs/audiobook_{datetime.now().strftime('%Y%m%d')}.log"),
                logging.StreamHandler(sys.stdout)
            ]
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
        """Get cache path for text chunk"""
        content = (
            f"{text}_{self.voice_mode}_"
            f"{CUSTOM_VOICE_SPEAKER if self.voice_mode == 'custom_voice' else ''}_"
            f"{Path(self.voice_clone_ref_audio).name if self.voice_clone_ref_audio else ''}"
        )
        hash_obj = hashlib.md5(content.encode())
        return Path("cache/audio_chunks") / f"{hash_obj.hexdigest()}.wav"

    def generate_chunk_via_backend(self, text: str, chunk_num: int) -> Optional[str]:
        """Generate audio chunk using faster-qwen3-tts backend"""
        try:
            # Check cache first
            cache_path = self.get_cache_path(text)
            if cache_path.exists():
                output_path = Path("chunks") / f"chunk_{chunk_num:04d}.wav"
                shutil.copy2(cache_path, output_path)
                self.logger.debug(f"Using cached audio for chunk {chunk_num}")
                return str(output_path)

            # Generate audio based on selected mode and settings
            start_time = time.time()

            if self.voice_mode == "custom_voice":
                if STREAMING_ENABLED:
                    audio = self.backend.generate_custom_voice_streaming(text)
                else:
                    audio = self.backend.generate_custom_voice(text)
            elif self.voice_mode == "voice_clone":
                if not self.backend.voice_clone_ref_audio:
                    raise ValueError("Reference audio not set for voice cloning")

                # ICL takes priority if ref_text is available on the backend
                if self.backend.voice_clone_ref_text:
                    if STREAMING_ENABLED:
                        audio = self.backend.generate_voice_clone_streaming(text)
                    else:
                        audio = self.backend.generate_voice_clone(text)
                elif VOICE_CLONE_USE_XVECTOR_ONLY or self.backend.speaker_embedding is not None:
                    # XVector mode - only use when no ref_text but xvector configured
                    if STREAMING_ENABLED:
                        audio = self.backend.generate_voice_clone_streaming(text)
                    else:
                        audio = self.backend.generate_voice_clone(text)
                else:
                    raise ValueError("Voice clone requires either ref_text (ICL mode) or speaker embedding (xvector mode)")
            elif self.voice_mode == "voice_design":
                if STREAMING_ENABLED:
                    audio = self.backend.generate_voice_design_streaming(text)
                else:
                    audio = self.backend.generate_voice_design(text)
            else:
                raise ValueError(f"Unknown voice mode: {self.voice_mode}")

            elapsed = time.time() - start_time
            rtf = elapsed / (len(audio) / self.backend.sample_rate) if len(audio) > 0 else 0
            self.logger.info(
                f"Chunk {chunk_num}: {elapsed:.1f}s, RTF: {rtf:.2f}, "
                f"Audio length: {len(audio)/self.backend.sample_rate:.1f}s"
            )

            # Convert numpy array to WAV file in chunks folder
            output_path = Path("chunks") / f"chunk_{chunk_num:04d}.wav"
            audio_segment = AudioSegment(
                (audio * 32768).astype(np.int16).tobytes(),
                frame_rate=self.backend.sample_rate,
                sample_width=2,  # 16-bit
                channels=1,
            )
            audio_segment.export(str(output_path), format="wav")

            # Cache the result
            shutil.copy2(output_path, cache_path)

            self.logger.debug(f"Chunk {chunk_num} generated successfully")
            return str(output_path)

        except Exception as e:
            self.logger.error(f"Backend chunk processing failed for chunk {chunk_num}: {e}")
            return None

    def process_chunk_with_retry(self, args: Tuple[int, str]) -> bool:
        """Process chunk with retry logic"""
        chunk_num, text = args

        # Small delay between chunks to allow GPU memory management
        if chunk_num > 1:
            time.sleep(MIN_DELAY_BETWEEN_CHUNKS)

        for attempt in range(MAX_RETRIES):
            try:
                result = self.generate_chunk_via_backend(text, chunk_num)
                if result and Path(result).exists():
                    return True
                else:
                    self.logger.warning(f"Chunk {chunk_num} attempt {attempt + 1} failed")
            except Exception as e:
                self.logger.warning(f"Chunk {chunk_num} attempt {attempt + 1} error: {e}")

            if attempt < MAX_RETRIES - 1:
                sleep_time = 5 + (2 ** attempt)
                self.logger.info(f"Waiting {sleep_time}s before retry...")
                time.sleep(sleep_time)

        self.logger.error(f"Chunk {chunk_num} failed after {MAX_RETRIES} attempts")
        return False

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
                    self.logger.info(f"EPUB extraction successful: {len(text)} characters")
                    return text
            except Exception as e:
                self.logger.warning(f"EPUB method failed: {e}")
                continue

        raise RuntimeError("All EPUB extraction methods failed")

    def _extract_epub_ebooklib(self, file_path: Path) -> str:
        """Extract using ebooklib"""
        book = epub.read_epub(str(file_path))
        text_parts = []

        for item_id, linear in book.spine:
            try:
                item = book.get_item_by_id(item_id)
                if item and isinstance(item, ebooklib.ITEM_DOCUMENT):
                    content = item.get_body_content()
                    if content:
                        if isinstance(content, bytes):
                            content = content.decode('utf-8', errors='ignore')
                        clean_text = self._clean_html(str(content))
                        if clean_text.strip():
                            text_parts.append(clean_text)
            except Exception:
                continue

        return '\n\n'.join(text_parts)

    def _extract_epub_zipfile(self, file_path: Path) -> str:
        """Extract using zipfile parsing"""
        text_parts = []
        with zipfile.ZipFile(file_path, 'r') as epub_zip:
            for file_name in epub_zip.namelist():
                if file_name.lower().endswith(('.html', '.xhtml', '.htm')):
                    try:
                        content = epub_zip.read(file_name).decode('utf-8', errors='ignore')
                        clean_text = self._clean_html(content)
                        if clean_text.strip():
                            text_parts.append(clean_text)
                    except Exception:
                        continue
        return '\n\n'.join(text_parts)

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
        return '\n\n'.join(text_parts)

    def _clean_html(self, html_content: str) -> str:
        """Clean HTML content"""
        if not html_content:
            return ""

        if BS4_AVAILABLE:
            try:
                soup = BeautifulSoup(html_content, 'html.parser')
                for script in soup(["script", "style"]):
                    script.decompose()
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                return ' '.join(chunk for chunk in chunks if chunk)
            except Exception:
                pass

        # Fallback regex cleaning
        html_content = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        html_content = re.sub(r'<[^>]+>', ' ', html_content)
        html_content = unescape(html_content)
        html_content = re.sub(r'\s+', ' ', html_content)
        return html_content.strip()

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
        for encoding in ['utf-8', 'utf-16', 'latin-1', 'cp1252']:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return self._clean_text(f.read())
            except UnicodeDecodeError:
                continue
        raise ValueError("Could not decode text file")

    def _extract_pdf(self, file_path: Path) -> str:
        """Extract from PDF"""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            self.logger.info(f"PDF has {total_pages} pages")

            for page_num, page in enumerate(pdf_reader.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text += f"\n\n{page_text}"
                    if page_num % 10 == 0:
                        self.logger.debug(f"Extracted {page_num}/{total_pages} pages")
                except Exception as e:
                    self.logger.warning(f"Failed to extract page {page_num}: {e}")
                    continue

            self.logger.info(f"Extracted text from {total_pages} pages, {len(text)} characters total")
        return self._clean_text(text)

    def _extract_docx(self, file_path: Path) -> str:
        """Extract from DOCX"""
        doc = Document(file_path)
        text = '\n\n'.join([para.text for para in doc.paragraphs if para.text.strip()])
        return self._clean_text(text)

    def _extract_doc(self, file_path: Path) -> str:
        """Extract from DOC"""
        text = docx2txt.process(str(file_path))
        return self._clean_text(text) if text else ""

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)
        text = text.replace('\n', ' ')
        text = re.sub(r'\b\d{1,3}\b(?=\s|$)', '', text)
        return text.strip()

    def split_into_chunks(self, text: str) -> List[str]:
        """Split text into manageable chunks"""
        if not text.strip():
            return []

        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        current_words = 0

        for sentence in sentences:
            sentence_words = len(sentence.split())

            if sentence_words > CHUNK_SIZE_WORDS:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    current_words = 0

                # Split long sentences
                parts = re.split(r'[,;:]', sentence)
                for part in parts:
                    part_words = len(part.split())
                    if current_words + part_words <= CHUNK_SIZE_WORDS:
                        current_chunk += part + " "
                        current_words += part_words
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = part + " "
                        current_words = part_words
            else:
                if current_words + sentence_words <= CHUNK_SIZE_WORDS:
                    current_chunk += sentence + " "
                    current_words += sentence_words
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "
                    current_words = sentence_words

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return [chunk for chunk in chunks if chunk.strip()]

    def combine_chunks(self, total_chunks: int, output_path: Path, results: Optional[Dict[int, bool]] = None) -> bool:
        """Combine audio chunks into final audiobook"""
        try:
            combined = AudioSegment.empty()
            successful = 0
            missing_chunks = []

            for i in range(1, total_chunks + 1):
                # Skip chunks that failed if we have results tracking
                if results is not None and not results.get(i, False):
                    missing_chunks.append(i)
                    continue

                chunk_file = Path("chunks") / f"chunk_{i:04d}.wav"
                if chunk_file.exists():
                    try:
                        chunk_audio = AudioSegment.from_wav(str(chunk_file))
                        combined += chunk_audio
                        successful += 1
                        if successful % 10 == 0:
                            self.logger.info(f"Combined {successful} chunks")
                    except Exception as e:
                        self.logger.warning(f"Failed to load chunk {i}: {e}")
                        missing_chunks.append(i)
                else:
                    self.logger.warning(f"Chunk file not found: {chunk_file}")
                    missing_chunks.append(i)

            if successful == 0:
                raise RuntimeError("No valid chunks found")

            if missing_chunks:
                self.logger.warning(f"Missing chunks: {missing_chunks}")

            combined.export(str(output_path), format=AUDIO_FORMAT, bitrate=AUDIO_BITRATE)
            self.logger.info(f"Audiobook saved: {output_path} ({successful}/{total_chunks} chunks)")
            print(f"[INFO] Saved audiobook: {output_path.name} ({successful}/{total_chunks} chunks)")
            if missing_chunks:
                print(f"[WARNING] Missing chunks: {missing_chunks}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to combine chunks: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def cleanup_chunks(self):
        """Remove temporary chunk files and cache"""
        try:
            # Clean up chunks folder
            chunk_count = 0
            for chunk_file in Path("chunks").glob("chunk_*.wav"):
                try:
                    chunk_file.unlink()
                    chunk_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to delete {chunk_file}: {e}")

            # Clean up cache folder
            cache_count = 0
            cache_dir = Path("cache/audio_chunks")
            if cache_dir.exists():
                for cache_file in cache_dir.glob("*.wav"):
                    try:
                        cache_file.unlink()
                        cache_count += 1
                    except Exception as e:
                        self.logger.warning(f"Failed to delete cache file {cache_file}: {e}")

            if chunk_count > 0 or cache_count > 0:
                self.logger.info(f"Cleaned up {chunk_count} chunk files and {cache_count} cache files")
                print(f"[INFO] Cleaned up {chunk_count} chunk files and {cache_count} cache files")
        except Exception as e:
            self.logger.warning(f"Cleanup failed: {e}")

    def convert_book(self, file_path: Path) -> bool:
        """Convert a single book to audiobook using faster-qwen3-tts"""
        self.logger.info(f"Converting: {file_path.name}")
        start_time = time.time()

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
            chunk_sizes = [len(chunk.split()) for chunk in chunks]
            avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
            self.logger.info(f"Split into {total_chunks} chunks (avg {avg_chunk_size:.0f} words per chunk)")

            # Estimate time: on RTX 4090, expected ~15-30 seconds per 1200-word chunk
            est_seconds_per_chunk = 20
            est_time = total_chunks * est_seconds_per_chunk
            est_minutes = int(est_time // 60)
            est_secs = est_time % 60
            print(f"[INFO] Processing {total_chunks} chunks via faster-qwen3-tts...")
            print(f"[INFO] Estimated time: ~{est_minutes} minutes ({est_secs} seconds)")

            # Process chunks - process in order to ensure correct naming
            chunk_args = [(i + 1, chunk) for i, chunk in enumerate(chunks)]

            print(f"\n{'=' * 50}")
            print(f"PROCESSING {total_chunks} CHUNKS")
            print(f"{'=' * 50}")

            # Track results by chunk number
            results = {}  # chunk_num -> success (bool)

            # Process chunks sequentially to ensure correct order and naming
            for chunk_num, chunk_text in chunk_args:
                try:
                    result = self.process_chunk_with_retry((chunk_num, chunk_text))
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

            successful_chunks = sum(1 for v in results.values() if v)
            print(f"\n{'=' * 50}")
            print(f"CHUNK PROCESSING COMPLETE")
            print(f"Successful: {successful_chunks}/{total_chunks}")
            print(f"{'=' * 50}")
            self.logger.info(f"Processing completed: {successful_chunks}/{total_chunks} chunks")

            if successful_chunks == 0:
                self.logger.error("No chunks were successfully processed")
                self.cleanup_chunks()
                return False

            if successful_chunks < total_chunks:
                self.logger.warning(
                    f"Only {successful_chunks}/{total_chunks} chunks succeeded. "
                    f"Proceeding with partial audiobook."
                )

            # Combine chunks (only the successful ones)
            output_path = Path(AUDIOBOOKS_FOLDER) / f"{file_path.stem}.{AUDIO_FORMAT}"
            success = self.combine_chunks(total_chunks, output_path, results)

            if success:
                duration = time.time() - start_time
                minutes = int(duration // 60)
                seconds = int(duration % 60)
                self.logger.info(f"Conversion completed in {minutes}m {seconds}s: {output_path}")
                print(f"[SUCCESS] Conversion completed in {minutes}m {seconds}s")
            else:
                self.logger.error("Failed to combine chunks into final audiobook")

            # Always cleanup, even on failure
            self.cleanup_chunks()
            return success

        except Exception as e:
            self.logger.error(f"Conversion failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.cleanup_chunks()
            return False

    def run(self):
        """Main conversion process"""
        print("=" * 70)
        print("QWEN AUDIOBOOK CONVERTER (faster-qwen3-tts backend)")
        print("=" * 70)
        print(f"Books folder: {BOOKS_FOLDER}")
        print(f"Output folder: {AUDIOBOOKS_FOLDER}")
        print(f"Voice mode: {self.voice_mode}")
        print(f"Model: {'CustomVoice' if self.voice_mode == 'custom_voice' else 'VoiceClone/VoiceDesign'}")
        print(f"Device: {DEVICE}")
        if self.voice_mode == "custom_voice":
            print(f"Speaker: {CUSTOM_VOICE_SPEAKER}")
            print(f"Language: {CUSTOM_VOICE_LANGUAGE}")
        elif self.voice_mode in ("voice_clone", "voice_design"):
            print(f"Backend model: {VOICE_CLONE_MODEL_ID if self.voice_mode == 'voice_clone' else CUSTOM_VOICE_MODEL_ID}")
        print(f"Output format: {AUDIO_FORMAT}")
        print(f"Streaming: {'enabled' if STREAMING_ENABLED else 'disabled'}")
        print(f"Chunk size: {STREAMING_CHUNK_SIZE} steps")
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

        print(f"[INFO] Found {len(book_files)} books to convert")

        # Convert each book
        results = {}
        for book_file in book_files:
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

    # Determine voice mode
    if args.voice_design:
        voice_mode = "voice_design"
        voice_clone_ref_audio = None
    elif args.voice_clone:
        if not args.voice_sample:
            print("[ERROR] --voice-clone requires --voice-sample")
            print("Usage: python audiobook_converter.py --voice-clone --voice-sample <path>")
            sys.exit(1)
        voice_mode = "voice_clone"
        voice_clone_ref_audio = args.voice_sample
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
