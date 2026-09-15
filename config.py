# =============================================================================
# QWEN AUDIOBOOK CONVERTER - CONFIGURATION
# Uses faster-qwen3-tts backend (CUDA graph optimized)
# =============================================================================

# =============================================================================
# BACKEND SETTINGS
# =============================================================================

# faster-qwen3-tts settings
FASTER_QWEN_DEVICE = "cuda"          # torch device: cuda, cpu
FASTER_QWEN_DTYPE = "bfloat16"       # torch dtype: bfloat16, float16

# Streaming settings (enable for lower latency / faster first chunk)
STREAMING_ENABLED = True
STREAMING_CHUNK_SIZE = 8              # Decode steps per chunk (~667ms audio per chunk)

# =============================================================================
# TTS TOKEN BUDGET SETTINGS (chunk truncation prevention)
# =============================================================================
# faster-qwen3-tts limits each generation to `max_seq_len` TOTAL tokens
# (prompt + audio). The codec produces 12 audio tokens per second.
#
# Narration rate (measured 2026-09-11, Ryan, Labyrinths): a full 700-word
# chunk took 483.8s = 86.8 wpm. The previous 100 wpm calibration came from a
# TRUNCATED run (an upper bound) and underestimated token needs by ~15%,
# letting 700-word chunks nearly hit the cap. Budget at 85 wpm for headroom.
#
# Chunk length is also a QUALITY constraint, not just a token constraint:
# the talker is trained on short utterances and degrades to noise-like audio
# when a single generation runs too long (garble onset observed at ~184s in
# a 483.8s single generation; per-step text hints run out after ~83s).
# Keep each generation well under ~2 minutes of audio:
#
#   150 words  -> 4096   (default: ~106s of audio at 85 wpm)
#   200 words  -> 4096   (aggressive: ~141s; near the observed boundary)
#   300 words  -> 8192   (not recommended: ~212s, beyond the safe zone)
#
# Larger values use more VRAM (~+0.7 GB per 4096 tokens). If many chunks
# fail with "Generation hit the token cap", raise FASTER_QWEN_MAX_SEQ_LEN.

FASTER_QWEN_MAX_SEQ_LEN = 4096        # Total prompt+audio token budget per chunk
TTS_WORDS_PER_MIN = 85                # Narration rate used for budgeting (measured: Ryan ~87 wpm on literary text)
TTS_CODEC_TOKENS_PER_SEC = 12         # Codec frame rate (12 Hz models)
TTS_BUDGET_SAFETY_FACTOR = 1.2        # Headroom over the estimated speech length
TTS_MIN_MAX_NEW_TOKENS = 256          # Floor for the per-chunk audio token budget
TTS_VOICE_CLONE_REF_MARGIN = 512      # Extra prompt estimate for voice-clone ICL (ref audio in context)

# "Too short" floor for the post-generation length check (check_chunk_audio):
# audio shorter than TTS_TOO_SHORT_RATIO x expected is flagged as premature EOS.
#
# Measured 2026-09-11 (Ryan, Labyrinths, 150-word chunks, 216 generations):
# complete audio lands at 0.34-0.53 of the 85-wpm-based estimate (median 0.43)
# — the model narrates short chunks at ~160-236 wpm, i.e. ~2x faster than the
# rate measured on 700-word chunks (87 wpm). The audio is complete and clean;
# only the pace differs. The old 0.40 floor sat INSIDE that band, so ~1 in 6
# chunks failed all 3 retries and was dropped (chunks 14, 23, 52, 64, 66, 88,
# 89, 93, 104, 163, 164, 172, 177). 0.30 sits below the observed floor
# (0.34) with margin, while still catching catastrophic truncation (audio with
# >70% of the text missing). Note: a moderate early-EOS (30-70% of text) is
# length-ambiguous with fast-but-complete audio and cannot be separated by
# duration alone — the quality guard and the cap-hit check cover the other
# failure modes.
TTS_TOO_SHORT_RATIO = 0.30

# =============================================================================
# VOICE GENERATION MODE
# =============================================================================

# Options: "custom_voice", "voice_clone", "voice_design"
VOICE_MODE = "custom_voice"

# =============================================================================
# CUSTOM VOICE SETTINGS (Pre-built speakers)
# =============================================================================
# Uses Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice model
# Best for: General audiobook narration with professional quality voices

CUSTOM_VOICE_SPEAKER = "Ryan"        # Options: Aiden, Dylan, Eric, Ono_anna, Ryan, Serena, Sohee, Uncle_fu, Vivian
CUSTOM_VOICE_LANGUAGE = "English"    # Auto, Chinese, English, Japanese, Korean, French, German, Spanish, Portuguese, Russian
CUSTOM_VOICE_INSTRUCT = "Speak naturally and clearly, as if reading a book."  # Style instruction (1.7B only)
CUSTOM_VOICE_SEED = -1               # -1 for auto, or specific seed for consistency

# =============================================================================
# VOICE CLONE SETTINGS (Custom voice from reference audio)
# =============================================================================
# Uses Qwen/Qwen3-TTS-12Hz-1.7B-Base model
# Best for: Cloning a specific person's voice from a reference sample

VOICE_CLONE_REF_AUDIO = ""           # Path to reference audio file (WAV format)
VOICE_CLONE_REF_TEXT = ""            # Text matching what's spoken in the reference audio (required if VOICE_CLONE_USE_XVECTOR_ONLY=False)
VOICE_CLONE_LANGUAGE = "Auto"
VOICE_CLONE_USE_XVECTOR_ONLY = False  # True: lower quality but faster, no ref_text needed. False: ICL mode, requires ref_text, better quality
VOICE_CLONE_MAX_CHUNK_CHARS = 200
VOICE_CLONE_CHUNK_GAP = 0
VOICE_CLONE_SEED = -1                # -1 for auto
VOICE_CLONE_APPEND_SILENCE = True     # Append 0.5s silence to reference audio (prevents phoneme bleed)

# =============================================================================
# VOICE DESIGN SETTINGS (Describe the voice you want)
# =============================================================================
# Uses Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign model
# Best for: Expressive narration, character voices with custom tone/emotion

VOICE_DESIGN_LANGUAGE = "Auto"
VOICE_DESIGN_DESCRIPTION = "Speak in a clear, professional narrator voice suitable for reading audiobooks."
VOICE_DESIGN_SEED = -1               # -1 for auto

# =============================================================================
# PROCESSING SETTINGS
# =============================================================================

BOOKS_FOLDER = "book_to_convert"     # Input folder for books
CHUNK_SIZE_WORDS = 150               # Words per chunk. Must stay short for QUALITY, not just VRAM:
                                     # the talker produces noise-like garble when a single generation
                                     # runs past ~3 min (onset observed at 184s in a 483.8s chunk).
                                     # 150 words ~= 106s of audio at 85 wpm (safe zone).
                                     # 200 words (~141s) is the aggressive upper bound.
MAX_WORKERS = 1                      # Concurrent chunks (keep at 1 to avoid rate limiting)
MIN_DELAY_BETWEEN_CHUNKS = 0.5       # Seconds delay between API calls

# =============================================================================
# AUDIO OUTPUT SETTINGS
# =============================================================================

AUDIO_FORMAT = "mp3"                 # Output format: mp3, wav, m4a
AUDIO_BITRATE = "128k"               # Audio quality: 64k, 128k, 192k, 256k, 320k

# =============================================================================
# ADVANCED SETTINGS
# =============================================================================

# Supported file extensions
SUPPORTED_FORMATS = ['.txt', '.pdf', '.epub', '.docx', '.doc']

# Text cleaning options
CLEAN_PAGE_NUMBERS = True            # Remove standalone numbers
NORMALIZE_WHITESPACE = True          # Clean up spacing
SENTENCE_BOUNDARY_DETECTION = True   # Smart sentence splitting

# Cache settings
ENABLE_CACHING = True                # Cache processed chunks
CACHE_CLEANUP_DAYS = 30              # Remove cache older than X days

# Logging settings
LOG_LEVEL = "INFO"                   # DEBUG, INFO, WARNING, ERROR
LOG_TO_FILE = True                   # Save logs to file
LOG_TO_CONSOLE = True                # Display logs in terminal
