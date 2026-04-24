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
CHUNK_SIZE_WORDS = 700               # Words per chunk (adjust based on your needs)
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
