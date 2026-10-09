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

# Streaming settings. Streaming only lowers time-to-first-audio, which an
# offline converter never uses: each chunk is decoded in one pass when off.
STREAMING_ENABLED = False
STREAMING_CHUNK_SIZE = 8              # Decode steps per chunk (~640ms audio per chunk) when streaming

# =============================================================================
# TTS TOKEN BUDGET SETTINGS (chunk truncation prevention)
# =============================================================================
# faster-qwen3-tts limits each generation to `max_seq_len` TOTAL tokens
# (prompt + audio). The codec produces 12.5 audio tokens per second (1920
# samples per token at 24 kHz, despite the "12Hz" model name).
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
TTS_CODEC_TOKENS_PER_SEC = 12.5       # Codec frame rate (measured: chunks are whole multiples of 1920 samples)
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

# "Too long" ceiling (runaway detection): audio longer than
# TTS_TOO_LONG_RATIO x expected + TTS_TOO_LONG_SLACK_SEC is retried. The
# expectation uses the slow 85 wpm budgeting rate, while real narration runs
# ~150-250 wpm, so only generations that babble or repeat get this long
# (2026-09-11: chunk 36 ran 113.9s on its first attempt, 48.2s on retry).
# Numbers count as one word per digit when estimating length.
TTS_TOO_LONG_RATIO = 1.0
TTS_TOO_LONG_SLACK_SEC = 3.0

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
# Style instruction (1.7B only). Also controls pace: without a pace cue the
# model narrates at ~190 wpm (median, Labyrinths), faster than the ~150-160
# wpm of human audiobook narrators.
CUSTOM_VOICE_INSTRUCT = ("Narrate like a professional audiobook reader: calm and clear, "
                         "at a slow, relaxed pace, with natural pauses between sentences.")
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
VOICE_CLONE_SEED = -1                # -1 for auto

# Pace: voice cloning takes no style instruction (on the Base model one
# doubled chunk-to-chunk voice variation while slowing the pace only ~4%).
# In ICL mode the clone continues on from the reference clip and picks up
# its pace, so for slower narration use a clip read slowly and calmly
# (~150 wpm is a typical audiobook pace). The log reports the clip's pace
# ("Reference clip pace") and the narration's ("Narration pace").
VOICE_CLONE_APPEND_SILENCE = True     # Append 0.5s silence to reference audio (prevents phoneme bleed)

# =============================================================================
# VOICE DESIGN SETTINGS (Describe the voice you want)
# =============================================================================
# Uses Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign model
# Best for: Expressive narration, character voices with custom tone/emotion

VOICE_DESIGN_LANGUAGE = "Auto"
VOICE_DESIGN_DESCRIPTION = "A clear, calm, professional audiobook narrator reading at a slow, relaxed pace."
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
                                     # Chunks never cross a chapter and prefer to end at a paragraph.

# Silence added after each chunk in the final audiobook, by the boundary it
# ends on. The model leaves only ~120 ms of its own silence at a chunk's end
# (vs ~340 ms between sentences inside a chunk), so without these, sentences
# ran together at chunk joins.
SENTENCE_PAUSE_MS = 250              # Chunk ends mid-paragraph
PARAGRAPH_PAUSE_MS = 1000            # Chunk ends at a paragraph break
CHAPTER_PAUSE_MS = 2500              # Chunk ends a chapter (EPUB document / section)

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
CLEAN_PAGE_NUMBERS = True            # TXT/PDF: drop lines that hold only a page number

# Cache settings
ENABLE_CACHING = True                # Cache generated chunks so an interrupted or incomplete
                                     # conversion resumes where it stopped. A book's cache is
                                     # deleted once its audiobook is complete.

# Logging settings
LOG_LEVEL = "INFO"                   # DEBUG, INFO, WARNING, ERROR
LOG_TO_FILE = True                   # Save logs to file
LOG_TO_CONSOLE = True                # Display logs in terminal
