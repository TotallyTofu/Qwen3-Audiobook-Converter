#!/usr/bin/env python3
"""
Gradio Web Interface for Qwen Audiobook Converter
Provides a user-friendly web UI for converting books to audiobooks using faster-qwen3-tts

The look lives in web/style.css (all colours are tokens, with a light and a
dark set) and web/theme.js (light/dark toggle); this file holds the layout
and the conversion job.
"""

import gc
import html
import shutil
import sys
import threading
import time
import traceback
import zipfile
from pathlib import Path
from statistics import median
from typing import List, Optional, Tuple
from urllib.parse import quote

import gradio as gr
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from audiobook_converter import (
    QwenAudiobookConverter,
    CUSTOM_VOICE_SPEAKER,
    CUSTOM_VOICE_LANGUAGE,
    CUSTOM_VOICE_INSTRUCT,
    BOOKS_FOLDER,
    AUDIOBOOKS_FOLDER,
    CHUNK_SIZE_WORDS,
    STREAMING_ENABLED,
    AUDIO_FORMAT,
    ENABLE_CACHING,
    FASTER_QWEN_MAX_SEQ_LEN,
    SENTENCE_PAUSE_MS,
    PARAGRAPH_PAUSE_MS,
    CHAPTER_PAUSE_MS,
    VOICE_DESIGN_DESCRIPTION,
    DEVICE,
)

WEB_DIR = Path(__file__).parent / "web"

BOOK_EXTENSIONS = {".pdf", ".epub", ".docx", ".txt"}

# (name, gender, note) - the names are what the model accepts
SPEAKERS = [
    ("Ryan", "Male", "Clear and professional"),
    ("Serena", "Female", "Warm and friendly"),
    ("Aiden", "Male", "Energetic and engaging"),
    ("Dylan", "Male", "Calm and soothing"),
    ("Eric", "Male", "Expressive and dynamic"),
    ("Ono_anna", "Female", "Japanese accent support"),
    ("Sohee", "Female", "Korean accent support"),
    ("Uncle_fu", "Male", "Chinese accent support"),
    ("Vivian", "Female", "Versatile"),
]
LANGUAGES = ["Auto", "English", "Chinese", "Japanese", "Korean", "French", "German",
             "Spanish", "Portuguese", "Russian"]

VOICE_MODES = [("Preset voice", "custom_voice"),
               ("Clone a voice", "voice_clone"),
               ("Design a voice", "voice_design")]
MODE_HINTS = {
    "custom_voice": "Pick one of the ready-made narrators.",
    "voice_clone": "Upload a short sample and the narrator will sound like it. "
                   "The clone also copies the sample's pace, so use one read slowly and calmly.",
    "voice_design": "Describe the narrator you want in plain words.",
}


def detect_book_extension(path: Path) -> str:
    """Guess a book's extension from its content.

    Gradio cuts uploaded file names to 200 bytes, extension included, so a
    long name (e.g. an Anna's Archive download) arrives with no extension.
    The upload widget only accepts BOOK_EXTENSIONS, so anything that is not
    a PDF, EPUB or DOCX is plain text.
    """
    with open(path, "rb") as f:
        head = f.read(4)
    if head == b"%PDF":
        return ".pdf"
    if head == b"PK\x03\x04" and zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            names = set(z.namelist())
            if "mimetype" in names and z.read("mimetype").strip() == b"application/epub+zip":
                return ".epub"
            if "META-INF/container.xml" in names:
                return ".epub"
            if "word/document.xml" in names:
                return ".docx"
    return ".txt"


def list_gpus() -> List[Tuple[str, str]]:
    """(label, device) for each CUDA GPU, numbered as in nvidia-smi."""
    if not torch.cuda.is_available():
        return []
    gpus = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        gpus.append((f"GPU {i}: {props.name} ({props.total_memory / 2**30:.0f} GB)", f"cuda:{i}"))
    return gpus


# =============================================================================
# CONVERSION JOB
# =============================================================================
# A conversion runs on its own thread and the page only watches it. That lets
# the page show live progress, and lets you close the tab and come back later:
# reopening the page picks the job up again. Only one job runs at a time (one
# GPU, one model in memory).

ACTIVE_STAGES = ("loading", "reading", "generating", "combining")
STAGE_LABELS = {
    "loading": "Loading the voice model",
    "reading": "Reading the book",
    "generating": "Generating audio",
    "combining": "Assembling the audiobook",
}


class Job:
    """One book conversion: its progress while it runs, its outcome after."""

    def __init__(self, book: str, voice: str):
        self.book = book
        self.voice = voice
        self.started = time.time()
        self.ended: Optional[float] = None
        # loading > reading > generating > combining > done | partial | stopped | failed
        self.stage = "loading"
        self.done = 0
        self.total = 0
        self.marks: List[float] = []  # when each chunk finished (first entry: generation start)
        self.stop = threading.Event()
        self.output: Optional[Path] = None
        self.missing: List[int] = []
        self.error = ""

    @property
    def running(self) -> bool:
        return self.stage in ACTIVE_STAGES

    def on_progress(self, stage: str, done: int, total: int) -> None:
        """Receives the converter's progress reports (on the worker thread)."""
        if stage == "generating" and done == 0:
            self.marks = [time.time()]
        elif stage == "generating" and done > self.done:
            self.marks.append(time.time())
        self.done, self.total = done, total
        self.stage = stage

    def eta(self) -> Optional[float]:
        """Seconds left, from recent chunk times; None until a chunk has really been generated.

        Chunks served from the cache finish in milliseconds, so they are
        left out of the average.
        """
        if self.stage != "generating" or self.done >= self.total:
            return None
        recent = [b - a for a, b in zip(self.marks, self.marks[1:])][-8:]
        recent = [t for t in recent if t > 2.0]
        return median(recent) * (self.total - self.done) if recent else None


_job: Optional[Job] = None
_job_lock = threading.Lock()


def copy_book_into_library(src: str) -> Path:
    """Copy an uploaded book into the books folder, restoring a truncated extension."""
    dest = Path(BOOKS_FOLDER) / Path(src).name
    if dest.suffix.lower() not in BOOK_EXTENSIONS:
        ext = detect_book_extension(Path(src))
        print(f"[INFO] Upload name has no book extension (Gradio truncates names to 200 bytes); "
              f"detected {ext} from content")
        dest = dest.with_name(dest.name + ext)
    Path(BOOKS_FOLDER).mkdir(exist_ok=True)
    shutil.copy2(src, dest)
    return dest


def run_job(job: Job, book_path: Path, voice_mode: str, converter_args: dict) -> None:
    """Worker thread: load the model, convert the book, record the outcome on `job`."""
    converter = None
    try:
        converter = QwenAudiobookConverter(voice_mode=voice_mode, **converter_args)
        converter.progress_callback = job.on_progress
        converter.should_stop = job.stop.is_set
        if job.stop.is_set():
            job.stage = "stopped"
            return

        print(f"[INFO] Converting {book_path.name} on {converter_args['device']}...")
        success = converter.convert_book(book_path)

        # The outcome is filled in before `stage` flips, which is what the page watches
        job.output = converter.last_output_path
        job.missing = list(converter.last_missing_chunks)
        if success:
            job.stage = "done"
        elif job.output is not None:
            job.stage = "partial"
        elif job.stop.is_set():
            job.stage = "stopped"
        else:
            job.error = "The book could not be converted."
            job.stage = "failed"
    except SystemExit:
        # The converter exits when the voice model fails to load
        job.error = "The voice model could not be loaded."
        job.stage = "failed"
    except Exception as e:
        traceback.print_exc()
        job.error = str(e) or type(e).__name__
        job.stage = "failed"
    finally:
        job.ended = time.time()
        # Free the model's GPU memory before the next conversion loads one
        del converter
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# =============================================================================
# STATUS PANEL (HTML)
# =============================================================================

def fmt_duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def fmt_size(path: Path) -> str:
    size = path.stat().st_size
    if size >= 2**30:
        return f"{size / 2**30:.1f} GB"
    if size >= 2**20:
        return f"{size / 2**20:.0f} MB"
    return f"{size / 2**10:.0f} KB"


def file_url(path: Path) -> str:
    """URL Gradio serves a file from the audiobooks folder at (see allowed_paths in main)."""
    return "/file=" + quote(path.resolve().as_posix(), safe="/:")


def notice(kind: str, title: str, body: str = "") -> str:
    """A plain message block. kind: info | warn | error."""
    body_html = f"<p>{body}</p>" if body else ""
    return f'<div class="status {kind}"><strong>{title}</strong>{body_html}</div>'


IDLE_HTML = (
    '<p class="idle">Long books take a while. You can close this tab: the conversion keeps '
    'running, and reopening the page shows where it is.</p>'
)


def render_progress(job: Job) -> str:
    elapsed = fmt_duration(time.time() - job.started)
    stopping = job.stop.is_set()
    label = ("Stopping after the current chunk" if stopping and job.stage == "generating"
             else STAGE_LABELS[job.stage])
    determinate = job.stage == "generating" and job.total > 0
    pct = job.done / job.total * 100 if determinate else 0
    # The page redraws this panel each second, which would restart the CSS
    # animations; a negative delay equal to the clock's place in the 2s
    # cycle (see style.css) keeps them running smoothly across redraws.
    phase = f'style="animation-delay:-{time.time() % 2:.2f}s"'

    if determinate:
        eta = job.eta()
        counts = f"{job.done} of {job.total} chunks"
        timing = f"{elapsed} elapsed" + (f" · about {fmt_duration(eta)} left" if eta else "")
        bar = (f'<div class="bar" role="progressbar" aria-valuemin="0" aria-valuemax="100" '
               f'aria-valuenow="{pct:.0f}"><span style="width:{pct:.1f}%"></span></div>')
        percent = f'<span class="pct">{pct:.0f}%</span>'
    else:
        counts = ""
        timing = f"{elapsed} elapsed"
        bar = f'<div class="bar indeterminate" role="progressbar"><span {phase}></span></div>'
        percent = ""

    return (
        f'<div class="status running">'
        f'<div class="status-top"><span class="pulse" {phase}></span><strong>{label}</strong>{percent}</div>'
        f'{bar}'
        f'<div class="meta"><span>{counts}</span><span>{timing}</span></div>'
        f'<p class="what">{html.escape(job.book)}<br>{html.escape(job.voice)}</p>'
        f'</div>'
    )


def render_player(path: Path) -> str:
    url = file_url(path)
    shown = f"{AUDIOBOOKS_FOLDER}/{path.name}"
    return (
        f'<audio controls preload="none" src="{url}"></audio>'
        f'<div class="actions"><a class="btn-link" href="{url}" download="{html.escape(path.name)}">'
        f'Download</a><span class="path" title="{html.escape(shown)}">{html.escape(shown)}</span></div>'
    )


def render_finished(job: Job) -> str:
    took = fmt_duration((job.ended or time.time()) - job.started)
    if job.stage == "done" and job.output:
        return (f'<div class="status done"><strong>Your audiobook is ready</strong>'
                f'<p>Finished in {took} · {fmt_size(job.output)}</p>{render_player(job.output)}</div>')
    if job.stage == "partial" and job.output:
        retry = ("Convert the same book with the same settings again to retry only those parts."
                 if ENABLE_CACHING else "Caching is off, so another run starts over.")
        return (f'<div class="status warn"><strong>Saved, with {len(job.missing)} missing '
                f'chunk{"s" if len(job.missing) != 1 else ""}</strong>'
                f'<p>The text of the missing parts is in the .MISSING.txt file next to the audiobook. '
                f'{retry}</p>{render_player(job.output)}</div>')
    if job.stage == "stopped":
        kept = (" Finished chunks are saved: convert the same book with the same settings again to carry on."
                if ENABLE_CACHING and job.done else "")
        progress = f"{job.done} of {job.total} chunks were processed." if job.total else ""
        return notice("info", "Stopped", f"{progress}{kept}".strip())
    return notice("error", "The conversion failed",
                  f'{html.escape(job.error)} The terminal and the logs/ folder have the details.')


def render_job(job: Optional[Job]) -> str:
    if job is None:
        return IDLE_HTML
    return render_progress(job) if job.running else render_finished(job)


def button_states(running: bool):
    """(convert button, stop button) updates for a running / idle conversion."""
    if running:
        return gr.update(value="Converting…", interactive=False), gr.update(visible=True)
    return gr.update(value="Convert to audiobook", interactive=True), gr.update(visible=False)


def follow_job():
    """Stream the current job's state to the page until it ends (also used on page load)."""
    while True:
        job = _job
        running = bool(job and job.running)
        convert_btn, stop_btn = button_states(running)
        yield render_job(job), convert_btn, stop_btn
        if not running:
            return
        time.sleep(1)


def start_conversion(file_path, voice_mode, speaker, language, instruct,
                     reference_audio, reference_text, description, device):
    """Validate the form, start a job (unless one is running), then follow it."""
    global _job

    problem = None
    if not file_path:
        problem = "Add a book first: drop a PDF, EPUB, DOCX or TXT file in step 1."
    elif voice_mode == "voice_clone" and not reference_audio:
        problem = "Upload a voice sample in step 2, or switch to another voice type."
    elif voice_mode == "voice_design" and not (description or "").strip():
        problem = "Describe the voice you want in step 2."
    if problem:
        convert_btn, stop_btn = button_states(False)
        yield notice("warn", "Almost there", problem), convert_btn, stop_btn
        return

    with _job_lock:
        if _job is None or not _job.running:
            reference_text = (reference_text or "").strip()
            book_path = copy_book_into_library(file_path)
            if voice_mode == "custom_voice":
                args = dict(speaker=speaker, language=language, instruct=instruct, device=device)
                voice = f"{speaker} · {language}"
            elif voice_mode == "voice_clone":
                args = dict(voice_clone_ref_audio=reference_audio,
                            voice_clone_ref_text=reference_text or None,
                            force_xvector=not reference_text, language=language, device=device)
                voice = f"Cloned from {Path(reference_audio).name} · {language}"
            else:
                args = dict(language=language, voice_description=description, device=device)
                voice = f"Designed voice · {language}"
            _job = Job(book_path.stem, voice)
            threading.Thread(target=run_job, args=(_job, book_path, voice_mode, args),
                             daemon=True, name="conversion").start()
        # else: a job is already running (started from another tab); just follow it
    yield from follow_job()


def stop_conversion():
    """Ask the running job to stop after the current chunk."""
    job = _job
    if job and job.running:
        job.stop.set()


# =============================================================================
# GRADIO INTERFACE
# =============================================================================

HEADER_HTML = """
<header class="app-header">
  <div class="brand">
    <span class="logo" aria-hidden="true">
      <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="2"
           stroke-linecap="round" stroke-linejoin="round"><path d="M3 14h3a2 2 0 0 1 2 2v3a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-7a9 9 0 0 1 18 0v7a2 2 0 0 1-2 2h-1a2 2 0 0 1-2-2v-3a2 2 0 0 1 2-2h3"/></svg>
    </span>
    <div>
      <h1>Audiobook Converter</h1>
      <p>Turn any book into narrated audio, on your own GPU</p>
    </div>
  </div>
  <button id="theme-toggle" type="button" class="icon-btn" aria-label="Switch between light and dark theme"
          title="Switch between light and dark theme">
    <svg class="icon-moon" viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor"
         stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/></svg>
    <svg class="icon-sun" viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor"
         stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2m-7.07-2.93 1.41-1.41m11.32-11.32 1.41-1.41M2 12h2m16 0h2M4.93 4.93l1.41 1.41m11.32 11.32 1.41 1.41"/></svg>
  </button>
</header>
"""


def step_title(number: int, title: str, subtitle: str = "") -> str:
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    return f'<div class="step"><span class="num">{number}</span><div><h2>{title}</h2>{sub}</div></div>'


def settings_html() -> str:
    """Read-only summary of the conversion settings in config.py."""
    rows = [
        ("Chunk size", f"{CHUNK_SIZE_WORDS} words"),
        ("Token budget", f"{FASTER_QWEN_MAX_SEQ_LEN} per chunk"),
        ("Streaming decode", "on" if STREAMING_ENABLED else "off"),
        ("Pause after chunk / paragraph / chapter",
         f"{SENTENCE_PAUSE_MS} / {PARAGRAPH_PAUSE_MS} / {CHAPTER_PAUSE_MS} ms"),
        ("Output format", html.escape(str(AUDIO_FORMAT))),
        ("Resume after a stop", "on" if ENABLE_CACHING else "off"),
    ]
    cells = "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in rows)
    return (f'<p class="hint">These come from <code>config.py</code>. Edit it and restart the app '
            f'to change them.</p><dl class="facts">{cells}</dl>')


def create_interface():
    """Create the Gradio web interface"""

    gpus = list_gpus() or [(DEVICE, DEVICE)]
    default_gpu = DEVICE if DEVICE in [d for _, d in gpus] else gpus[0][1]
    speaker_names = [name for name, _, _ in SPEAKERS]

    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.orange,
        neutral_hue=gr.themes.colors.stone,
        radius_size=gr.themes.sizes.radius_lg,
        font=["Inter", "Segoe UI Variable Text", "Segoe UI", "system-ui", "sans-serif"],
        font_mono=["Cascadia Code", "Consolas", "ui-monospace", "monospace"],
    )
    css = (WEB_DIR / "style.css").read_text(encoding="utf-8")
    head = f"<script>{(WEB_DIR / 'theme.js').read_text(encoding='utf-8')}</script>"

    with gr.Blocks(title="Audiobook Converter", theme=theme, css=css, head=head) as demo:
        gr.HTML(HEADER_HTML, elem_id="app-header")

        with gr.Row(elem_id="layout", equal_height=False):
            # ---- Left: set up the conversion -------------------------------
            with gr.Column(scale=7, elem_id="setup"):
                with gr.Column(elem_classes="card"):
                    gr.HTML(step_title(1, "Choose a book", "PDF, EPUB, DOCX or TXT"))
                    book_file = gr.File(
                        show_label=False,
                        file_types=sorted(BOOK_EXTENSIONS),
                        type="filepath",
                        elem_id="book-file",
                    )

                with gr.Column(elem_classes="card"):
                    gr.HTML(step_title(2, "Choose a voice"))
                    voice_mode = gr.Radio(
                        choices=VOICE_MODES,
                        value="custom_voice",
                        show_label=False,
                        container=False,
                        elem_id="voice-mode",
                    )
                    mode_hint = gr.HTML(f'<p class="hint">{MODE_HINTS["custom_voice"]}</p>')

                    with gr.Row(elem_classes="pair"):
                        speaker = gr.Dropdown(
                            choices=[(f"{n} · {g} · {d}", n) for n, g, d in SPEAKERS],
                            value=CUSTOM_VOICE_SPEAKER if CUSTOM_VOICE_SPEAKER in speaker_names else speaker_names[0],
                            label="Narrator",
                            scale=3,
                        )
                        language = gr.Dropdown(
                            choices=LANGUAGES,
                            value=CUSTOM_VOICE_LANGUAGE,
                            label="Language",
                            scale=2,
                        )

                    with gr.Column(elem_classes="panel") as preset_panel:
                        instruct = gr.Textbox(
                            label="Narration style",
                            info="Tone and pace. Words like “slow, relaxed” slow the narration down.",
                            value=CUSTOM_VOICE_INSTRUCT,
                            lines=2,
                        )

                    with gr.Column(elem_classes="panel", visible=False) as clone_panel:
                        reference_audio = gr.Audio(
                            sources=["upload"],
                            type="filepath",
                            show_label=False,
                            elem_id="voice-sample",
                        )
                        reference_text = gr.Textbox(
                            label="What is said in the sample (optional)",
                            info="The exact words give a closer clone. Leave it empty to match "
                                 "the voice only: quicker, but less accurate.",
                            lines=3,
                        )

                    with gr.Column(elem_classes="panel", visible=False) as design_panel:
                        description = gr.Textbox(
                            label="Voice description",
                            info="For example: a warm, unhurried older woman with a soft British accent.",
                            value=VOICE_DESIGN_DESCRIPTION,
                            lines=3,
                        )

            # ---- Right: run it ---------------------------------------------
            with gr.Column(scale=5, elem_id="run"):
                with gr.Column(elem_classes="card"):
                    gr.HTML(step_title(3, "Convert"))
                    convert_btn = gr.Button("Convert to audiobook", variant="primary", size="lg",
                                            elem_id="convert-btn")
                    stop_btn = gr.Button("Stop", variant="secondary", visible=False, elem_id="stop-btn")
                    status = gr.HTML(IDLE_HTML, elem_id="status")

        with gr.Accordion("Settings", open=False, elem_id="settings"):
            gpu_device = gr.Dropdown(
                choices=gpus,
                value=default_gpu,
                label="Run conversions on",
                info="Numbered as in nvidia-smi. Applies from the next conversion. "
                     "The default is FASTER_QWEN_DEVICE in config.py.",
            )
            gr.HTML(settings_html())

        # =============================================================================
        # EVENT HANDLERS
        # =============================================================================

        def on_voice_mode_change(mode):
            """Show the settings that apply to the selected voice type"""
            return (
                gr.update(visible=mode == "custom_voice"),
                f'<p class="hint">{MODE_HINTS[mode]}</p>',
                gr.update(visible=mode == "custom_voice"),
                gr.update(visible=mode == "voice_clone"),
                gr.update(visible=mode == "voice_design"),
            )

        voice_mode.change(
            fn=on_voice_mode_change,
            inputs=[voice_mode],
            outputs=[speaker, mode_hint, preset_panel, clone_panel, design_panel],
            show_progress="hidden",
        )

        live = [status, convert_btn, stop_btn]
        # No concurrency limit: the job lock keeps it to one conversion, and a
        # second tab should follow the running job, not wait in the queue
        convert_btn.click(
            fn=start_conversion,
            inputs=[book_file, voice_mode, speaker, language, instruct,
                    reference_audio, reference_text, description, gpu_device],
            outputs=live,
            show_progress="hidden",
            concurrency_limit=None,
            trigger_mode="once",
        )
        stop_btn.click(fn=stop_conversion, show_progress="hidden", concurrency_limit=None)
        demo.load(fn=follow_job, outputs=live, show_progress="hidden", concurrency_limit=None)

    return demo


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    """Start the Gradio interface"""
    import argparse

    parser = argparse.ArgumentParser(description="Qwen Audiobook Converter - Web UI")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7861, help="Port to listen on (default: 7861)")
    args = parser.parse_args()

    demo = create_interface()
    demo.queue()
    print(f"\n{'='*70}")
    print("Qwen Audiobook Converter - Web Interface")
    print(f"{'='*70}")
    print(f"Local URL: http://localhost:{args.port}")
    print(f"Network URL: http://{args.host}:{args.port}")
    print(f"{'='*70}\n")

    Path(AUDIOBOOKS_FOLDER).mkdir(exist_ok=True)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=False,
        # lets the page play and download finished audiobooks straight from disk
        allowed_paths=[str(Path(AUDIOBOOKS_FOLDER).resolve())],
    )


if __name__ == "__main__":
    main()
