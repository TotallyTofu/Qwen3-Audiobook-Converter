#!/usr/bin/env python3
"""
Gradio Web Interface for Qwen Audiobook Converter
Provides a user-friendly web UI for converting books to audiobooks using faster-qwen3-tts
"""

import gc
import os
import shutil
import sys
from pathlib import Path
from typing import Optional, Tuple

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
    CHUNK_SIZE_WORDS,
    STREAMING_ENABLED,
    AUDIO_FORMAT,
    FASTER_QWEN_MAX_SEQ_LEN,
    SENTENCE_PAUSE_MS,
    PARAGRAPH_PAUSE_MS,
    CHAPTER_PAUSE_MS,
    VOICE_DESIGN_DESCRIPTION,
)

# =============================================================================
# GLOBAL STATE
# =============================================================================

_reference_audio_path: Optional[str] = None
_reference_text: str = ""


def convert_book_to_audio(file_path: str, voice_mode: str, speaker: str, language: str,
                          instruct: str, reference_audio: Optional[str], reference_text: str,
                          use_xvector: bool, description: str) -> Tuple[str, str]:
    """Convert a single book file to audiobook using the converter"""
    global _reference_audio_path, _reference_text

    if not file_path:
        raise gr.Error("Please upload a book file")

    # Save uploaded file to book_to_convert folder
    if file_path and hasattr(file_path, 'name'):
        src = file_path.name if hasattr(file_path, 'name') else file_path
    else:
        src = str(file_path)

    dest_path = Path(BOOKS_FOLDER) / Path(src).name
    shutil.copy2(src, dest_path)

    # Update reference audio path
    if reference_audio and hasattr(reference_audio, 'path'):
        _reference_audio_path = reference_audio.path
    elif reference_audio:
        _reference_audio_path = reference_audio

    converter = None
    try:
        print(f"[INFO] Converting {dest_path.name}...")

        # Create converter instance
        if voice_mode == "custom_voice":
            converter = QwenAudiobookConverter(
                voice_mode=voice_mode,
                speaker=speaker,
                language=language,
                instruct=instruct,
            )
        elif voice_mode == "voice_clone":
            _reference_text = reference_text
            # Only force xvector if checkbox is checked AND no ref_text provided
            force_xvec = use_xvector and not reference_text
            converter = QwenAudiobookConverter(
                voice_mode=voice_mode,
                voice_clone_ref_audio=_reference_audio_path,
                voice_clone_ref_text=reference_text if reference_text else None,
                force_xvector=force_xvec,
                language=language,
            )
        elif voice_mode == "voice_design":
            converter = QwenAudiobookConverter(
                voice_mode=voice_mode,
                language=language,
                voice_description=description,
            )
        else:
            raise gr.Error(f"Unknown voice mode: {voice_mode}")

        # Convert the book
        success = converter.convert_book(dest_path)

        if success:
            return (str(converter.last_output_path), "Successfully converted to audiobook!")
        if converter.last_output_path is not None:
            missing = converter.last_missing_chunks
            return (str(converter.last_output_path),
                    f"Audiobook saved, but {len(missing)} chunk(s) could not be generated: {missing}. "
                    f"Convert the same book again to retry only those chunks.")
        raise gr.Error("Conversion failed - check logs for details")

    except gr.Error:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Conversion error: {str(e)}")
    finally:
        # Free the model's GPU memory before the next conversion loads one
        del converter
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# =============================================================================
# GRADIO INTERFACE
# =============================================================================

def create_interface():
    """Create the Gradio web interface"""

    speakers = ["Ryan", "Serena", "Aiden", "Dylan", "Eric", "Ono_anna", "Sohee", "Uncle_fu", "Vivian"]
    languages = ["Auto", "English", "Chinese", "Japanese", "Korean", "French", "German",
                 "Spanish", "Portuguese", "Russian"]

    with gr.Blocks(title="Qwen Audiobook Converter", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎧 Qwen Audiobook Converter

        Convert books to audiobooks using **faster-qwen3-tts** with CUDA graph optimization.

        Requires NVIDIA GPU with CUDA and PyTorch >= 2.5.1 (model auto-downloads on first use).
        """)

        with gr.Tabs():
            # Tab 1: Book Conversion
            with gr.Tab("📚 Book Converter"):
                gr.Markdown("""
                Upload a book file (PDF, EPUB, DOCX, TXT) to convert it into an audiobook.

                Processing time depends on book length: the book is generated in chunks of up to
                {CHUNK_SIZE_WORDS} words, each taking roughly 20-30 seconds on an RTX 4090.
                """.format(CHUNK_SIZE_WORDS=CHUNK_SIZE_WORDS))

                with gr.Row():
                    with gr.Column(scale=2):
                        book_file = gr.File(
                            label="Upload Book",
                            file_types=[".pdf", ".epub", ".docx", ".txt"],
                            type="filepath"
                        )

                        convert_voice_mode = gr.Radio(
                            choices=["custom_voice", "voice_clone", "voice_design"],
                            value="custom_voice",
                            label="Voice Mode"
                        )
                        convert_language = gr.Dropdown(
                            choices=languages,
                            value=CUSTOM_VOICE_LANGUAGE,
                            label="Language"
                        )

                        with gr.Group() as book_custom_group:
                            convert_speaker = gr.Dropdown(
                                choices=speakers,
                                value=CUSTOM_VOICE_SPEAKER,
                                label="Speaker"
                            )
                            convert_instruct = gr.Textbox(
                                label="Style Instruction (also sets the pace)",
                                value=CUSTOM_VOICE_INSTRUCT
                            )

                        with gr.Group(visible=False) as book_clone_group:
                            book_ref_audio = gr.Audio(
                                sources=["upload"],
                                type="filepath",
                                label="Reference Audio (the clone also copies its pace)"
                            )
                            book_ref_text = gr.Textbox(
                                label="Reference Transcript (ICL Mode)",
                                lines=3
                            )
                            book_use_xvector = gr.Checkbox(
                                label="Use XVector Mode (no transcript needed)",
                                value=False
                            )

                        with gr.Group(visible=False) as book_design_group:
                            book_design_desc = gr.Textbox(
                                label="Voice Description",
                                value=VOICE_DESIGN_DESCRIPTION
                            )

                    with gr.Column(scale=1):
                        convert_btn = gr.Button("📖 Convert Book to Audiobook", variant="primary")
                        output_audio_book = gr.Audio(
                            label="Generated Audiobook",
                            type="filepath"
                        )
                        convert_status = gr.Textbox(label="Conversion Status", interactive=False)

            # Tab 2: Settings
            with gr.Tab("⚙️ Settings"):
                gr.Markdown("### Book Conversion Settings")
                gr.Markdown(f"""
                These come from `config.py`; edit it and restart the app to change them.

                | Setting | Value |
                |---------|-------|
                | Chunk size | {CHUNK_SIZE_WORDS} words |
                | Token budget (`FASTER_QWEN_MAX_SEQ_LEN`) | {FASTER_QWEN_MAX_SEQ_LEN} |
                | Streaming decode | {'on' if STREAMING_ENABLED else 'off'} |
                | Pause after chunk / paragraph / chapter | {SENTENCE_PAUSE_MS} / {PARAGRAPH_PAUSE_MS} / {CHAPTER_PAUSE_MS} ms |
                | Output format | {AUDIO_FORMAT} |
                """)

                gr.Markdown("### Speaker Presets")
                gr.Markdown("""
                | Speaker | Type | Description |
                |---------|------|-------------|
                | Ryan | Male | Clear and professional (default) |
                | Serena | Female | Warm and friendly |
                | Aiden | Male | Energetic and engaging |
                | Dylan | Male | Calm and soothing |
                | Eric | Male | Expressive and dynamic |
                | Ono_anna | Female | Japanese accent support |
                | Sohee | Female | Korean accent support |
                | Uncle_fu | Male | Chinese accent support |
                | Vivian | Female | Versatile |
                """)

        # =============================================================================
        # EVENT HANDLERS
        # =============================================================================

        def on_voice_mode_change(voice_mode):
            """Show the settings that apply to the selected voice mode"""
            return (
                gr.update(visible=voice_mode == "custom_voice"),
                gr.update(visible=voice_mode == "voice_clone"),
                gr.update(visible=voice_mode == "voice_design"),
            )

        convert_voice_mode.change(
            fn=on_voice_mode_change,
            inputs=[convert_voice_mode],
            outputs=[book_custom_group, book_clone_group, book_design_group]
        )

        def convert_book(file_path, voice_mode, speaker, language, instruct,
                        reference_audio, reference_text, use_xvector, description):
            """Handle book conversion"""
            if not file_path:
                return None, "❌ Please upload a book file"

            try:
                output_path, msg = convert_book_to_audio(
                    file_path, voice_mode, speaker, language, instruct,
                    reference_audio, reference_text, use_xvector, description
                )

                status = f"✅ {msg}"

                if os.path.exists(output_path):
                    return output_path, status
                else:
                    return None, f"❌ Output file not found: {output_path}"

            except Exception as e:
                import traceback
                traceback.print_exc()
                return None, f"❌ Conversion error: {str(e)}"

        convert_btn.click(
            fn=convert_book,
            inputs=[book_file, convert_voice_mode, convert_speaker, convert_language,
                   convert_instruct, book_ref_audio, book_ref_text, book_use_xvector, book_design_desc],
            outputs=[output_audio_book, convert_status]
        )

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

    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=False,
    )


if __name__ == "__main__":
    main()
