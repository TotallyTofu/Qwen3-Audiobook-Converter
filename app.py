#!/usr/bin/env python3
"""
Gradio Web Interface for Qwen Audiobook Converter
Provides a user-friendly web UI for converting books to audiobooks using faster-qwen3-tts
"""

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import numpy as np
import torch
from faster_qwen3_tts import FasterQwen3TTS
from pydub import AudioSegment

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from audiobook_converter import (
    QwenAudiobookConverter,
    CUSTOM_VOICE_SPEAKER,
    CUSTOM_VOICE_LANGUAGE,
    CUSTOM_VOICE_INSTRUCT,
    VOICE_CLONE_LANGUAGE,
    VOICE_CLONE_USE_XVECTOR_ONLY,
    VOICE_CLONE_SEED,
    BOOKS_FOLDER,
    AUDIOBOOKS_FOLDER,
    CHUNK_SIZE_WORDS,
    STREAMING_ENABLED,
    STREAMING_CHUNK_SIZE,
    AUDIO_FORMAT,
)

# =============================================================================
# GLOBAL STATE
# =============================================================================

# Singleton model instance to avoid reloading
_model_instance: Optional[FasterQwen3TTS] = None
_voice_mode = "custom_voice"  # custom_voice, voice_clone, voice_design
_reference_audio_path: Optional[str] = None
_reference_text: str = ""
_speaker_embedding = None


def get_or_init_model(voice_mode: str) -> Tuple[Optional[FasterQwen3TTS], Optional[str]]:
    """Get existing model instance or initialize a new one. Returns (model, error_msg)"""
    global _model_instance, _voice_mode
    
    if voice_mode != _voice_mode:
        # Reload model for different voice mode
        _voice_mode = voice_mode
        _model_instance = None  # Force fresh load with correct model for new mode
        
    if _model_instance is not None:
        return _model_instance, None
    
    try:
        print("[INFO] Loading faster-qwen3-tts model...")
        model_id = (
            "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice" 
            if voice_mode == "custom_voice" 
            else "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        )
        
        _model_instance = FasterQwen3TTS.from_pretrained(
            model_id,
            device="cuda",
            dtype=torch.bfloat16,
        )
        print(f"[OK] Model loaded successfully (sample rate: {_model_instance.sample_rate} Hz)")
        return _model_instance, None
    except Exception as e:
        error_msg = f"Failed to load model: {str(e)}"
        print(f"[ERROR] {error_msg}")
        import traceback
        traceback.print_exc()
        return None, error_msg


def generate_custom_voice_audio(text: str, speaker: str, language: str, instruct: str) -> Tuple[int, np.ndarray]:
    """Generate audio using CustomVoice mode"""
    global _model_instance
    
    model, err = get_or_init_model("custom_voice")
    if err:
        raise gr.Error(err)
    
    try:
        result = model.generate_custom_voice_streaming(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct,
            chunk_size=STREAMING_CHUNK_SIZE,
        )
        
        all_chunks = []
        sr = model.sample_rate
        for audio_chunk, chunk_sr, timing in result:
            all_chunks.append(audio_chunk)
            sr = chunk_sr
        
        if len(all_chunks) == 1:
            audio = all_chunks[0]
        else:
            audio = np.concatenate(all_chunks)
        
        return (sr, audio)
    except Exception as e:
        raise gr.Error(f"Generation failed: {str(e)}")


def voice_clone_from_file(reference_audio: Optional[str], reference_text: str, text: str, 
                          language: str, use_xvector: bool) -> Tuple[int, np.ndarray]:
    """Generate audio using Voice Clone mode"""
    global _model_instance, _reference_audio_path, _reference_text, _speaker_embedding
    
    model, err = get_or_init_model("voice_clone")
    if err:
        raise gr.Error(err)
    
    try:
        ref_audio = reference_audio if reference_audio else _reference_audio_path
        if not ref_audio:
            raise gr.Error("Please upload a reference audio file")
        
        if not os.path.exists(ref_audio):
            raise gr.Error(f"Reference audio not found: {ref_audio}")
        
        # Use provided text or cached reference text
        ref_text = reference_text if reference_text else _reference_text
        
        use_xvec = use_xvector or (_speaker_embedding is not None)
        
        if use_xvec and _speaker_embedding is not None:
            result = model.generate_voice_clone_streaming(
                text=text,
                language=language,
                ref_audio=ref_audio,
                ref_text="",
                xvec_only=True,
                voice_clone_prompt={"ref_spk_embedding": [_speaker_embedding]},
                chunk_size=STREAMING_CHUNK_SIZE,
            )
        elif use_xvec:
            result = model.generate_voice_clone_streaming(
                text=text,
                language=language,
                ref_audio=ref_audio,
                ref_text="",
                xvec_only=True,
                chunk_size=STREAMING_CHUNK_SIZE,
            )
        else:
            if not ref_text:
                raise gr.Error("Please provide the transcript of your reference audio when using ICL mode")
            result = model.generate_voice_clone_streaming(
                text=text,
                language=language,
                ref_audio=ref_audio,
                ref_text=ref_text,
                xvec_only=False,
                append_silence=True,
                chunk_size=STREAMING_CHUNK_SIZE,
            )
        
        # Extract speaker embedding if xvector mode for future reuse
        if use_xvector:
            try:
                prompt_items = model.model.create_voice_clone_prompt(
                    ref_audio=ref_audio,
                    ref_text="",
                    x_vector_only_mode=True,
                )
                _speaker_embedding = prompt_items[0].ref_spk_embedding
            except Exception:
                pass  # Non-critical
        
        all_chunks = []
        sr = model.sample_rate
        for audio_chunk, chunk_sr, timing in result:
            all_chunks.append(audio_chunk)
            sr = chunk_sr
        
        if len(all_chunks) == 1:
            audio = all_chunks[0]
        else:
            audio = np.concatenate(all_chunks)
        
        return (sr, audio)
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"Generation failed: {str(e)}")


def voice_design_audio(text: str, description: str, language: str) -> Tuple[int, np.ndarray]:
    """Generate audio using VoiceDesign mode"""
    global _model_instance
    
    model, err = get_or_init_model("voice_design")
    if err:
        raise gr.Error(err)
    
    try:
        result = model.generate_voice_design_streaming(
            text=text,
            language=language,
            instruct=description,
            chunk_size=STREAMING_CHUNK_SIZE,
        )
        
        all_chunks = []
        sr = model.sample_rate
        for audio_chunk, chunk_sr, timing in result:
            all_chunks.append(audio_chunk)
            sr = chunk_sr
        
        if len(all_chunks) == 1:
            audio = all_chunks[0]
        else:
            audio = np.concatenate(all_chunks)
        
        return (sr, audio)
    except Exception as e:
        raise gr.Error(f"Generation failed: {str(e)}")


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
    
    try:
        print(f"[INFO] Converting {dest_path.name}...")
        
        # Create converter instance
        if voice_mode == "custom_voice":
            converter = QwenAudiobookConverter(voice_mode=voice_mode)
        elif voice_mode == "voice_clone":
            _reference_text = reference_text
            # Only force xvector if checkbox is checked AND no ref_text provided
            force_xvec = use_xvector and not reference_text
            converter = QwenAudiobookConverter(
                voice_mode=voice_mode,
                voice_clone_ref_audio=_reference_audio_path,
                voice_clone_ref_text=reference_text if reference_text else None,
                force_xvector=force_xvec
            )
        elif voice_mode == "voice_design":
            converter = QwenAudiobookConverter(voice_mode=voice_mode)
        
        # Convert the book
        success = converter.convert_book(dest_path)
        
        if success:
            output_path = Path(AUDIOBOOKS_FOLDER) / f"{dest_path.stem}.{AUDIO_FORMAT}"
            return (str(output_path), f"Successfully converted to audiobook!")
        else:
            raise gr.Error("Conversion failed - check logs for details")
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Conversion error: {str(e)}")


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

        Convert text to audiobooks using **faster-qwen3-tts** with CUDA graph optimization.
        
        Requires NVIDIA GPU with CUDA and PyTorch >= 2.5.1 (model auto-downloads on first use).
        """)
        
        with gr.Tabs():
            # Tab 1: Quick Text-to-Speech
            with gr.Tab("🎙️ Text to Speech"):
                with gr.Row():
                    with gr.Column(scale=2):
                        text_input = gr.Textbox(
                            label="Input Text",
                            placeholder="Enter the text you want to convert to speech...",
                            lines=5,
                            value="Hello! Welcome to the Qwen Audiobook Converter."
                        )
                        voice_mode_tts = gr.Radio(
                            choices=["custom_voice", "voice_clone", "voice_design"],
                            value="custom_voice",
                            label="Voice Mode"
                        )
                        
                        with gr.Group():
                            speaker_select = gr.Dropdown(
                                choices=speakers,
                                value=CUSTOM_VOICE_SPEAKER,
                                label="Speaker (Custom Voice)"
                            )
                            language_select_tts = gr.Dropdown(
                                choices=languages,
                                value=CUSTOM_VOICE_LANGUAGE,
                                label="Language"
                            )
                            instruct_input = gr.Textbox(
                                label="Style Instruction (Optional)",
                                placeholder="e.g., Speak naturally and clearly, as if reading a book.",
                                value=CUSTOM_VOICE_INSTRUCT
                            )
                        
                        with gr.Group(visible=False) as voice_clone_group:
                            ref_audio = gr.Audio(
                                sources=["upload", "microphone"],
                                type="filepath",
                                label="Reference Audio (for Voice Clone)"
                            )
                            ref_text_input = gr.Textbox(
                                label="Reference Audio Transcript (ICL Mode)",
                                placeholder="Transcript of the reference audio...",
                                lines=3
                            )
                            use_xvector_cb = gr.Checkbox(
                                label="Use XVector Mode (no transcript needed, slightly lower quality)",
                                value=False
                            )
                        
                        with gr.Group(visible=False) as voice_design_group:
                            design_desc_input = gr.Textbox(
                                label="Voice Description",
                                placeholder='e.g., Warm, confident narrator with slight British accent',
                                value="Speak in a clear, professional narrator voice suitable for reading audiobooks."
                            )
                    
                    with gr.Column(scale=1):
                        output_audio = gr.Audio(
                            label="Generated Audio",
                            type="numpy"
                        )
                        generate_btn = gr.Button("🎵 Generate Speech", variant="primary")
                        status_text = gr.Textbox(
                            label="Status",
                            interactive=False
                        )
            
            # Tab 2: Book Conversion
            with gr.Tab("📚 Book Converter"):
                gr.Markdown("""
                Upload a book file (PDF, EPUB, DOCX, TXT) to convert it into an audiobook.
                
                Processing time depends on book length. Each ~1200 word chunk takes ~30 seconds on RTX 4090.
                """)
                
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
                        convert_speaker = gr.Dropdown(
                            choices=speakers,
                            value=CUSTOM_VOICE_SPEAKER,
                            label="Speaker (Custom Voice)"
                        )
                        convert_language = gr.Dropdown(
                            choices=languages,
                            value=CUSTOM_VOICE_LANGUAGE,
                            label="Language"
                        )
                        
                        with gr.Group():
                            convert_instruct = gr.Textbox(
                                label="Style Instruction",
                                value=CUSTOM_VOICE_INSTRUCT
                            )
                        
                        with gr.Group(visible=False) as book_clone_group:
                            book_ref_audio = gr.Audio(
                                sources=["upload"],
                                type="filepath",
                                label="Reference Audio"
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
                                value="Speak in a clear, professional narrator voice suitable for reading audiobooks."
                            )
                    
                    with gr.Column(scale=1):
                        convert_btn = gr.Button("📖 Convert Book to Audiobook", variant="primary")
                        output_audio_book = gr.Audio(
                            label="Generated Audiobook",
                            type="filepath"
                        )
                        convert_status = gr.Textbox(label="Conversion Status", interactive=False)
            
            # Tab 3: Settings
            with gr.Tab("⚙️ Settings"):
                gr.Markdown("### Model & Performance Settings")
                
                streaming_enabled = gr.Checkbox(
                    label="Enable Streaming (faster first audio, slightly lower quality)",
                    value=True
                )
                streaming_chunk_size = gr.Slider(
                    minimum=2,
                    maximum=16,
                    value=STREAMING_CHUNK_SIZE,
                    step=1,
                    label="Streaming Chunk Size (2=min latency, 16=max throughput)"
                )
                chunk_size_words = gr.Slider(
                    minimum=500,
                    maximum=3000,
                    value=CHUNK_SIZE_WORDS,
                    step=100,
                    label="Chunk Size (words per TTS segment)"
                )
                
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
            """Show/hide relevant groups based on voice mode"""
            clone_visibility = gr.update(visible=voice_mode == "voice_clone")
            design_visibility = gr.update(visible=voice_mode == "voice_design")
            return clone_visibility, design_visibility
        
        voice_mode_tts.change(
            fn=on_voice_mode_change,
            inputs=[voice_mode_tts],
            outputs=[voice_clone_group, voice_design_group]
        )
        
        convert_voice_mode.change(
            fn=on_voice_mode_change,
            inputs=[convert_voice_mode],
            outputs=[book_clone_group, book_design_group]
        )
        
        def generate_speech(text, voice_mode, speaker, language, instruct, 
                           reference_audio, reference_text, use_xvector, description):
            """Handle text-to-speech generation"""
            try:
                status = "Generating audio..."
                
                if voice_mode == "custom_voice":
                    sr, audio = generate_custom_voice_audio(text, speaker, language, instruct)
                elif voice_mode == "voice_clone":
                    sr, audio = voice_clone_from_file(reference_audio, reference_text, text, language, use_xvector)
                elif voice_mode == "voice_design":
                    sr, audio = voice_design_audio(text, description, language)
                else:
                    raise ValueError(f"Unknown voice mode: {voice_mode}")
                
                duration = len(audio) / sr if sr else 0
                status = f"✅ Generated {duration:.1f}s of audio"
                
                return (sr, audio), status
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                return None, f"❌ Error: {str(e)}"
        
        def convert_book(file_path, voice_mode, speaker, language, instruct,
                        reference_audio, reference_text, use_xvector, description):
            """Handle book conversion"""
            if not file_path:
                return None, "❌ Please upload a book file"
            
            try:
                status = "🔄 Converting book... This may take several minutes."
                
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
        
        generate_btn.click(
            fn=generate_speech,
            inputs=[text_input, voice_mode_tts, speaker_select, language_select_tts, 
                   instruct_input, ref_audio, ref_text_input, use_xvector_cb, design_desc_input],
            outputs=[output_audio, status_text]
        )
        
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
