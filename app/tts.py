import os
import io
import tempfile
from pathlib import Path

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

MODEL_PATH = os.getenv("QWEN_MODEL_PATH", "/models/Qwen3-TTS-12Hz-0.6B-Base")


def load_model():
    if not os.path.isdir(MODEL_PATH):
        raise RuntimeError(f"Model path does not exist: {MODEL_PATH}")

    print(f"[startup] Loading model from: {MODEL_PATH}")

    model = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        device_map="cpu",
        dtype=torch.float32,
        local_files_only=True,
    )

    print("[startup] Model loaded successfully")
    return model


def generate_audio_clone(model, text: str, ref_audio_bytes: bytes, ref_filename: str, ref_text: str) -> bytes:
    suffix = Path(ref_filename).suffix if ref_filename else ".wav"
    if not suffix:
        suffix = ".wav"

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(ref_audio_bytes)
            tmp_path = tmp.name

        wavs, sr = model.generate_voice_clone(
            text=text,
            language="Russian",
            ref_audio=tmp_path,
            ref_text=ref_text,
        )

        buffer = io.BytesIO()
        sf.write(buffer, wavs[0], sr, format="WAV")
        buffer.seek(0)
        return buffer.getvalue()

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)