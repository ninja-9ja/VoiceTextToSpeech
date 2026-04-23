import io
import os

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

MODEL_PATH = os.getenv("QWEN_MODEL_PATH", "/models/Qwen3-TTS-12Hz-0.6B-CustomVoice")
SPEAKER = os.getenv("QWEN_SPEAKER", "Ryan")


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


def generate_audio(model: Qwen3TTSModel, text: str) -> bytes:
    wavs, sr = model.generate_custom_voice(
        text=text,
        language="English",
        speaker=SPEAKER,
    )

    buffer = io.BytesIO()
    sf.write(buffer, wavs[0], sr, format="WAV")
    buffer.seek(0)
    return buffer.getvalue()