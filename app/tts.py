import io
import os

import soundfile as sf
from qwen_tts import Qwen3TTSModel

MODEL_PATH = os.getenv("QWEN_MODEL_PATH", "/models/Qwen3-TTS-12Hz-0.6B-CustomVoice")
SPEAKER = os.getenv("QWEN_SPEAKER", "Serena")
REF_AUDIO = os.getenv("QWEN_SPEAKER", "Serena")
SPEAKER = os.getenv("QWEN_SPEAKER", "Serena")


def load_model():
    if not os.path.isdir(MODEL_PATH):
        raise RuntimeError(f"Model path does not exist: {MODEL_PATH}")

    print(f"[startup] Loading model from: {MODEL_PATH}")

    model = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        device_map="cpu",
        dtype="float32",
        local_files_only=True,
    )

    print("[startup] Model loaded successfully")
    return model


def generate_audio(model: Qwen3TTSModel, text: str) -> bytes:
    wavs, sr = model.generate_custom_voice(
        text=text,
        language="Russian",
        speaker=SPEAKER,
    )
    buffer = io.BytesIO()
    sf.write(buffer, wavs[0], sr, format="WAV")
    buffer.seek(0)
    return buffer.getvalue()

def clone_audio(model: Qwen3TTSModel, text: str, ref_audio: str, ref_text: str) -> bytes:
    wavs, sr = model.generate_voice_clone(
    text=text,
    language="Russian",
    ref_audio=ref_audio,
    ref_text=ref_text,
    )
    buffer = io.BytesIO()
    sf.write(buffer, wavs[0], sr, format="WAV")
    buffer.seek(0)
    return buffer.getvalue()