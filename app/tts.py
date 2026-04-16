import torch
from qwen_tts import Qwen3TTSModel
import soundfile as sf
import io


async def generate_audio(text: str):
    model = Qwen3TTSModel.from_pretrained(
        # /app/model/Qwen3-TTS-12Hz-1.7B-Base
        "model",
        device_map="cpu",
        dtype=torch.float32,
        local_files_only=True
        )
    wavs, sr = model.generate_voice_clone(
        text=text,
        language="English",
    )
    buffer = io.BytesIO()
    sf.write(buffer, wavs[0], sr, format="WAV")
    buffer.seek(0)
    return buffer.getvalue()