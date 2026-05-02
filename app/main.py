from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel

from tts import clone_audio, generate_audio, load_model


class TTSRequest(BaseModel):
    text: str

class CloneTTSRequest(BaseModel):
    text: str
    ref_audio: str
    ref_text: str 

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.tts_model = load_model()
    yield
    app.state.tts_model = None


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/tts", response_class=Response)
def tts_endpoint(payload: TTSRequest, request: Request):
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is empty")

    audio_bytes = generate_audio(request.app.state.tts_model, text)
    return Response(content=audio_bytes, media_type="audio/wav")

@app.post("/clonetts", response_class=Response)
def tts_endpoint(payload: CloneTTSRequest, request: Request):
    text = payload.text.strip()
    ref_audio = payload.ref_audio.strip()
    ref_text = payload.ref_text.strip()
    print(text, ref_audio, ref_text)
    if not text:
        raise HTTPException(status_code=400, detail="text is empty")

    audio_bytes = clone_audio(request.app.state.tts_model, text, ref_audio, ref_text)
    return Response(content=audio_bytes, media_type="audio/wav")