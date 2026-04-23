from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel

from tts import generate_audio, load_model


class TTSRequest(BaseModel):
    text: str


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