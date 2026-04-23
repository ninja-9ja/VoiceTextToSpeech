from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request, Response

from tts import load_model, generate_audio_clone


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.tts_model = load_model()
    yield
    app.state.tts_model = None


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/tts-clone", response_class=Response)
async def tts_clone_endpoint(
    request: Request,
    text: str = Form(...),
    ref_text: str = Form(...),
    ref_audio: UploadFile = File(...),
):
    text = text.strip()
    ref_text = ref_text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="text is empty")

    if not ref_text:
        raise HTTPException(status_code=400, detail="ref_text is empty")

    audio_bytes = await ref_audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="ref_audio is empty")

    output_wav = generate_audio_clone(
        model=request.app.state.tts_model,
        text=text,
        ref_audio_bytes=audio_bytes,
        ref_filename=ref_audio.filename or "reference.wav",
        ref_text=ref_text,
    )

    return Response(content=output_wav, media_type="audio/wav")