from fastapi import FastAPI, Request
from tts import generate_audio


app = FastAPI()
@app.post("/tts")
async def tts_endpoint(request: Request):
    data = await request.json()
    text = data.get("text")
    audio_bytes = await generate_audio(text)
    return Response(content=audio_bytes, media_type="audio/wav")