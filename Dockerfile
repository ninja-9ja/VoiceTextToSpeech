#FROM qwen3-tts-fastapi:latest

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
	sox \
	curl \
	wget \
	unzip \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY app/requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app .

RUN pip install huggingface_hub && \
    python -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='Qwen/Qwen3-TTS-12Hz-1.7B-Base', local_dir='model')"

ENV TRANSFORMERS_OFFLINE=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]