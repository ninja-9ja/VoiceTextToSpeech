FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    sox \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY app/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    pip install --no-cache-dir "huggingface_hub[cli]"

ARG HF_TOKEN=""
RUN if [ -n "$HF_TOKEN" ]; then huggingface-cli login --token "$HF_TOKEN"; fi

# В образ скачиваем tokenizer и Base-модель для voice cloning
RUN mkdir -p /models && \
    huggingface-cli download Qwen/Qwen3-TTS-Tokenizer-12Hz \
      --local-dir /models/Qwen3-TTS-Tokenizer-12Hz && \
    # huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    #   --local-dir /models/Qwen3-TTS-12Hz-1.7B-Base
    huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base \
      --local-dir /models/Qwen3-TTS-12Hz-0.6B-Base

COPY app/ /app/

ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1
# ENV QWEN_MODEL_PATH=/models/Qwen3-TTS-12Hz-1.7B-Base
ENV QWEN_MODEL_PATH=/models/Qwen3-TTS-12Hz-0.6B-Base

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-level", "debug"]