# syntax=docker/dockerfile:1
#
# ARCHITECTURE-AGNOSTIC ON PURPOSE - there is no `--platform` pin anywhere.
#
# paddlepaddle 3.2.2 publishes both manylinux2014_aarch64 and manylinux1_x86_64
# wheels for cp310, so this one file builds natively for either target:
#
#   local (Apple Silicon):  docker compose up --build          -> linux/arm64
#   AWS ECS Fargate (x86):  docker buildx build --platform linux/amd64 ...
#   AWS ECS Fargate (ARM):  docker buildx build --platform linux/arm64 ...
#
# Same source, same Dockerfile, no emulation in either direction.

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FLAGS_allocator_strategy=auto_growth \
    # PaddleX resolves model weights from the official PaddlePaddle org on
    # Hugging Face. Baidu's bcebos CDN is unreachable from some networks
    # (including the one this was developed on), so we pin the source rather
    # than rely on the default staying put.
    PADDLE_PDX_MODEL_SOURCE=huggingface \
    # The weights are baked in below, so the runtime must never probe a model
    # host. This skips a ~5 s connectivity check on every cold start and keeps
    # the container working with no outbound internet at all.
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

WORKDIR /app

# libgomp1     - OpenMP runtime that paddlepaddle links against
# libglib2.0-0 - required by opencv
# libgl1       - required by opencv-contrib-python, the GUI build. paddlex pins
#                that exact distribution and verifies it BY NAME, so the leaner
#                headless twin cannot be substituted (it breaks PDFReaderBackend).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Dependencies get their own layer, copied before the scripts, so editing a
# script does not trigger a full reinstall of paddlepaddle on every rebuild.
COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

COPY scripts ./scripts
RUN bash scripts/check-opencv.sh

# Non-root at runtime. Created before the warm-up so the model cache lands in
# this user's HOME and is readable by the process that needs it.
RUN useradd --create-home --uid 10001 app && chown -R app:app /app
USER app
ENV HOME=/home/app \
    HF_HOME=/home/app/.cache/huggingface

ARG OCR_LANGS="en,ar"
ARG OCR_CPU_THREADS="4"
ARG DET_MODEL="PP-OCRv5_mobile_det"
ARG REC_MODEL_EN="en_PP-OCRv5_mobile_rec"
ARG REC_MODEL_AR="arabic_PP-OCRv5_mobile_rec"
ENV OCR_LANGS=${OCR_LANGS} \
    OCR_CPU_THREADS=${OCR_CPU_THREADS} \
    DET_MODEL=${DET_MODEL} \
    REC_MODEL_EN=${REC_MODEL_EN} \
    REC_MODEL_AR=${REC_MODEL_AR}

# Caches the weights into the image AND verifies the whole inference path.
# Build fails here if cv2 is broken, an import is missing, or paddle/paddlex
# disagree on a version.
RUN python scripts/warmup.py

# Application code copied LAST so editing the extractor does not invalidate the
# expensive dependency + model layers above. Rebuilds after a code change take
# seconds, not minutes.
COPY --chown=app:app app ./app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=6s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
