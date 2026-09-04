# UAE Mulkiya Reader

Local + AWS OCR service that extracts 14 structured fields from UAE vehicle
registration cards (Mulkiya). FastAPI + PaddleOCR (PP-OCRv5) + OpenCV, no paid
OCR APIs, no macOS-only code.

Measured on a real Dubai RTA card: **14/14 fields in ~3.5 s**, native arm64.

## Run it

Requires Docker Desktop. On Apple Silicon nothing special is needed — there is
no `platform:` pin anywhere, so the image builds native `linux/arm64`.

```bash
docker compose up --build
```

Then open <http://localhost:8000>.

First build takes several minutes: it installs PaddlePaddle and bakes the
PP-OCRv5 weights into the image. After that, code changes rebuild in seconds
because `COPY app ./app` is the last layer.

## API

| Method | Path | Body |
|---|---|---|
| `POST` | `/api/v1/mulkiya/extract` | multipart, field `file` |
| `POST` | `/api/v1/mulkiya/extract/batch` | multipart, field `files` (≤20) |
| `GET` | `/health` | — |

JPG, JPEG, PNG, WEBP and PDF are accepted.

```json
{
  "success": true,
  "filename": "mulkiya.jpg",
  "processing_time_ms": 3456,
  "data": {
    "plate_source": "Dubai",       "plate_category": "Private",
    "plate_code": "AA",            "plate_number": "88271",
    "vin": "SAL1P9EU2RA165631",
    "make": "RANGE ROVER",         "model": "SPORT",
    "year": 2024,                  "color": "Grey",
    "insurance_company": "ادمجي انشورنس كومباني ليمتد",
    "policy_number": "2510061602",
    "insurance_expiry": "2026-07-19",
    "registration_expiry": "2026-06-19",
    "registration_issuance": "2025-04-28"
  },
  "confidence": { "vin": 0.99, "...": null },
  "warnings": []
}
```

A field that cannot be read reliably is `null` and raises a warning. Nothing is
ever guessed. `processing_time_ms` covers decode + preprocess + OCR + extraction
only, never model startup.

In a batch, one unreadable file returns `success: false` with an `error` for
that file and does not affect the others.

## Debug mode

```bash
INCLUDE_RAW_OCR=true docker compose up
```

The response then also carries `raw_ocr` (every detected line with its box,
language and confidence) and `timings`. The web UI grows a **Raw OCR** button
showing the same table. This is how you diagnose a missed field: look at what
the OCR actually read and where. Off by default.

## Extraction strategy

Fields are found by their labels and geometry, not by scanning text globally:

1. Detection runs **once** per page; both recognisers read the same crops.
2. Values are located on the **same row** as their label, English labels to the
   left of the value and Arabic to the right.
3. When OCR merges a label and its value into one box, the label is stripped
   rather than the box discarded.
4. Dictionaries are matched after folding Arabic letter variants (أ إ آ → ا,
   ة → ه, ى → ي), stripping tashkeel, and converting Arabic-Indic digits.
5. Closed vocabularies (the seven emirates, colours) fall back to fuzzy
   matching next to their label only, and only when exactly one candidate wins.

Arabic text direction is **calibrated per document**: the extractor counts how
many known labels match as-read versus reversed. PaddleX carries a python-bidi
call for the Arabic models that the shipped version does not apply, and a
library bump could change that silently — measuring avoids hard-coding either.

VIN handling is deliberately conservative. A clean read is returned untouched.
`I`, `O` and `Q` cannot appear in a VIN, so those are corrected — and the
correction is reported in `warnings`. Ambiguous pairs (`S`/`5`, `B`/`8`) are
never rewritten, because both characters are legal and a "fix" would be
invention.

## Performance notes

Defaults come from measurement on a real card, not intuition:

| Setting | Finding |
|---|---|
| `OCR_REC_BATCH_SIZE=1` | Batching is a loss. PaddleX pads every crop in a batch to the widest aspect ratio in it; 1 beat 2/4/8/16/32 by ~40%. |
| `OCR_CPU_THREADS=4` | 4 was ~2x faster than 8 or 10 on a 10-core host. |
| Arabic as generalist | Its charset covers latin and digits, so it reads the whole card. English is spent only on uncertain latin boxes and anything VIN-shaped: 6.0 s → 3.5 s at identical accuracy. |
| `OCR_REC_INPUT_SHAPE` empty | A fixed `3,48,320` was 9% faster and lost 2 of the 15 fields then extracted. Not worth it. |
| `ENABLE_MKLDNN` auto | oneDNN is an x86 path; measured no effect on arm64. Expect it to help on Fargate x86. |

The single largest win was a bug, not a tunable: `dict.setdefault(k, f())`
evaluates `f()` even when `k` is present, so an extra full recognition pass per
language ran on every crop and was thrown away — about 6 s of the original 9 s.

## Tests

`app.extractor` imports nothing from PaddleOCR, so extraction logic is tested
without a container or a model download:

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest tests/ -q
```

45 tests, ~0.3 s. They include a fixture tracing a real RTA card's layout and
coordinates, plus a regression test for every bug found during development.

## Privacy

Uploads are held in memory and discarded. Starlette normally spools uploads
over 1 MB to a temp file on disk — most Mulkiya photos exceed that — so the
spool threshold is raised above the upload limit to keep documents in RAM.
Nothing is written to disk, logged, or sent anywhere. The container needs no
outbound network at runtime: the OCR weights are baked into the image.

## Deploying to AWS

The same Dockerfile and the same source build for Fargate. Only the target
architecture differs.

```bash
# 1. Build for the Fargate architecture (x86_64 shown; use linux/arm64 for Graviton)
docker buildx build --platform linux/amd64 -t uae-mulkiya-reader:prod --load .

# 2. Push to ECR
aws ecr create-repository --repository-name uae-mulkiya-reader
aws ecr get-login-password --region me-central-1 \
  | docker login --username AWS --password-stdin <acct>.dkr.ecr.me-central-1.amazonaws.com
docker tag uae-mulkiya-reader:prod <acct>.dkr.ecr.me-central-1.amazonaws.com/uae-mulkiya-reader:latest
docker push <acct>.dkr.ecr.me-central-1.amazonaws.com/uae-mulkiya-reader:latest
```

For the ECS task definition: 2 vCPU / 4 GB is a reasonable start, with
`OCR_CPU_THREADS` set to match the vCPU count and `INCLUDE_RAW_OCR=false`. Put
an ALB in front for HTTPS. Because model load takes ~10 s, set the target group
health check to `/health` with a grace period of at least 60 s, and scale
horizontally for concurrency rather than raising `--workers` — one warm OCR
worker per container is the intended shape.

Graviton (`linux/arm64`) works and is cheaper; x86 may be faster per request
since oneDNN applies there. Benchmark both against your own volume.
