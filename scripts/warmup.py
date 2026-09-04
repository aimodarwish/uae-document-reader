"""Bake the PP-OCRv5 models into the image and prove the inference path works.

Runs at BUILD time. Two jobs:

1. Pull the detection + recognition weights from the official PaddlePaddle
   Hugging Face org so the running container never downloads anything on its
   first request and needs no model CDN on AWS.
2. Actually run detection AND recognition. A broken cv2, a missing transitive
   import or a paddle/paddlex version mismatch then fails the *build* loudly
   instead of surfacing as a 500 on the first real Mulkiya.
"""

import os
import platform
import sys

import cv2
import numpy as np
from paddleocr import TextDetection, TextRecognition

det_model = os.environ.get("DET_MODEL", "PP-OCRv5_mobile_det")
rec_models = {
    "en": os.environ.get("REC_MODEL_EN", "en_PP-OCRv5_mobile_rec"),
    "ar": os.environ.get("REC_MODEL_AR", "arabic_PP-OCRv5_mobile_rec"),
}
langs = [x.strip() for x in os.environ.get("OCR_LANGS", "en,ar").split(",") if x.strip()]
threads = int(os.environ.get("OCR_CPU_THREADS", "4"))
mkldnn = platform.machine().lower() in {"x86_64", "amd64"}

import paddle  # noqa: E402  (imported after cv2 so a cv2 break reports first)

print(f"[warmup] python {sys.version.split()[0]} | {platform.machine()}")
print(f"[warmup] paddle {paddle.__version__} | cv2 {cv2.__version__} | numpy {np.__version__}")
print(f"[warmup] source={os.environ.get('PADDLE_PDX_MODEL_SOURCE', 'huggingface')} "
      f"mkldnn={mkldnn} threads={threads}")

common = {"device": "cpu", "cpu_threads": threads, "enable_mkldnn": mkldnn}

# Synthetic label/value rows so detection and recognition both genuinely run.
canvas = np.full((260, 940, 3), 255, np.uint8)
for i, text in enumerate(("Chassis No. SAL1P9EU2RA165631", "Exp. Date 19/06/2026")):
    cv2.putText(canvas, text, (20, 90 + i * 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

print(f"[warmup] detector: {det_model}")
detector = TextDetection(model_name=det_model, **common)
results = detector.predict(canvas)
polys = results[0]["dt_polys"] if results else []
print(f"[warmup]   detected {len(polys)} box(es)")

if len(polys) == 0:
    raise SystemExit("[warmup] FAILED: detector found no text on a clean synthetic image")


def crop(box):
    box = np.array(box, dtype=np.float32).reshape(4, 2)
    w = int(max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[2] - box[3])))
    h = int(max(np.linalg.norm(box[0] - box[3]), np.linalg.norm(box[1] - box[2])))
    w, h = max(w, 1), max(h, 1)
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    out = cv2.warpPerspective(canvas, cv2.getPerspectiveTransform(box, dst), (w, h),
                              borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    return np.ascontiguousarray(np.rot90(out) if h / max(w, 1) >= 1.5 else out)


crops = [crop(p) for p in polys]
for lang in langs:
    name = rec_models.get(lang)
    if not name:
        print(f"[warmup] no recognition model mapped for {lang!r} - skipped")
        continue
    print(f"[warmup] recogniser [{lang}]: {name}")
    readings = TextRecognition(model_name=name, **common).predict(crops)
    for item in readings:
        print(f"[warmup]   {item['rec_text']!r}  ({float(item['rec_score']):.2f})")

print("[warmup] OK - models cached and full inference path verified")
