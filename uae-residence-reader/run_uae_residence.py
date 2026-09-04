from __future__ import annotations

import os

from car_rental_document_reader.pipeline import DocumentReader
from car_rental_document_reader.privacy import configure_privacy_environment
from uae_residence_app import build_uae_demo


HOST = "127.0.0.1"
PORT = int(os.environ.get("UAE_RESIDENCE_PORT", "7871"))


def main() -> None:
    os.environ.update({
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "DO_NOT_TRACK": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
    })
    configure_privacy_environment()

    reader = DocumentReader(initialize_vlm=False)
    reader.ocr.initialize()
    if not reader.ocr.engines:
        raise RuntimeError(
            f"OCR failed to initialize: {reader.ocr.initialization_warnings}"
        )

    print({
        "app": "UAE Residence Document Reader",
        "local_url": f"http://{HOST}:{PORT}",
        "workflow": "UAE_RESIDENT",
        "ocr_languages": sorted(reader.ocr.engines),
        "ocr_warnings": reader.ocr.initialization_warnings,
        "vlm_enabled": False,
        "public_share": False,
    })

    demo = build_uae_demo(reader)
    demo.queue(default_concurrency_limit=1).launch(
        server_name=HOST,
        server_port=PORT,
        share=False,
        inbrowser=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
