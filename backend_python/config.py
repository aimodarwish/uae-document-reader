from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "UAE Mulkiya Reader"

    # OCR
    ocr_langs: str = "en,ar"
    ocr_max_width: int = 1800

    # Phone photos and WhatsApp-compressed scans can arrive small enough that a
    # short Arabic word is only a few dozen pixels wide, which costs whole
    # fields. Below this width the page is upscaled before OCR. 0 disables.
    ocr_min_width: int = 1600
    ocr_min_confidence: float = 0.35
    ocr_cpu_threads: int = 4

    # Measured, not assumed: PaddleX pads every crop in a batch to the widest
    # aspect ratio in that batch, so batching these variable-width crops is a
    # LOSS. 1 measured 5.5 s against 8.5-9.2 s for 4/8/16 on the same card.
    ocr_rec_batch_size: int = 1

    # The Arabic model's charset covers latin and digits, so it reads the whole
    # card on its own. The English specialist is only worth its cost on latin
    # boxes the Arabic model was unsure about -- above this confidence it is
    # skipped. Measured: same fields either way, 6.0 s -> 3.5 s.
    ocr_second_pass_threshold: float = 0.90

    # How much more confident the English reading must be before it overrides an
    # Arabic-script reading of the same box. Guards against detection merging
    # table cells: Arabic returns plausible-looking noise, English returns the
    # date that is actually there.
    ocr_script_override_margin: float = 0.15

    # PaddleOCR 3.x model names, resolved from the official PaddlePaddle
    # Hugging Face org. "mobile" variants are the small/fast ones -- the server
    # variants are markedly slower and we have a 5 s budget.
    det_model: str = "PP-OCRv5_mobile_det"
    rec_model_en: str = "en_PP-OCRv5_mobile_rec"
    rec_model_ar: str = "arabic_PP-OCRv5_mobile_rec"

    # oneDNN/MKL-DNN is an x86 acceleration path. Left as None it is switched on
    # for x86_64 and off on arm64, which is what runs on Apple Silicon.
    enable_mkldnn: bool | None = None

    # Fixed recognition input as "C,H,W". PaddleX otherwise reshapes per crop,
    # which prevents shape-dependent optimisation. Empty = dynamic (default).
    ocr_rec_input_shape: str = ""

    # Uploads
    max_upload_mb: int = 15
    max_batch_files: int = 20
    max_pdf_pages: int = 2
    pdf_render_dpi: int = 200

    # Debug: return every OCR line (text + box + confidence) alongside the
    # extracted fields so a missed field can be diagnosed. Off in production.
    include_raw_ocr: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def languages(self) -> list[str]:
        return [x.strip() for x in self.ocr_langs.split(",") if x.strip()]

    @field_validator("enable_mkldnn", mode="before")
    @classmethod
    def _blank_means_auto(cls, value: Any) -> Any:
        # docker-compose writes "${ENABLE_MKLDNN:-}" as an empty string when the
        # variable is unset, which is not parseable as a bool. Treat it as
        # "unset" so the architecture default applies.
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def rec_input_shape(self) -> tuple[int, int, int] | None:
        if not self.ocr_rec_input_shape.strip():
            return None
        parts = [int(x) for x in self.ocr_rec_input_shape.split(",")]
        return (parts[0], parts[1], parts[2]) if len(parts) == 3 else None

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


settings = Settings()
