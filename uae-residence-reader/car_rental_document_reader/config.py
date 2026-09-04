from __future__ import annotations

import gc
import os
import platform
import sys
from dataclasses import dataclass, field
from importlib import metadata
from typing import Any


@dataclass(frozen=True)
class RuntimeInfo:
    python_version: str
    platform: str
    system_ram_gb: float
    gpu_available: bool
    gpu_name: str | None
    gpu_vram_gb: float
    cuda_version: str | None
    torch_version: str | None


@dataclass(frozen=True)
class ModelChoice:
    model_id: str | None
    quantization: str | None
    estimated_model_gb: float
    reserve_gb: float
    reason: str


@dataclass
class AppConfig:
    max_file_mb: int = 40
    max_pdf_pages: int = 8
    pdf_dpi: int = 250
    max_image_dimension: int = 3200
    # An ID-1 card occupying one half of a common 848x1080 two-sided upload is
    # about 848x535 pixels.  Its 8pt rows are still recoverable (and are
    # upscaled for OCR below), so treating every such split as low-resolution
    # creates a quality warning on a demonstrably readable card.  This is only
    # the evidence-quality floor; the independent OCR target remains 1600 and
    # therefore recognition speed/scale is unchanged.
    min_image_side: int = 500
    blur_threshold: float = 80.0
    glare_fraction_threshold: float = 0.012
    ocr_languages: tuple[str, ...] = ("en", "ar")
    # How far a small capture is enlarged before recognition. Character size,
    # not page size, is what a recogniser resolves.
    ocr_target_min_side: int = 1600
    # UAE Resident is an interactive card-reading route.  It prefers a clear
    # first reading and operator review over retrying alternate full-page OCR
    # views or a generative model.  Keeping the rendered card compact leaves
    # enough headroom for the complete two-card request to meet its fast-path
    # service target on a warm GPU.
    uae_fast_ocr_target_min_side: int = 1280
    uae_fast_path_enabled: bool = True
    uae_processing_budget_seconds: float = 4.25
    ocr_variant_names: tuple[str, ...] = ("original_normalized",)
    # Tried only for fields the first pass left unread, so a clean capture pays
    # nothing for them. Ordered by what they repair: flattening removes the
    # glare band a laminated card throws across its own print, deblurring
    # restores edges a hand-held capture smeared, and CLAHE lifts local
    # contrast last -- run on a flattened page rather than on a glared one,
    # which is what it was doing before and why it amplified the glare.
    ocr_fallback_variant_names: tuple[str, ...] = (
        "illumination_flattened", "deblurred", "contrast_enhanced",
    )
    ocr_model: str = "PP-OCRv5"
    enable_document_vl: bool = False
    document_vl_pipeline_version: str = "v1.6"
    document_vl_max_new_tokens: int = 1536
    document_vl_detail_crops: bool = True
    enable_vlm: bool = True
    automatic_vlm_fallback: bool = True
    max_new_tokens: int = 256
    model_choice: ModelChoice | None = None
    package_licenses: dict[str, str] = field(default_factory=lambda: {
        "paddleocr": "Apache-2.0", "paddlepaddle": "Apache-2.0",
        "transformers": "Apache-2.0", "accelerate": "Apache-2.0",
        "bitsandbytes": "MIT", "gradio": "Apache-2.0",
        "zxing-cpp": "Apache-2.0", "opencv-python-headless": "Apache-2.0",
    })


def _system_ram_gb() -> float:
    try:
        import psutil
        return round(psutil.virtual_memory().total / 2**30, 2)
    except ImportError:
        if hasattr(os, "sysconf"):
            return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 2)
    return 0.0


def detect_runtime() -> RuntimeInfo:
    try:
        import torch
        torch_version = torch.__version__
        cuda_version = torch.version.cuda
        available = bool(torch.cuda.is_available())
        if available:
            props = torch.cuda.get_device_properties(0)
            gpu_name = props.name
            gpu_vram = round(props.total_memory / 2**30, 2)
        else:
            gpu_name, gpu_vram = None, 0.0
    except ImportError:
        torch_version = cuda_version = gpu_name = None
        available, gpu_vram = False, 0.0
    return RuntimeInfo(
        python_version=platform.python_version(), platform=platform.platform(),
        system_ram_gb=_system_ram_gb(), gpu_available=available,
        gpu_name=gpu_name, gpu_vram_gb=gpu_vram,
        cuda_version=cuda_version, torch_version=torch_version,
    )


def select_vlm(runtime: RuntimeInfo) -> ModelChoice:
    vram = runtime.gpu_vram_gb
    if not runtime.gpu_available:
        return ModelChoice(None, None, 0.0, 0.0, "CPU fallback: VLM disabled")
    if vram >= 35:
        return ModelChoice("Qwen/Qwen3-VL-32B-Instruct", "4-bit", 19.5, 8.0, "Large-memory GPU")
    if vram >= 14:
        return ModelChoice("Qwen/Qwen3-VL-8B-Instruct", "4-bit", 6.5, 4.0, "Mid-memory GPU")
    if vram >= 7:
        return ModelChoice("Qwen/Qwen3-VL-4B-Instruct", "4-bit", 3.7, 2.5, "Low-memory GPU")
    return ModelChoice(None, None, 0.0, 1.5, "GPU VRAM below safe VLM threshold")


def clear_accelerators() -> None:
    gc.collect()
    # Cleanup must not load the accelerator stack in a Google-only process.
    # A local model that used Torch has already imported it into sys.modules.
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def installed_versions(names: list[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def runtime_table(info: RuntimeInfo, choice: ModelChoice) -> dict[str, Any]:
    return {
        "python": info.python_version, "platform": info.platform,
        "system_ram_gb": info.system_ram_gb, "gpu": info.gpu_name or "CPU",
        "gpu_vram_gb": info.gpu_vram_gb, "cuda": info.cuda_version,
        "torch": info.torch_version, "selected_vlm": choice.model_id,
        "quantization": choice.quantization, "estimated_model_gb": choice.estimated_model_gb,
        "reserved_working_gb": choice.reserve_gb, "selection_reason": choice.reason,
    }
