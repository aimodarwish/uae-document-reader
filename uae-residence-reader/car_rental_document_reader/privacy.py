from __future__ import annotations

import logging
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class PiiRedactionFilter(logging.Filter):
    _long_token = re.compile(r"(?<!\w)[A-Z0-9<\-]{7,}(?!\w)", re.I)

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        redacted = self._long_token.sub("[REDACTED]", message)
        record.msg, record.args = redacted, ()
        return True


logger = logging.getLogger("car_rental_document_reader")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.addFilter(PiiRedactionFilter())
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(handler)
logger.propagate = False


def configure_privacy_environment() -> None:
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"


@contextmanager
def private_temp_file(suffix: str = "") -> Iterator[Path]:
    fd, raw_path = tempfile.mkstemp(prefix="docreader_", suffix=suffix)
    os.close(fd)
    path = Path(raw_path)
    try:
        yield path
    finally:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Temporary file cleanup failed")
