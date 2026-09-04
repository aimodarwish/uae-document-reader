#!/usr/bin/env bash
# Guard, not a fixer.
#
# Every OpenCV distribution (opencv-python, opencv-contrib-python, and their
# -headless twins) unpacks into the SAME site-packages/cv2/ directory. Install
# two and the last one written silently wins, leaving a mixed-version cv2 that
# fails in confusing ways at runtime.
#
# paddlex pins opencv-contrib-python==4.10.0.84 and verifies it by DISTRIBUTION
# NAME via importlib.metadata, so the headless twin does not satisfy it -- which
# is why this image installs the GUI build and carries libgl1. Do not add
# opencv-python-headless to requirements.txt.
#
# This script fails the build if that invariant is ever broken.
set -euo pipefail

echo "[check-opencv] installed distributions:"
pip list 2>/dev/null | grep -i '^opencv' || echo "  (none)"

COUNT="$(pip list 2>/dev/null | grep -ci '^opencv' || true)"
if [ "${COUNT}" -ne 1 ]; then
    echo "[check-opencv] FAILED: expected exactly 1 opencv distribution, found ${COUNT}." >&2
    echo "[check-opencv] They share the cv2/ directory and will corrupt each other." >&2
    exit 1
fi

python - <<'PY'
import cv2, importlib.metadata as md
print(f"[check-opencv] cv2 {cv2.__version__} imports cleanly")
print(f"[check-opencv] paddlex sees opencv-contrib-python == {md.version('opencv-contrib-python')}")
PY
