#!/usr/bin/env python3
"""
download_models.py
==================
Pre-download TrOCR and EasyOCR models into the local  models/  directory so
they can be bundled into the Windows EXE by PyInstaller.

Run this ONCE before building with PyInstaller:

    python download_models.py

Models are written to:
    models/trocr/    – HuggingFace Hub cache for microsoft/trocr-large-handwritten
    models/easyocr/  – EasyOCR CRAFT + English recognition weights

These directories are excluded from git (see .gitignore) but included in the
PyInstaller bundle via handwriting_extractor.spec.
"""

import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
MODELS_DIR = SCRIPT_DIR / "models"
TROCR_CACHE_DIR = MODELS_DIR / "trocr"
EASYOCR_MODEL_DIR = MODELS_DIR / "easyocr"

TROCR_MODEL = "microsoft/trocr-large-handwritten"


def download_trocr() -> None:
    TROCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Tell HuggingFace Hub where to store the downloaded files.
    # Must be set before importing transformers / huggingface_hub.
    os.environ["HF_HUB_CACHE"] = str(TROCR_CACHE_DIR)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(TROCR_CACHE_DIR)
    os.environ["TRANSFORMERS_CACHE"] = str(TROCR_CACHE_DIR)

    print(f"\n[TrOCR] Downloading '{TROCR_MODEL}' to {TROCR_CACHE_DIR} …")
    print("        This is ~1 GB and may take several minutes on first run.")

    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    TrOCRProcessor.from_pretrained(TROCR_MODEL)
    VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL)

    print(f"[TrOCR] ✅  Model saved to {TROCR_CACHE_DIR}")


def download_easyocr() -> None:
    EASYOCR_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n[EasyOCR] Downloading models to {EASYOCR_MODEL_DIR} …")
    print("          This is ~250 MB and may take a moment.")

    import easyocr  # noqa: F401 – triggers model download via model_storage_directory

    # Instantiate a Reader with gpu=False so the download works without CUDA.
    # The gpu setting used here only affects this download; app.py chooses
    # gpu/cpu at inference time.
    easyocr.Reader(
        ["en"],
        gpu=False,
        model_storage_directory=str(EASYOCR_MODEL_DIR),
        download_enabled=True,
    )

    print(f"[EasyOCR] ✅  Models saved to {EASYOCR_MODEL_DIR}")


def main() -> None:
    print("=" * 60)
    print(" HandwritingExtractor – pre-download ML models for bundling")
    print("=" * 60)

    download_trocr()
    download_easyocr()

    print("\n" + "=" * 60)
    print(" ✅  All models downloaded successfully.")
    print(f"     Models directory: {MODELS_DIR}")
    print("     You can now run:  pyinstaller handwriting_extractor.spec")
    print("=" * 60)


if __name__ == "__main__":
    main()
