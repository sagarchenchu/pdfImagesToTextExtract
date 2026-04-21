#!/usr/bin/env python3
"""
download_models.py
==================
Pre-download TrOCR and EasyOCR models into the local  models/  directory so
they can be bundled into the Windows EXE by PyInstaller.

Run this ONCE before building with PyInstaller:

    python download_models.py

Models are written to:
    models/trocr/    – microsoft/trocr-large-handwritten weights (flat layout)
    models/easyocr/  – EasyOCR CRAFT + English recognition weights

These directories are excluded from git (see .gitignore) but included in the
PyInstaller bundle via handwriting_extractor.spec.
"""

import sys
from pathlib import Path

# On Windows the default console encoding (cp1252 / charmap) cannot represent
# Unicode characters such as the FULL BLOCK (U+2588) used by tqdm progress bars
# during model downloads.  Reconfigure stdout/stderr to UTF-8 so those
# characters are printed correctly instead of raising UnicodeEncodeError.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = Path(__file__).parent.resolve()
MODELS_DIR = SCRIPT_DIR / "models"
TROCR_CACHE_DIR = MODELS_DIR / "trocr"
EASYOCR_MODEL_DIR = MODELS_DIR / "easyocr"

TROCR_MODEL = "microsoft/trocr-large-handwritten"


def download_trocr() -> None:
    TROCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n[TrOCR] Downloading '{TROCR_MODEL}' to {TROCR_CACHE_DIR} …")
    print("        This is ~1.35 GB and may take several minutes on first run.")

    # Use snapshot_download with local_dir so every model file is written
    # directly into TROCR_CACHE_DIR (flat layout: config.json, model weights,
    # tokenizer files …).  This avoids the HuggingFace Hub blob-cache format
    # which on Windows – where symlinks require elevated privileges – stores
    # the same data twice (once in blobs/, once in snapshots/), nearly doubling
    # the on-disk footprint and pushing the resulting zip above GitHub's 2 GB
    # release-asset limit.
    import warnings
    import shutil
    from huggingface_hub import snapshot_download

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # suppress local_dir_use_symlinks deprecation
        snapshot_download(
            repo_id=TROCR_MODEL,
            local_dir=str(TROCR_CACHE_DIR),
            local_dir_use_symlinks=False,  # store real copies, not symlinks
        )

    # Remove the download-tracking metadata directory that huggingface_hub
    # creates inside local_dir (named ".cache" as of huggingface_hub 0.20+).
    # It is not needed at runtime; silently skip if the name changes in a
    # future release – it only contains small lock/progress files.
    hf_tracking = TROCR_CACHE_DIR / ".cache"
    if hf_tracking.exists():
        shutil.rmtree(hf_tracking)

    print(f"[TrOCR] [OK] Model saved to {TROCR_CACHE_DIR}")


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

    print(f"[EasyOCR] [OK] Models saved to {EASYOCR_MODEL_DIR}")


def main() -> None:
    print("=" * 60)
    print(" HandwritingExtractor – pre-download ML models for bundling")
    print("=" * 60)

    download_trocr()
    download_easyocr()

    print("\n" + "=" * 60)
    print(" [OK] All models downloaded successfully.")
    print(f"     Models directory: {MODELS_DIR}")
    print("     You can now run:  pyinstaller handwriting_extractor.spec")
    print("=" * 60)


if __name__ == "__main__":
    main()
