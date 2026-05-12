"""
Handwriting Text Extractor
===========================
Extracts printed or handwritten fields from scanned checks using:
  - EasyOCR  : printed check-field recognition
  - TrOCR    : Microsoft's transformer OCR model for handwriting recognition

GUI built with tkinter (ships with Python – no extra install needed).
Packaged to a Windows .exe with PyInstaller via the included spec file.
"""

import concurrent.futures
import errno as _errno
import io
import logging
import os
import re
import sys
import tempfile
import threading
import traceback
import zipfile
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple, Union

# In a PyInstaller console=False build sys.stdout and sys.stderr are set to
# None by the bootloader because no console is attached.  Some third-party
# libraries (pip, rich, tqdm, …) call sys.stdout.isatty() or sys.stdout.write()
# without guarding against None, which raises:
#
#   AttributeError: 'NoneType' object has no attribute 'isatty'
#
# Replace None streams with a no-op sink before any other code runs.
# rthook_stdio.py does the same for the rthook phase; this guard covers the
# case where app.py is run directly (non-frozen) and also provides defence-in-
# depth for the frozen path.
if sys.stdout is None or sys.stderr is None:
    _devnull = open(os.devnull, "w", encoding="utf-8", errors="replace")  # noqa: WPS515
    if sys.stdout is None:
        sys.stdout = _devnull
    if sys.stderr is None:
        sys.stderr = _devnull

# On Windows the default console encoding (cp1252 / charmap) cannot represent
# Unicode characters such as the FULL BLOCK (U+2588) used by tqdm progress bars
# when ML models are downloaded on first run.  Reconfigure stdout/stderr to
# UTF-8 early so those characters are printed correctly instead of raising
# UnicodeEncodeError.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# File-based logging — captures everything even when console=False in the EXE
# ---------------------------------------------------------------------------
_LOG_PATH = Path(tempfile.gettempdir()) / "HandwritingExtractor.log"
_ERROR_SEP = "═" * 60

def _setup_logging() -> None:
    """Configure the root logger to write full tracebacks to a log file."""
    log_dir = _LOG_PATH.parent
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError as _mkdir_err:
        sys.stderr.write(f"WARNING: could not create log directory {log_dir}: {_mkdir_err}\n")

    logging.basicConfig(
        filename=str(_LOG_PATH),
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        encoding="utf-8",
        errors="replace",
        force=True,  # override any existing handler (e.g. from transformers)
    )
    logging.info("=" * 60)
    logging.info("HandwritingExtractor started")
    logging.info("Python %s", sys.version)
    logging.info("frozen=%s", getattr(sys, "frozen", False))
    if getattr(sys, "frozen", False):
        logging.info("_MEIPASS=%s", getattr(sys, "_MEIPASS", "n/a"))
    logging.info("Log file: %s", _LOG_PATH)
    logging.info("=" * 60)

_setup_logging()

import numpy as np
from PIL import Image, ImageFilter, ImageOps

# ---------------------------------------------------------------------------
# Tkinter GUI
# ---------------------------------------------------------------------------
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk



# ---------------------------------------------------------------------------
# Lazy-load heavy ML libraries so the window opens immediately
# ---------------------------------------------------------------------------
_easyocr_reader = None
_trocr_processor = None
_trocr_model = None
_device = None

TROCR_MODEL = "microsoft/trocr-base-handwritten"

# Supported file extensions — single source of truth used by both the upload
# dialog and the ZIP entry filter.
_IMAGE_EXTS: frozenset = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
)
_PDF_EXT: str = ".pdf"
_ZIP_EXT: str = ".zip"

_CHECK_MODE_PRINTED = "printed"
_CHECK_MODE_HANDWRITTEN = "handwritten"
_CHECK_MODE_LABELS = {
    _CHECK_MODE_PRINTED: "EasyOCR Printed Check",
    _CHECK_MODE_HANDWRITTEN: "Handwritten Check",
}
_SUPPORTED_CHECK_EXTS: FrozenSet[str] = frozenset({_PDF_EXT, ".tif", ".tiff", ".png", ".jpg", ".jpeg"})

# Percentage-based crop boxes: (left, top, right, bottom).  These are broad
# enough to cover common personal/business check layouts while avoiding the
# signature line for the memo field.
_CHECK_FIELD_BOXES: Dict[str, Tuple[float, float, float, float]] = {
    "pay_to_order_of": (0.16, 0.30, 0.86, 0.47),
    "memo": (0.06, 0.72, 0.48, 0.88),
}
_CHECK_FIELD_LABELS: Dict[str, str] = {
    "pay_to_order_of": "Pay to the Order of",
    "memo": "For/Memo",
}
_CHECK_DEBUG_FILENAMES: Dict[str, Tuple[str, str]] = {
    "pay_to_order_of": ("pay_to_order_of_original.png", "pay_to_order_of_preprocessed.png"),
    "memo": ("memo_original.png", "memo_preprocessed.png"),
}
_LOW_CONFIDENCE_HANDWRITING_MESSAGE = (
    "Low confidence handwriting OCR. Please check debug crop alignment."
)
_OCR_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_HANDWRITTEN_CROP_SMALL_SIDE_MIN_PX = 120
_HANDWRITTEN_SMALL_CROP_SCALE = 3
_HANDWRITTEN_LARGE_CROP_SCALE = 2
_HANDWRITTEN_MIN_ALPHA_RATIO = 0.55
_HANDWRITTEN_MAX_PUNCTUATION_RATIO = 0.25
_HANDWRITTEN_MIN_TOKENS_FOR_SHORT_RATIO = 3
_HANDWRITTEN_MAX_SHORT_TOKEN_RATIO = 0.75


def _get_device():
    global _device
    if _device is None:
        try:
            import torch
            _device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            _device = "cpu"
    return _device


_NETWORK_ERRNO: frozenset = frozenset({
    _errno.ECONNREFUSED,   # Connection refused
    _errno.ETIMEDOUT,      # Connection timed out
    _errno.ENETUNREACH,    # Network unreachable
    _errno.EHOSTUNREACH,   # No route to host
    _errno.ECONNRESET,     # Connection reset by peer
})


def _is_connection_error(exc: BaseException) -> bool:
    """Return True if *exc* (or any chained cause) is a network/connection error."""
    try:
        import requests.exceptions as req_exc
        import urllib3.exceptions as urllib3_exc
        _network_types = (
            req_exc.ConnectionError,
            req_exc.Timeout,
            req_exc.SSLError,
            urllib3_exc.NewConnectionError,
            urllib3_exc.MaxRetryError,
        )
    except ImportError:
        _network_types = ()

    checked: Set[int] = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in checked:
        checked.add(id(current))
        if isinstance(current, _network_types):
            return True
        # OSError covers socket-level errors; use errno codes for reliable detection
        if isinstance(current, OSError) and (
            getattr(current, "errno", None) in _NETWORK_ERRNO
            or any(
                kw in str(current).lower()
                for kw in ("ssl", "certificate", "handshake")
            )
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _resolve_models_dir() -> Optional[Path]:
    """Locate the ``models/`` directory at runtime (frozen EXE only).

    Search order (first match wins):

    1. ``models/`` folder **next to the EXE** — sideloaded by the user.
    2. ``models/`` folder **inside** ``sys._MEIPASS`` — legacy bundled path.

    Returns the :class:`~pathlib.Path` to the found directory, or *None*
    when running from source or when neither location is present.
    """
    if not getattr(sys, "frozen", False):
        return None

    # 1. Sideloaded: a models/ folder the user placed next to the EXE
    sideloaded = Path(sys.executable).parent / "models"
    if sideloaded.exists():
        logging.info("Using sideloaded models from %s", sideloaded)
        return sideloaded

    # 2. Bundled inside the frozen archive (fallback for builds that still
    #    include models in _MEIPASS)
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass is not None:
        bundled = Path(meipass) / "models"
        if bundled.exists():
            logging.info("Using bundled models from %s", bundled)
            return bundled

    return None


def _bundled_easyocr_model_dir() -> Optional[str]:
    """Return the EasyOCR model directory when running as a frozen EXE."""
    models_dir = _resolve_models_dir()
    if models_dir is not None:
        easyocr_dir = models_dir / "easyocr"
        if easyocr_dir.exists():
            return str(easyocr_dir)
    return None


def _warm_torchvision() -> None:
    """
    Pre-import torchvision sub-packages in the correct dependency order.

    torchvision >= 0.16 uses a lazy loader in ``__init__.py``.  If a
    downstream package (easyocr, timm) executes
    ``from torchvision import models`` while ``__init__`` is still running,
    Python raises::

        ImportError: cannot import name 'models' from partially initialized
        module 'torchvision' (most likely due to a circular import)

    Calling this function once, before ``import easyocr``, ensures the full
    module registry is populated and avoids that race.  The rthook
    ``rthooks/rthook_torchvision.py`` calls this same function at EXE startup
    for the frozen-build path.
    """
    try:
        import torchvision                        # noqa: F401
        import torchvision.models                 # noqa: F401
        import torchvision.ops                    # noqa: F401
        import torchvision.transforms             # noqa: F401
        import torchvision.transforms.functional  # noqa: F401
    except Exception:
        logging.warning("torchvision pre-import warning:\n%s", traceback.format_exc())


def _load_easyocr(status_cb):
    global _easyocr_reader
    if _easyocr_reader is None:
        status_cb("Loading EasyOCR model…")
        _warm_torchvision()
        import easyocr
        kwargs: dict = {"gpu": _get_device() == "cuda"}
        model_dir = _bundled_easyocr_model_dir()
        if model_dir:
            # Bundled EXE: load from local models directory, no download
            kwargs["model_storage_directory"] = model_dir
            kwargs["download_enabled"] = False
        try:
            _easyocr_reader = easyocr.Reader(["en"], **kwargs)
        except Exception as exc:
            logging.error("EasyOCR Reader init failed:\n%s", traceback.format_exc())
            if _is_connection_error(exc):
                raise ConnectionError(
                    "Could not download EasyOCR models — check your internet connection and try again.\n\n"
                    f"Details: {exc}"
                ) from exc
            raise
    return _easyocr_reader


def _load_trocr(status_cb):
    global _trocr_processor, _trocr_model
    if _trocr_processor is None or _trocr_model is None:
        # Prefer the top-level lazy import; fall back to direct module paths so
        # the frozen EXE works even when transformers' lazy-loader silently
        # suppresses TrOCRProcessor because an optional backend is absent.
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            # Verify the lazy import actually resolved the class (it can silently
            # produce a dummy object when sentencepiece / tokenizers is missing).
            if not isinstance(TrOCRProcessor, type):
                raise AttributeError("TrOCRProcessor is not a valid class")
        except (ImportError, AttributeError) as _lazy_err:
            logging.warning(
                "transformers lazy import of TrOCRProcessor failed (%s) — "
                "using direct module-path import.", _lazy_err
            )
            from transformers.models.trocr.processing_trocr import TrOCRProcessor  # type: ignore[no-redef]
            from transformers.models.vision_encoder_decoder.modeling_vision_encoder_decoder import (  # type: ignore[no-redef]
                VisionEncoderDecoderModel,
            )

        frozen = getattr(sys, "frozen", False)

        if frozen:
            # Running as frozen EXE — locate the models/ directory.
            # Priority: sideloaded folder next to EXE > bundled inside _MEIPASS.
            models_dir = _resolve_models_dir()
            if models_dir is None:
                raise FileNotFoundError(
                    "models/ directory not found.\n\n"
                    "Place the models/ folder next to HandwritingExtractor.exe:\n"
                    "  HandwritingExtractor.exe\n"
                    "  models\\\n"
                    "    trocr\\\n"
                    "      config.json, pytorch_model.bin, …\n"
                    "    easyocr\\\n"
                    "      craft_mlt_25k.pth, english_g2.pth\n\n"
                    "Download links are in the project README."
                )
            status_cb(f"Loading TrOCR model '{TROCR_MODEL}' from {models_dir}…")
            trocr_local = str(models_dir / "trocr")
            _trocr_dir_exists = Path(trocr_local).exists()
            logging.info("Frozen EXE: loading TrOCR from %s", trocr_local)
            logging.info("trocr dir exists: %s", _trocr_dir_exists)
            if _trocr_dir_exists:
                logging.info("trocr dir contents: %s", list(Path(trocr_local).iterdir()))
            _trocr_processor = TrOCRProcessor.from_pretrained(trocr_local, local_files_only=True)
            _trocr_model = VisionEncoderDecoderModel.from_pretrained(trocr_local, local_files_only=True)
        else:
            # Running from source — try local cache first, download if not cached.
            # HuggingFace raises OSError when the model is not cached and local_files_only=True.
            status_cb(f"Loading TrOCR model '{TROCR_MODEL}'…")
            try:
                _trocr_processor = TrOCRProcessor.from_pretrained(TROCR_MODEL, local_files_only=True)
                _trocr_model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL, local_files_only=True)
            except OSError:
                # Model not cached yet — download it now
                status_cb(f"Downloading TrOCR model '{TROCR_MODEL}' (one-time download)…")
                try:
                    _trocr_processor = TrOCRProcessor.from_pretrained(TROCR_MODEL)
                    _trocr_model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL)
                except Exception as exc:
                    if _is_connection_error(exc):
                        raise ConnectionError(
                            "Could not download the TrOCR model — check your internet connection and try again.\n"
                            "The model needs to be downloaded once before it can be used offline.\n\n"
                            f"Details: {exc}"
                        ) from exc
                    raise

        _trocr_model.to(_get_device())
        _trocr_model.eval()
    return _trocr_processor, _trocr_model


# ---------------------------------------------------------------------------
# Core extraction helpers
# ---------------------------------------------------------------------------

def _pdf_to_images(pdf_path: str) -> list:
    """Convert every page of a PDF to a PIL Image (2× zoom for quality)."""
    import fitz  # PyMuPDF
    images = []
    doc = fitz.open(pdf_path)
    try:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat)
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            images.append((page_num + 1, img))
    finally:
        doc.close()
    return images


def _load_check_image(filepath: str) -> Image.Image:
    """Load a supported check file as one RGB image; PDFs use the first page."""
    ext = Path(filepath).suffix.lower()
    if ext == _PDF_EXT:
        pages = _pdf_to_images(filepath)
        if not pages:
            raise ValueError("PDF contains no pages.")
        return pages[0][1].convert("RGB")

    with Image.open(filepath) as img:
        # PIL opens the first frame of TIFF files by default.  Copy before the
        # context manager closes the source file, and normalize mode to RGB.
        return img.convert("RGB").copy()


def _normalize_check_image(image: Image.Image) -> Image.Image:
    """Normalize a check image before percentage-based field cropping."""
    gray = ImageOps.grayscale(image)
    contrast = ImageOps.autocontrast(gray)
    sharpened = contrast.filter(ImageFilter.SHARPEN)
    return sharpened.convert("RGB")


def _autocontrast_or_clahe(gray_image: Image.Image) -> Image.Image:
    """Boost grayscale contrast with CLAHE when OpenCV is available."""
    try:
        import cv2

        arr = np.array(gray_image)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return Image.fromarray(clahe.apply(arr), mode="L")
    except Exception:
        logging.debug("CLAHE unavailable; using Pillow autocontrast.", exc_info=True)
        return ImageOps.autocontrast(gray_image)


def _preprocess_handwritten_crop(image_crop: Image.Image) -> Image.Image:
    """Prepare a handwritten crop for TrOCR recognition."""
    crop = image_crop.convert("RGB")
    pad = max(8, int(round(min(crop.size) * 0.08)))
    padded = ImageOps.expand(crop, border=pad, fill="white")
    # Crops below 120 px on the shortest side are usually thin check fields;
    # _HANDWRITTEN_SMALL_CROP_SCALE keeps strokes legible before TrOCR's
    # fixed-size resize, while _HANDWRITTEN_LARGE_CROP_SCALE avoids needless
    # enlargement for already-large crops.
    scale = (
        _HANDWRITTEN_SMALL_CROP_SCALE
        if min(padded.size) < _HANDWRITTEN_CROP_SMALL_SIDE_MIN_PX
        else _HANDWRITTEN_LARGE_CROP_SCALE
    )
    upscaled = padded.resize((padded.width * scale, padded.height * scale), Image.Resampling.LANCZOS)
    gray = ImageOps.grayscale(upscaled)
    contrast = _autocontrast_or_clahe(gray)
    # MedianFilter(size=3) applies a 3x3 filter that removes light speckle noise
    # without erasing strokes.
    denoised = contrast.filter(ImageFilter.MedianFilter(size=3))
    sharpened = denoised.filter(ImageFilter.SHARPEN)
    return sharpened.convert("RGB")


def _percent_crop(image: Image.Image, box: Tuple[float, float, float, float]) -> Image.Image:
    """Crop *image* using percentage coordinates clamped to image bounds."""
    left, top, right, bottom = box
    width, height = image.size
    x1 = max(0, min(width, int(round(left * width))))
    y1 = max(0, min(height, int(round(top * height))))
    x2 = max(0, min(width, int(round(right * width))))
    y2 = max(0, min(height, int(round(bottom * height))))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Invalid crop box {box!r}: resulting coordinates ({x1}, {y1}, {x2}, {y2}) "
            f"produce zero or negative dimensions for image size {image.size!r}"
        )
    return image.crop((x1, y1, x2, y2))


def _crop_check_fields(image: Image.Image) -> Dict[str, Image.Image]:
    """Return the required check field crops keyed by structured field name."""
    return {
        field_name: _percent_crop(image, crop_box)
        for field_name, crop_box in _CHECK_FIELD_BOXES.items()
    }


def _read_printed_check_crop(image_crop: Image.Image, reader: object) -> str:
    """Run printed OCR on one cropped check field."""
    results = reader.readtext(np.array(image_crop), detail=0, paragraph=True)
    texts: List[str] = []
    for item in results:
        # EasyOCR returns strings with detail=0, but tests and some fallback
        # paths may pass detail=1-style tuples: (bbox, text, confidence).
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            texts.append(str(item[1]))
        else:
            texts.append(str(item))
    return " ".join(t.strip() for t in texts if t and t.strip()).strip()


def _read_printed_check_page(image: Image.Image, reader: object) -> str:
    """Run printed OCR on the full check page."""
    return extract_printed_check_full_page(image, reader)


def _sort_easyocr_detail_results(results: list) -> list:
    """Sort EasyOCR detail=1 results top-to-bottom, then left-to-right."""
    def _reading_order_position(item: object) -> Tuple[float, float]:
        if isinstance(item, (list, tuple)) and item:
            bbox = item[0]
            try:
                xs = [float(pt[0]) for pt in bbox]
                ys = [float(pt[1]) for pt in bbox]
                return min(ys), min(xs)
            except Exception:
                logging.debug("Could not sort EasyOCR bbox %r", bbox, exc_info=True)
        return 0.0, 0.0

    return sorted(results, key=_reading_order_position)


def _extract_easyocr_text(item) -> str:
    """Extract text from EasyOCR detail=1/detail=0 result shapes."""
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        return str(item[1]).strip()
    return str(item).strip()


def _guess_printed_payee(sorted_results: list) -> Optional[str]:
    """Best-effort payee guess from printed OCR results near the pay-to-order label."""
    for idx, item in enumerate(sorted_results):
        text = _extract_easyocr_text(item)
        normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
        if "pay to the order" in normalized and idx + 1 < len(sorted_results):
            candidate = _extract_easyocr_text(sorted_results[idx + 1])
            return candidate or None
    return None


def extract_printed_check_full_page(
    image: Image.Image,
    reader: object,
    return_possible_payee: bool = False,
) -> Union[str, Tuple[str, Optional[str]]]:
    """Run full-page printed OCR and return text in reading order."""
    preprocessed = _normalize_check_image(image.convert("RGB"))
    results = reader.readtext(np.array(preprocessed), detail=1, paragraph=False)
    sorted_results = _sort_easyocr_detail_results(results)
    text_items = [_extract_easyocr_text(item) for item in sorted_results]
    full_text = "\n".join(text for text in text_items if text).strip()
    if return_possible_payee:
        return full_text, _guess_printed_payee(sorted_results)
    return full_text


def looks_garbage(text: str) -> bool:
    """Return True when OCR output looks too noisy to display as handwriting."""
    normalized = (text or "").strip()
    if not normalized:
        return False

    chars = [ch for ch in normalized if not ch.isspace()]
    if not chars:
        return False

    alpha_count = sum(ch.isalpha() for ch in chars)
    digit_count = sum(ch.isdigit() for ch in chars)
    punctuation_count = sum(not ch.isalnum() for ch in chars)
    alpha_ratio = alpha_count / len(chars)
    punctuation_ratio = punctuation_count / len(chars)

    tokens = _OCR_TOKEN_RE.findall(normalized)
    alpha_tokens = [token for token in tokens if any(ch.isalpha() for ch in token)]
    digit_mixed_words = [
        token for token in alpha_tokens
        if any(ch.isdigit() for ch in token) and any(ch.isalpha() for ch in token)
    ]

    if digit_mixed_words:
        return True
    if alpha_ratio < _HANDWRITTEN_MIN_ALPHA_RATIO:
        return True
    if punctuation_ratio > _HANDWRITTEN_MAX_PUNCTUATION_RATIO:
        return True
    if alpha_count < 3 and digit_count > alpha_count:
        return True
    if alpha_tokens and all(len(token) <= 1 for token in alpha_tokens):
        return True
    if (
        len(alpha_tokens) >= _HANDWRITTEN_MIN_TOKENS_FOR_SHORT_RATIO
        and sum(len(token) <= 2 for token in alpha_tokens) / len(alpha_tokens)
        > _HANDWRITTEN_MAX_SHORT_TOKEN_RATIO
    ):
        return True

    return False


def _preprocess_check_debug_full_image(image: Image.Image) -> Image.Image:
    """Create a full-check preview of the preprocessing used before OCR."""
    gray = ImageOps.grayscale(image.convert("RGB"))
    contrast = _autocontrast_or_clahe(gray)
    sharpened = contrast.filter(ImageFilter.SHARPEN)
    return sharpened.convert("RGB")


def _save_debug_check_images(
    original_full_check: Image.Image,
    preprocessed_full_check: Image.Image,
    original_crops: Dict[str, Image.Image],
    preprocessed_crops: Dict[str, Image.Image],
    source_path: str,
) -> Path:
    """Save check debug images next to the source file and return the directory."""
    src = Path(source_path)
    # Save only the exact TrOCR input crops requested for alignment debugging;
    # full-check debug images are intentionally omitted here.
    debug_dir = src.parent / "debug_crops"
    debug_dir.mkdir(parents=True, exist_ok=True)
    for field_name, original_crop in original_crops.items():
        original_filename, preprocessed_filename = _CHECK_DEBUG_FILENAMES[field_name]
        original_crop.save(debug_dir / original_filename)
        preprocessed_crops[field_name].save(debug_dir / preprocessed_filename)
    return debug_dir


def _save_debug_check_crops(crops: Dict[str, Image.Image], source_path: str) -> Path:
    """Save debug images when only field crops are available."""
    crop_values = list(crops.values())
    if not crop_values:
        raise ValueError("Cannot save debug images: crops dictionary is empty.")
    # Legacy crop-only callers do not have the original full check available.
    # Use the first crop as a preview so the modern debug filename set is still
    # produced without changing the old helper signature.
    original_preview = crop_values[0].convert("RGB")
    preprocessed_full = _preprocess_check_debug_full_image(original_preview)
    return _save_debug_check_images(
        original_preview,
        preprocessed_full,
        crops,
        crops,
        source_path,
    )


def _extract_check_fields(
    image: Image.Image,
    mode: str,
    reader: Optional[object] = None,
    save_debug_crops: bool = False,
    source_path: Optional[str] = None,
) -> Dict[str, str]:
    """Extract structured check fields from percentage-based crops only."""
    original = image.convert("RGB")
    original_crops = _crop_check_fields(original)

    if mode == _CHECK_MODE_HANDWRITTEN:
        preprocessed_full = _preprocess_check_debug_full_image(original)
        ocr_crops = {
            field_name: _preprocess_handwritten_crop(crop)
            for field_name, crop in original_crops.items()
        }
    else:
        preprocessed_full = _normalize_check_image(original)
        ocr_crops = _crop_check_fields(preprocessed_full)

    if save_debug_crops and source_path:
        _save_debug_check_images(
            original,
            preprocessed_full,
            original_crops,
            ocr_crops,
            source_path,
        )

    fields = {}
    if mode == _CHECK_MODE_HANDWRITTEN and reader is not None:
        # Preserve printed context from handwritten checks without sending it to TrOCR.
        fields["printed_text"] = _read_printed_check_page(original, reader)

    for field_name, crop in ocr_crops.items():
        if mode == _CHECK_MODE_HANDWRITTEN:
            text = _trocr_read(crop)
            if looks_garbage(text):
                logging.info("Low-confidence TrOCR result for %s: %r", field_name, text)
                text = _LOW_CONFIDENCE_HANDWRITING_MESSAGE
        else:
            text = _read_printed_check_crop(crop, reader)
        fields[field_name] = text.strip()
    return fields


def _format_check_results(fields: Dict[str, str]) -> str:
    """Format structured check OCR output for display/save."""
    printed_text = fields.get("printed_text", "")
    printed_section = f"Printed OCR: {printed_text}\n" if printed_text else ""
    return (
        printed_section +
        f"{_CHECK_FIELD_LABELS['pay_to_order_of']}: "
        f"{fields.get('pay_to_order_of', '')}\n"
        f"{_CHECK_FIELD_LABELS['memo']}: {fields.get('memo', '')}\n"
    )


def _trocr_read(image_crop: Image.Image) -> str:
    """Run TrOCR inference on a single cropped line/region image."""
    return _trocr_read_batch([image_crop])[0]


# Number of crops processed in one model.generate() call.
# Higher values = less per-call overhead → faster throughput on CPU.
# The ViT encoder runs once for the whole batch, so doubling the batch
# size roughly halves the number of expensive encoder forward passes.
# Keep at ≤ 16 to avoid excessive peak memory usage.
_TROCR_BATCH_SIZE = 16

# Maximum number of background threads used for parallel EasyOCR detection.
# All page-detection futures are submitted upfront so up to _WORKER_THREADS
# EasyOCR jobs run concurrently while TrOCR processes earlier pages.
# EasyOCR's PyTorch backend releases the GIL during inference, enabling
# genuine CPU parallelism across cores.
_WORKER_THREADS = 10


def _trocr_read_batch(image_crops: List[Image.Image]) -> List[str]:
    """Run TrOCR inference on a batch of cropped images.

    All crops are resized to the model's fixed input size (384 × 384) by the
    ``TrOCRProcessor`` / ``ViTImageProcessor``, so no explicit padding is
    needed.  Returns one decoded string per crop, in the same order.
    """
    import torch
    processor, model = _trocr_processor, _trocr_model
    device = _get_device()
    pixel_values = processor(image_crops, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        ids = model.generate(
            pixel_values,
            max_new_tokens=64,
            num_beams=1,
            do_sample=False,
        )
    return [s.strip() for s in processor.batch_decode(ids, skip_special_tokens=True)]


# ---------------------------------------------------------------------------
# Legacy full-page OCR helpers
# ---------------------------------------------------------------------------
#
# The active GUI uses structured check extraction above: fixed check-field crops,
# EasyOCR for printed mode, and TrOCR for handwritten mode.  The helpers below
# are retained only for compatibility with older tests and archived flows that
# processed whole pages by detecting regions with EasyOCR and reading each
# detected region with TrOCR.

def _detect_regions(image: Image.Image, reader) -> list:
    """Run EasyOCR text-region detection on *image* and return the raw detections.

    Separated from ``_extract_from_image`` so detection can be executed in a
    background thread, overlapping with TrOCR inference on the previous page
    (pipeline parallelism).  Both EasyOCR and TrOCR release the Python GIL
    during their heavy C++/BLAS operations, enabling genuine parallel CPU
    execution when run from separate threads.

    Returns the same list that ``reader.readtext(..., detail=1, paragraph=False)``
    would return.
    """
    img_array = np.array(image)
    return reader.readtext(img_array, detail=1, paragraph=False)


def _extract_from_image(
    image: Image.Image,
    label: str,
    progress_cb,
    status_cb,
    append_cb,
    progress_start: float = 0.0,
    progress_share: float = 100.0,
    reader=None,
    detections=None,
) -> list:
    """
    Detect text regions with EasyOCR, read each with TrOCR.

    Parameters
    ----------
    reader:
        An initialised ``easyocr.Reader`` instance.  When *None* (default)
        the module-level ``_easyocr_reader`` global is used; callers should
        prefer passing the reader explicitly to avoid relying on global state.
    detections:
        Pre-computed EasyOCR detections returned by ``_detect_regions()``.
        When provided the EasyOCR step is skipped entirely, which enables
        pipeline parallelism: the caller can run EasyOCR for the *next* page
        in a background thread while TrOCR processes the *current* page.
        When *None* (default) EasyOCR is called inline as before.

    Returns a list of recognised strings.
    """
    if reader is None:
        reader = _easyocr_reader

    if detections is None:
        status_cb(f"[{label}] Detecting text regions with EasyOCR…")
        detections = _detect_regions(image, reader)

    if not detections:
        append_cb(f"\n{'─' * 60}\n{label}\n{'─' * 60}\n[No text detected]\n")
        return []

    append_cb(f"\n{'─' * 60}\n{label}  ({len(detections)} regions found)\n{'─' * 60}\n")

    # ── Phase 1: build crops (fast – no model inference) ──────────────────
    # Each entry is (easy_text, crop_image) for valid crops, or
    # (easy_text, None) for degenerate bounding boxes that are skipped.
    crop_items: list = []
    for bbox, easy_text, _confidence in detections:
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        pad = 4
        x1 = max(0, int(min(xs)) - pad)
        y1 = max(0, int(min(ys)) - pad)
        x2 = min(image.width, int(max(xs)) + pad)
        y2 = min(image.height, int(max(ys)) + pad)

        if (x2 - x1) < 5 or (y2 - y1) < 5:
            crop_items.append((easy_text, None))
        else:
            crop_items.append((easy_text, image.crop((x1, y1, x2, y2))))

    # ── Phase 2: batch TrOCR inference ────────────────────────────────────
    # Process _TROCR_BATCH_SIZE crops per model.generate() call.  This
    # reduces per-call overhead by up to _TROCR_BATCH_SIZE× compared with
    # calling _trocr_read() once per region (the previous approach).
    results = []
    n = len(crop_items)

    for batch_start in range(0, n, _TROCR_BATCH_SIZE):
        batch = crop_items[batch_start:batch_start + _TROCR_BATCH_SIZE]
        batch_end = batch_start + len(batch)

        pct = progress_start + (batch_end / n) * progress_share
        progress_cb(pct)
        status_cb(
            f"[{label}] Reading regions {batch_start + 1}–{batch_end} / {n} with TrOCR…"
        )

        # Collect only the valid (non-None) crops from this batch so that we
        # make a single batched inference call rather than one per crop.
        valid_indices = [i for i, (_, crop) in enumerate(batch) if crop is not None]
        valid_crops = [batch[i][1] for i in valid_indices]

        trocr_texts: List[str] = []
        if valid_crops:
            try:
                trocr_texts = _trocr_read_batch(valid_crops)
            except Exception:
                logging.warning(
                    "[%s] TrOCR batch %d–%d failed — falling back to EasyOCR.\n%s",
                    label, batch_start + 1, batch_end, traceback.format_exc(),
                )
                trocr_texts = [""] * len(valid_crops)

        trocr_iter = iter(trocr_texts)
        for i, (easy_text, crop) in enumerate(batch):
            abs_idx = batch_start + i + 1
            if crop is None:
                # Degenerate bbox — skip silently (same as before)
                continue

            text = next(trocr_iter, "").strip()
            if not text:
                text = easy_text.strip()  # fallback to EasyOCR text

            if text:
                results.append(text)
                append_cb(text + "\n")

    return results


# ---------------------------------------------------------------------------
# Main Application Window
# ---------------------------------------------------------------------------

class HandwritingExtractorApp:
    """Main tkinter application."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Handwriting Text Extractor")
        self.root.geometry("960x720")
        self.root.minsize(720, 560)
        self.root.configure(bg="#f5f5f5")

        self._selected_file: Optional[str] = None
        self._is_processing = False
        self._check_mode = tk.StringVar(value=_CHECK_MODE_PRINTED)
        self._save_debug_crops = tk.BooleanVar(value=False)

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # ── Header ──────────────────────────────────────────────────────
        header = tk.Frame(self.root, bg="#1a237e", pady=18)
        header.pack(fill=tk.X)

        tk.Label(
            header,
            text="✍  Handwriting Text Extractor",
            font=("Segoe UI", 20, "bold"),
            fg="white",
            bg="#1a237e",
        ).pack()

        tk.Label(
            header,
            text="Printed checks: EasyOCR fields  |  Handwritten checks: preprocessed TrOCR fields",
            font=("Segoe UI", 9),
            fg="#90caf9",
            bg="#1a237e",
        ).pack(pady=(2, 0))

        # ── Toolbar ─────────────────────────────────────────────────────
        toolbar = tk.Frame(self.root, bg="#e8eaf6", pady=10)
        toolbar.pack(fill=tk.X, padx=0)

        # Upload button
        self._btn_upload = self._make_button(
            toolbar, "📂  Upload PDF / Image", self._on_upload, "#1565c0"
        )
        self._btn_upload.pack(side=tk.LEFT, padx=(14, 6))

        # Extract button
        self._btn_extract = self._make_button(
            toolbar, "🔍  Extract Text", self._on_extract, "#2e7d32", state=tk.DISABLED
        )
        self._btn_extract.pack(side=tk.LEFT, padx=6)

        # Clear button
        self._btn_clear = self._make_button(
            toolbar, "🗑  Clear", self._on_clear, "#6d4c41", state=tk.DISABLED
        )
        self._btn_clear.pack(side=tk.LEFT, padx=6)

        # Save button (right-aligned)
        self._btn_save = self._make_button(
            toolbar, "💾  Save Results", self._on_save, "#6a1b9a", state=tk.DISABLED
        )
        self._btn_save.pack(side=tk.RIGHT, padx=14)

        # View Log button (right-aligned, always enabled)
        self._btn_log = self._make_button(
            toolbar, "📋  View Log", self._on_view_log, "#455a64"
        )
        self._btn_log.pack(side=tk.RIGHT, padx=(0, 6))

        # File name label
        self._lbl_file = tk.Label(
            toolbar,
            text="No file selected",
            font=("Segoe UI", 9, "italic"),
            fg="#546e7a",
            bg="#e8eaf6",
            anchor="w",
        )
        self._lbl_file.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)

        # ── Check extraction options ──────────────────────────────────────
        options = tk.LabelFrame(
            self.root,
            text="  Check Extraction Options  ",
            font=("Segoe UI", 10, "bold"),
            fg="#1a237e",
            bg="#f5f5f5",
            padx=10,
            pady=8,
            relief=tk.GROOVE,
        )
        options.pack(fill=tk.X, padx=14, pady=(10, 4))

        for mode, label in _CHECK_MODE_LABELS.items():
            tk.Radiobutton(
                options,
                text=label,
                variable=self._check_mode,
                value=mode,
                font=("Segoe UI", 9),
                fg="#263238",
                bg="#f5f5f5",
                activebackground="#f5f5f5",
                anchor="w",
            ).pack(side=tk.LEFT, padx=(0, 18))

        tk.Checkbutton(
            options,
            text="Save debug crops",
            variable=self._save_debug_crops,
            font=("Segoe UI", 9),
            fg="#263238",
            bg="#f5f5f5",
            activebackground="#f5f5f5",
            anchor="w",
        ).pack(side=tk.LEFT, padx=(8, 0))

        # ── Progress area ────────────────────────────────────────────────
        prog_frame = tk.Frame(self.root, bg="#f5f5f5", pady=6)
        prog_frame.pack(fill=tk.X, padx=14)

        self._lbl_status = tk.Label(
            prog_frame,
            text=f"Ready – upload a PDF, image, or ZIP file to begin.  (Log: {_LOG_PATH})",
            font=("Segoe UI", 9),
            fg="#37474f",
            bg="#f5f5f5",
            anchor="w",
        )
        self._lbl_status.pack(fill=tk.X)

        self._progress = ttk.Progressbar(prog_frame, mode="determinate", maximum=100)
        self._progress.pack(fill=tk.X, pady=(4, 0))

        # ── Results area ─────────────────────────────────────────────────
        results_outer = tk.LabelFrame(
            self.root,
            text="  Extracted Text  ",
            font=("Segoe UI", 10, "bold"),
            fg="#1a237e",
            bg="#f5f5f5",
            padx=8,
            pady=8,
            relief=tk.GROOVE,
        )
        results_outer.pack(fill=tk.BOTH, expand=True, padx=14, pady=(4, 14))

        self._txt_results = scrolledtext.ScrolledText(
            results_outer,
            font=("Consolas", 11),
            wrap=tk.WORD,
            bg="#ffffff",
            fg="#212121",
            insertbackground="#1a237e",
            relief=tk.FLAT,
            borderwidth=0,
            state=tk.DISABLED,
        )
        self._txt_results.pack(fill=tk.BOTH, expand=True)

    @staticmethod
    def _make_button(parent, text, command, color, state=tk.NORMAL):
        return tk.Button(
            parent,
            text=text,
            command=command,
            font=("Segoe UI", 10, "bold"),
            bg=color,
            fg="white",
            activebackground=color,
            activeforeground="white",
            relief=tk.FLAT,
            padx=14,
            pady=6,
            cursor="hand2",
            state=state,
        )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_upload(self):
        img_glob = " ".join(f"*{e}" for e in sorted(_SUPPORTED_CHECK_EXTS - {_PDF_EXT}))
        filetypes = [
            ("Supported check files", f"*.pdf {img_glob}"),
            ("PDF files", "*.pdf"),
            ("Image files", img_glob),
            ("All files", "*.*"),
        ]
        path = filedialog.askopenfilename(title="Select scanned check file", filetypes=filetypes)
        if path:
            self._selected_file = path
            name = os.path.basename(path)
            self._lbl_file.config(text=f"📎  {name}", fg="#1a237e")
            self._btn_extract.config(state=tk.NORMAL)
            self._set_status(f"File selected: {name}")
            self._clear_output()

    def _on_extract(self):
        if self._is_processing:
            return
        self._is_processing = True
        self._btn_extract.config(state=tk.DISABLED)
        self._btn_save.config(state=tk.DISABLED)
        self._btn_clear.config(state=tk.DISABLED)
        self._clear_output()
        threading.Thread(target=self._run_extraction, daemon=True).start()

    def _on_clear(self):
        self._clear_output()
        self._set_status("Cleared.")

    def _on_save(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Save Extracted Text",
        )
        if path:
            content = self._txt_results.get("1.0", tk.END)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(content)
            messagebox.showinfo("Saved", f"Results saved to:\n{path}")

    def _on_view_log(self):
        """Open the log file in Notepad (or the default text editor)."""
        if not _LOG_PATH.exists():
            messagebox.showinfo(
                "No Log Yet",
                f"No log file found yet.\nIt will be created at:\n{_LOG_PATH}",
            )
            return
        try:
            if sys.platform == "win32":
                os.startfile(str(_LOG_PATH))  # opens with default .log/.txt handler
            else:
                import subprocess
                subprocess.Popen(["xdg-open", str(_LOG_PATH)])
        except Exception:
            # Fallback: show path so user can open it manually
            messagebox.showinfo("Log File", f"Log file location:\n{_LOG_PATH}")

    # ------------------------------------------------------------------
    # UI helpers (thread-safe via after())
    # ------------------------------------------------------------------

    def _set_status(self, msg: str):
        self.root.after(0, lambda: self._lbl_status.config(text=msg))

    def _set_progress(self, value: float):
        self.root.after(0, lambda: self._progress.config(value=min(value, 100)))

    def _append_text(self, text: str):
        def _do():
            self._txt_results.config(state=tk.NORMAL)
            self._txt_results.insert(tk.END, text)
            self._txt_results.see(tk.END)
            self._txt_results.config(state=tk.DISABLED)

        self.root.after(0, _do)

    def _clear_output(self):
        def _do():
            self._txt_results.config(state=tk.NORMAL)
            self._txt_results.delete("1.0", tk.END)
            self._txt_results.config(state=tk.DISABLED)
            self._btn_save.config(state=tk.DISABLED)
            self._btn_clear.config(state=tk.DISABLED)

        self.root.after(0, _do)

    # ------------------------------------------------------------------
    # Extraction pipeline (runs in background thread)
    # ------------------------------------------------------------------

    def _run_zip_extraction(self, zip_path: str, ocr_reader) -> List[str]:
        """Extract and process every supported file found inside a ZIP archive.

        Files are extracted one at a time into a temporary directory so peak
        disk usage stays low.  PDFs are converted page-by-page; images are
        processed directly.  The temporary directory is deleted when done.

        Returns the combined list of recognised text lines.
        """
        import shutil

        all_results: List[str] = []
        supported_exts = _IMAGE_EXTS | {_PDF_EXT}

        tmp_dir = tempfile.mkdtemp(prefix="HandwritingExtractor_zip_")
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                # Collect supported entries in a single open of the archive
                entries = [
                    e for e in zf.infolist()
                    if not e.is_dir()
                    and not os.path.basename(e.filename).startswith(".")
                    and Path(e.filename).suffix.lower() in supported_exts
                ]

                if not entries:
                    self._append_text(
                        f"ZIP: {os.path.basename(zip_path)}\n"
                        "No supported files (PDF/image) found inside the archive.\n"
                    )
                    return all_results

                total = len(entries)
                self._append_text(
                    f"ZIP : {os.path.basename(zip_path)}\n"
                    f"Files: {total}\n"
                )
                logging.info("ZIP extraction: %d supported files in %s", total, zip_path)

                for file_idx, entry in enumerate(entries):
                    name = os.path.basename(entry.filename)
                    ext = Path(name).suffix.lower()
                    file_share = 100.0 / total
                    progress_start = file_idx * file_share

                    self._set_status(f"[ZIP {file_idx + 1}/{total}] Extracting '{name}'…")
                    self._set_progress(progress_start)

                    # Extract this single entry into the temp dir
                    extracted_path = os.path.join(tmp_dir, name)
                    with zf.open(entry) as src, open(extracted_path, "wb") as dst:
                        dst.write(src.read())

                    if ext == _PDF_EXT:
                        pages = _pdf_to_images(extracted_path)
                        n_pages = len(pages)
                        # Pipeline: submit EasyOCR detection for ALL pages upfront
                        # so up to _WORKER_THREADS detections run simultaneously.
                        # _next_det is always a Future inside the loop because
                        # pages is non-empty (n_pages > 0 was just computed).
                        with concurrent.futures.ThreadPoolExecutor(max_workers=_WORKER_THREADS) as _pool:
                            det_futures = [
                                _pool.submit(_detect_regions, img, ocr_reader)
                                for _, img in pages
                            ]
                            for page_i, (page_num, img) in enumerate(pages):
                                page_share = file_share / n_pages
                                detections = det_futures[page_i].result()
                                results = _extract_from_image(
                                    img,
                                    f"{name} – Page {page_num}/{n_pages}",
                                    progress_cb=self._set_progress,
                                    status_cb=self._set_status,
                                    append_cb=self._append_text,
                                    progress_start=progress_start + page_i * page_share,
                                    progress_share=page_share,
                                    reader=ocr_reader,
                                    detections=detections,
                                )
                                all_results.extend(results)
                    else:
                        img = Image.open(extracted_path).convert("RGB")
                        results = _extract_from_image(
                            img,
                            f"{name} ({file_idx + 1}/{total})",
                            progress_cb=self._set_progress,
                            status_cb=self._set_status,
                            append_cb=self._append_text,
                            progress_start=progress_start,
                            progress_share=file_share,
                            reader=ocr_reader,
                        )
                        all_results.extend(results)

                    # Remove the extracted file immediately to keep disk usage low
                    try:
                        os.remove(extracted_path)
                    except OSError:
                        pass

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        return all_results

    def _run_extraction(self):
        try:
            logging.info("Extraction started for: %s", self._selected_file)
            filepath = self._selected_file
            ext = Path(filepath).suffix.lower()
            mode = self._check_mode.get()
            if mode not in _CHECK_MODE_LABELS:
                mode = _CHECK_MODE_PRINTED

            self._set_progress(0)
            if ext not in _SUPPORTED_CHECK_EXTS:
                raise ValueError(
                    "Unsupported check file type. Please select a PDF, TIF/TIFF, PNG, JPG, or JPEG file."
                )

            ocr_reader = None
            if mode == _CHECK_MODE_HANDWRITTEN:
                ocr_reader = _load_easyocr(self._set_status)
                logging.info("EasyOCR loaded OK")
                _load_trocr(self._set_status)
                logging.info("TrOCR loaded OK")
            else:
                ocr_reader = _load_easyocr(self._set_status)
                logging.info("EasyOCR loaded OK")

            self._set_progress(15)
            self._set_status("Loading and preprocessing check image…")
            img = _load_check_image(filepath)

            self._append_text(
                f"File: {os.path.basename(filepath)}\n"
                f"Mode: {_CHECK_MODE_LABELS[mode]}\n\n"
            )

            self._set_progress(45)
            if mode == _CHECK_MODE_PRINTED:
                self._set_status("Running full-page printed OCR…")
                full_text = extract_printed_check_full_page(img, ocr_reader)
                self._append_text(f"Full Extracted Text:\n\n{full_text}\n")
                non_empty_fields = 1 if full_text else 0
            else:
                self._set_status("Cropping required check fields and running OCR…")
                fields = _extract_check_fields(
                    img,
                    mode,
                    reader=ocr_reader,
                    save_debug_crops=bool(self._save_debug_crops.get()),
                    source_path=filepath,
                )
                self._append_text(_format_check_results(fields))
                non_empty_fields = sum(1 for text in fields.values() if text)

            self._set_progress(100)
            self._set_status(
                f"✅  Check extraction complete — {non_empty_fields} field(s) recognised."
            )
            self.root.after(0, lambda: self._btn_save.config(state=tk.NORMAL))
            self.root.after(0, lambda: self._btn_clear.config(state=tk.NORMAL))

        except BaseException as exc:  # noqa: BLE001 – catch *everything* so nothing is ever silent
            # Re-raise signals and interpreter shutdown requests immediately
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            tb = traceback.format_exc()
            err_msg = str(exc)
            logging.error("Extraction failed:\n%s", tb)

            # Write the full traceback into the results area so it is always visible
            self._append_text(
                f"\n{_ERROR_SEP}\n"
                f"❌  ERROR — extraction failed\n"
                f"{_ERROR_SEP}\n"
                f"{tb}\n"
                f"Log file: {_LOG_PATH}\n"
            )
            self.root.after(0, lambda: self._btn_save.config(state=tk.NORMAL))
            self.root.after(0, lambda: self._btn_clear.config(state=tk.NORMAL))

            self._set_status(f"❌  Error: {err_msg}  |  See log: {_LOG_PATH}")
            self.root.after(
                0,
                lambda msg=err_msg: messagebox.showerror(
                    "Extraction Error",
                    f"An error occurred:\n\n{msg}\n\nFull details written to:\n{_LOG_PATH}",
                ),
            )
        finally:
            self._is_processing = False
            self.root.after(0, lambda: self._btn_extract.config(state=tk.NORMAL))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    root = tk.Tk()
    # High-DPI awareness on Windows
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    # Keep a reference on root so the GC doesn't collect the app before mainloop exits
    root._app = HandwritingExtractorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
