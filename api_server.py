"""
Text Extract API Server
=======================
Exposes a single HTTP endpoint on port 9102:

    POST http://127.0.0.1:9102/extractText

Send a PDF or image file as a multipart/form-data upload using the field
name ``file``.  The response body is the extracted text (UTF-8 plain text).

Supported input formats: PDF (.pdf), PNG, JPG/JPEG, BMP, TIFF/TIF, WebP.

Packaged to a Windows .exe with PyInstaller via api_server.spec.
No GUI — run the EXE from a command prompt or as a background service.
"""

import concurrent.futures
import errno as _errno
import io
import logging
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import List, Optional, Set

# ---------------------------------------------------------------------------
# Ensure stdout/stderr are never None (frozen console=True EXEs are fine,
# but keep the guard for defence-in-depth when launched without a console).
# ---------------------------------------------------------------------------
if sys.stdout is None or sys.stderr is None:
    _devnull = open(os.devnull, "w", encoding="utf-8", errors="replace")  # noqa: WPS515
    if sys.stdout is None:
        sys.stdout = _devnull
    if sys.stderr is None:
        sys.stderr = _devnull

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# File-based logging
# ---------------------------------------------------------------------------
_LOG_PATH = Path(tempfile.gettempdir()) / "TextExtractAPI.log"


def _setup_logging() -> None:
    log_dir = _LOG_PATH.parent
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError as err:
        sys.stderr.write(f"WARNING: could not create log directory {log_dir}: {err}\n")

    logging.basicConfig(
        filename=str(_LOG_PATH),
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        encoding="utf-8",
        errors="replace",
        force=True,
    )
    logging.info("=" * 60)
    logging.info("TextExtractAPI started")
    logging.info("Python %s", sys.version)
    logging.info("frozen=%s", getattr(sys, "frozen", False))
    if getattr(sys, "frozen", False):
        logging.info("_MEIPASS=%s", getattr(sys, "_MEIPASS", "n/a"))
    logging.info("Log file: %s", _LOG_PATH)
    logging.info("=" * 60)


_setup_logging()

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Lazy-load heavy ML libraries on first request
# ---------------------------------------------------------------------------
_easyocr_reader = None
_trocr_processor = None
_trocr_model = None
_device = None

TROCR_MODEL = "microsoft/trocr-large-handwritten"

_IMAGE_EXTS: frozenset = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
)
_PDF_EXT: str = ".pdf"

_TROCR_BATCH_SIZE = 16
_WORKER_THREADS = 10

_NETWORK_ERRNO: frozenset = frozenset({
    _errno.ECONNREFUSED,
    _errno.ETIMEDOUT,
    _errno.ENETUNREACH,
    _errno.EHOSTUNREACH,
    _errno.ECONNRESET,
})


# ---------------------------------------------------------------------------
# Helpers shared with the GUI app (duplicated to keep api_server.py standalone
# and avoid pulling in tkinter via app.py's module-level imports)
# ---------------------------------------------------------------------------

def _get_device():
    global _device
    if _device is None:
        try:
            import torch
            _device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            _device = "cpu"
    return _device


def _is_connection_error(exc: BaseException) -> bool:
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
    if not getattr(sys, "frozen", False):
        return None

    sideloaded = Path(sys.executable).parent / "models"
    if sideloaded.exists():
        logging.info("Using sideloaded models from %s", sideloaded)
        return sideloaded

    meipass = getattr(sys, "_MEIPASS", None)
    if meipass is not None:
        bundled = Path(meipass) / "models"
        if bundled.exists():
            logging.info("Using bundled models from %s", bundled)
            return bundled

    return None


def _bundled_easyocr_model_dir() -> Optional[str]:
    models_dir = _resolve_models_dir()
    if models_dir is not None:
        easyocr_dir = models_dir / "easyocr"
        if easyocr_dir.exists():
            return str(easyocr_dir)
    return None


def _warm_torchvision() -> None:
    try:
        import torchvision                        # noqa: F401
        import torchvision.models                 # noqa: F401
        import torchvision.ops                    # noqa: F401
        import torchvision.transforms             # noqa: F401
        import torchvision.transforms.functional  # noqa: F401
    except Exception:
        logging.warning("torchvision pre-import warning:\n%s", traceback.format_exc())


def _load_easyocr():
    global _easyocr_reader
    if _easyocr_reader is None:
        logging.info("Loading EasyOCR model…")
        _warm_torchvision()
        import easyocr
        kwargs: dict = {"gpu": _get_device() == "cuda"}
        model_dir = _bundled_easyocr_model_dir()
        if model_dir:
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


def _load_trocr():
    global _trocr_processor, _trocr_model
    if _trocr_processor is None or _trocr_model is None:
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
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
            models_dir = _resolve_models_dir()
            if models_dir is None:
                raise FileNotFoundError(
                    "models/ directory not found.\n\n"
                    "Place the models/ folder next to TextExtractAPI.exe:\n"
                    "  TextExtractAPI.exe\n"
                    "  models\\\n"
                    "    trocr\\\n"
                    "      config.json, pytorch_model.bin, …\n"
                    "    easyocr\\\n"
                    "      craft_mlt_25k.pth, english_g2.pth\n\n"
                    "Download links are in the project README."
                )
            logging.info("Frozen EXE: loading TrOCR from %s", models_dir / "trocr")
            trocr_local = str(models_dir / "trocr")
            _trocr_processor = TrOCRProcessor.from_pretrained(trocr_local, local_files_only=True)
            _trocr_model = VisionEncoderDecoderModel.from_pretrained(trocr_local, local_files_only=True)
        else:
            logging.info("Loading TrOCR model '%s'…", TROCR_MODEL)
            try:
                _trocr_processor = TrOCRProcessor.from_pretrained(TROCR_MODEL, local_files_only=True)
                _trocr_model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL, local_files_only=True)
            except OSError:
                logging.info("TrOCR not cached — downloading (~1 GB, one-time)…")
                try:
                    _trocr_processor = TrOCRProcessor.from_pretrained(TROCR_MODEL)
                    _trocr_model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL)
                except Exception as exc:
                    if _is_connection_error(exc):
                        raise ConnectionError(
                            "Could not download the TrOCR model — check your internet connection.\n"
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


def _trocr_read_batch(image_crops: List[Image.Image]) -> List[str]:
    import torch
    processor, model = _trocr_processor, _trocr_model
    device = _get_device()
    pixel_values = processor(image_crops, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        ids = model.generate(pixel_values)
    return [s.strip() for s in processor.batch_decode(ids, skip_special_tokens=True)]


def _detect_regions(image: Image.Image, reader) -> list:
    img_array = np.array(image)
    return reader.readtext(img_array, detail=1, paragraph=False)


def _extract_from_image(
    image: Image.Image,
    reader=None,
    detections=None,
) -> List[str]:
    """Detect text regions with EasyOCR and read each with TrOCR.

    Returns a list of recognised text strings.
    """
    if reader is None:
        reader = _easyocr_reader

    if detections is None:
        detections = _detect_regions(image, reader)

    if not detections:
        return []

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

    results: List[str] = []
    n = len(crop_items)

    for batch_start in range(0, n, _TROCR_BATCH_SIZE):
        batch = crop_items[batch_start:batch_start + _TROCR_BATCH_SIZE]

        valid_indices = [i for i, (_, crop) in enumerate(batch) if crop is not None]
        valid_crops = [batch[i][1] for i in valid_indices]

        trocr_texts: List[str] = []
        if valid_crops:
            try:
                trocr_texts = _trocr_read_batch(valid_crops)
            except Exception:
                logging.warning(
                    "TrOCR batch %d–%d failed — falling back to EasyOCR.\n%s",
                    batch_start + 1, batch_start + len(batch), traceback.format_exc(),
                )
                trocr_texts = [""] * len(valid_crops)

        trocr_iter = iter(trocr_texts)
        for easy_text, crop in batch:
            if crop is None:
                continue
            text = next(trocr_iter, "").strip()
            if not text:
                text = easy_text.strip()
            if text:
                results.append(text)

    return results


# ---------------------------------------------------------------------------
# High-level extraction entry point used by the API handler
# ---------------------------------------------------------------------------

def extract_text_from_file(file_path: str) -> str:
    """Load models (lazy), run extraction, return all recognised text joined by newlines."""
    ocr_reader = _load_easyocr()
    _load_trocr()

    ext = Path(file_path).suffix.lower()
    all_lines: List[str] = []

    if ext == _PDF_EXT:
        pages = _pdf_to_images(file_path)
        n_pages = len(pages)
        with concurrent.futures.ThreadPoolExecutor(max_workers=_WORKER_THREADS) as pool:
            det_futures = [
                pool.submit(_detect_regions, img, ocr_reader)
                for _, img in pages
            ]
            for page_i, (page_num, img) in enumerate(pages):
                detections = det_futures[page_i].result()
                lines = _extract_from_image(img, reader=ocr_reader, detections=detections)
                all_lines.extend(lines)
        logging.info("PDF extraction: %d pages, %d lines", n_pages, len(all_lines))
    elif ext in _IMAGE_EXTS:
        img = Image.open(file_path).convert("RGB")
        all_lines = _extract_from_image(img, reader=ocr_reader)
        logging.info("Image extraction: %d lines", len(all_lines))
    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Accepted: .pdf, {', '.join(sorted(_IMAGE_EXTS))}"
        )

    return "\n".join(all_lines)


# ---------------------------------------------------------------------------
# Flask API
# ---------------------------------------------------------------------------

_HOST = "127.0.0.1"
_PORT = 9102

from flask import Flask, request, Response  # noqa: E402

app = Flask(__name__)


@app.route("/extractText", methods=["POST"])
def extract_text():
    """Accept a PDF or image file upload and return the extracted text.

    Request:
        POST /extractText
        Content-Type: multipart/form-data
        Body field: file=<PDF or image file>

    Response (200):
        Content-Type: text/plain; charset=utf-8
        Body: extracted text, one recognised line per text line

    Error responses:
        400  — no file supplied or unsupported file type
        500  — extraction failed (details in response body and log file)
    """
    if "file" not in request.files:
        return Response("No file provided. Send a PDF or image as the 'file' field.\n", status=400, mimetype="text/plain")

    upload = request.files["file"]
    if upload.filename == "":
        return Response("Empty filename. Please attach a valid PDF or image file.\n", status=400, mimetype="text/plain")

    suffix = Path(upload.filename).suffix.lower()
    allowed = _IMAGE_EXTS | {_PDF_EXT}
    if suffix not in allowed:
        return Response(
            f"Unsupported file type '{suffix}'.\nAccepted: .pdf, {', '.join(sorted(_IMAGE_EXTS))}\n",
            status=400,
            mimetype="text/plain",
        )

    # Save the upload to a temp file so the ML pipeline can read it from disk.
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            upload.save(tmp_path)
    except OSError as exc:
        logging.error("Failed to save upload to temp file: %s", exc)
        return Response(f"Server error while saving upload: {exc}\n", status=500, mimetype="text/plain")

    try:
        logging.info("extractText request: file='%s' saved to '%s'", upload.filename, tmp_path)
        text = extract_text_from_file(tmp_path)
        logging.info("extractText complete: %d chars extracted", len(text))
        return Response(text, status=200, mimetype="text/plain; charset=utf-8")
    except ValueError as exc:
        logging.warning("extractText bad request: %s", exc)
        return Response(str(exc) + "\n", status=400, mimetype="text/plain")
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        logging.error("extractText error:\n%s", tb)
        return Response(
            f"Extraction failed: {exc}\n\nSee log: {_LOG_PATH}\n",
            status=500,
            mimetype="text/plain",
        )
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print(f"TextExtractAPI starting on http://{_HOST}:{_PORT}")
    print(f"Endpoint: POST http://{_HOST}:{_PORT}/extractText")
    print(f"Log file: {_LOG_PATH}")
    logging.info("Starting Flask server on %s:%d", _HOST, _PORT)
    # use_reloader=False is required inside a frozen EXE (no source files to watch)
    app.run(host=_HOST, port=_PORT, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
