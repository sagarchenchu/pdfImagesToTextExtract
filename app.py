"""
Handwriting Text Extractor
===========================
Extracts handwritten text from PDF files and images using:
  - EasyOCR  : layout / line-region detection
  - TrOCR    : Microsoft's transformer OCR model for handwriting recognition

GUI built with tkinter (ships with Python – no extra install needed).
Packaged to a Windows .exe with PyInstaller via the included spec file.
"""

import errno as _errno
import io
import logging
import os
import sys
import tempfile
import threading
import traceback
from pathlib import Path
from typing import List, Optional, Set

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
from PIL import Image

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

TROCR_MODEL = "microsoft/trocr-large-handwritten"


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


def _bundled_easyocr_model_dir() -> Optional[str]:
    """Return the bundled EasyOCR model directory when running as a frozen EXE."""
    if getattr(sys, "frozen", False):
        bundled = Path(sys._MEIPASS) / "models" / "easyocr"  # type: ignore[attr-defined]
        if bundled.exists():
            return str(bundled)
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
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        frozen = getattr(sys, "frozen", False)

        if frozen:
            # Running as bundled EXE — models are stored in a flat directory
            # (config.json, model weights, tokenizer files …) inside the bundle.
            # Pass the local path directly to from_pretrained(); do NOT rely on
            # HF_HUB_CACHE which expects the blob-cache layout, not a flat layout.
            status_cb(f"Loading TrOCR model '{TROCR_MODEL}' from bundle…")
            bundle_dir = Path(sys._MEIPASS)  # type: ignore[attr-defined]
            trocr_local = str(bundle_dir / "models" / "trocr")
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
                status_cb(f"Downloading TrOCR model '{TROCR_MODEL}' (~1 GB, one-time download)…")
                try:
                    _trocr_processor = TrOCRProcessor.from_pretrained(TROCR_MODEL)
                    _trocr_model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL)
                except Exception as exc:
                    if _is_connection_error(exc):
                        raise ConnectionError(
                            "Could not download the TrOCR model — check your internet connection and try again.\n"
                            "The model (~1 GB) needs to be downloaded once before it can be used offline.\n\n"
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
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        images.append((page_num + 1, img))
    doc.close()
    return images


def _trocr_read(image_crop: Image.Image) -> str:
    """Run TrOCR inference on a single cropped line/region image."""
    import torch
    processor, model = _trocr_processor, _trocr_model
    device = _get_device()
    pixel_values = processor(image_crop, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        ids = model.generate(pixel_values)
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


def _extract_from_image(
    image: Image.Image,
    label: str,
    progress_cb,
    status_cb,
    append_cb,
    progress_start: float = 0.0,
    progress_share: float = 100.0,
    reader=None,
) -> list:
    """
    Detect text regions with EasyOCR, read each with TrOCR.

    Parameters
    ----------
    reader:
        An initialised ``easyocr.Reader`` instance.  When *None* (default)
        the module-level ``_easyocr_reader`` global is used; callers should
        prefer passing the reader explicitly to avoid relying on global state.

    Returns a list of recognised strings.
    """
    if reader is None:
        reader = _easyocr_reader
    img_array = np.array(image)

    status_cb(f"[{label}] Detecting text regions with EasyOCR…")
    detections = reader.readtext(img_array, detail=1, paragraph=False)

    if not detections:
        append_cb(f"\n{'─' * 60}\n{label}\n{'─' * 60}\n[No text detected]\n")
        return []

    append_cb(f"\n{'─' * 60}\n{label}  ({len(detections)} regions found)\n{'─' * 60}\n")

    results = []
    n = len(detections)

    for idx, (bbox, easy_text, _confidence) in enumerate(detections):
        pct = progress_start + ((idx + 1) / n) * progress_share
        progress_cb(pct)
        status_cb(f"[{label}] Reading region {idx + 1} / {n} with TrOCR…")

        try:
            # Build crop coordinates from the polygon returned by EasyOCR
            xs = [pt[0] for pt in bbox]
            ys = [pt[1] for pt in bbox]
            pad = 4
            x1 = max(0, int(min(xs)) - pad)
            y1 = max(0, int(min(ys)) - pad)
            x2 = min(image.width, int(max(xs)) + pad)
            y2 = min(image.height, int(max(ys)) + pad)

            if (x2 - x1) < 5 or (y2 - y1) < 5:
                continue

            crop = image.crop((x1, y1, x2, y2))
            text = _trocr_read(crop)

            if not text:
                text = easy_text.strip()  # fallback to EasyOCR text

            if text:
                results.append(text)
                append_cb(text + "\n")

        except Exception as exc:  # noqa: BLE001
            logging.warning(
                "[%s] TrOCR failed on region %d/%d — falling back to EasyOCR text.\n%s",
                label, idx + 1, n, traceback.format_exc(),
            )
            fallback = easy_text.strip()
            if fallback:
                results.append(fallback)
                append_cb(fallback + "\n")
            status_cb(f"[{label}] Warning on region {idx + 1}: {exc}")

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
            text="EasyOCR (layout detection)  +  TrOCR (handwriting recognition)  |  PDF & Image support",
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

        # ── Progress area ────────────────────────────────────────────────
        prog_frame = tk.Frame(self.root, bg="#f5f5f5", pady=6)
        prog_frame.pack(fill=tk.X, padx=14)

        self._lbl_status = tk.Label(
            prog_frame,
            text=f"Ready – upload a PDF or image file to begin.  (Log: {_LOG_PATH})",
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
        filetypes = [
            ("Supported files", "*.pdf *.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp"),
            ("PDF files", "*.pdf"),
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp"),
            ("All files", "*.*"),
        ]
        path = filedialog.askopenfilename(title="Select PDF or Image", filetypes=filetypes)
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

    def _run_extraction(self):
        try:
            logging.info("Extraction started for: %s", self._selected_file)
            # Load models and capture the returned objects explicitly so we
            # never depend on the module-level globals being readable from
            # another call stack (e.g. after an edge-case exception path).
            ocr_reader = _load_easyocr(self._set_status)
            logging.info("EasyOCR loaded OK")
            _load_trocr(self._set_status)
            logging.info("TrOCR loaded OK")

            filepath = self._selected_file
            ext = Path(filepath).suffix.lower()
            all_results: List[str] = []

            self._set_progress(0)

            if ext == ".pdf":
                self._set_status("Converting PDF pages to images…")
                pages = _pdf_to_images(filepath)
                total = len(pages)
                self._append_text(
                    f"File : {os.path.basename(filepath)}\n"
                    f"Pages: {total}\n"
                )
                for i, (page_num, img) in enumerate(pages):
                    share = 100.0 / total
                    results = _extract_from_image(
                        img,
                        f"Page {page_num}/{total}",
                        progress_cb=self._set_progress,
                        status_cb=self._set_status,
                        append_cb=self._append_text,
                        progress_start=i * share,
                        progress_share=share,
                        reader=ocr_reader,
                    )
                    all_results.extend(results)
            else:
                img = Image.open(filepath).convert("RGB")
                self._append_text(f"File: {os.path.basename(filepath)}\n")
                all_results = _extract_from_image(
                    img,
                    "Image",
                    progress_cb=self._set_progress,
                    status_cb=self._set_status,
                    append_cb=self._append_text,
                    progress_start=0.0,
                    progress_share=100.0,
                    reader=ocr_reader,
                )

            self._set_progress(100)
            self._set_status(
                f"✅  Extraction complete — {len(all_results)} text line(s) recognised."
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
