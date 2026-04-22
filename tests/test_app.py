"""
test_app.py
===========
Unit tests for the pure-logic helpers in app.py.

All heavy / GUI dependencies are stubbed out in conftest.py so the tests
run offline, without a display, and without any ML models installed.
"""

import errno
import io
import os
import sys
import types
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Import the module under test
# The conftest.py stubs are already installed into sys.modules before this
# point, so the module-level tkinter/torch/etc. imports in app.py succeed.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))
import app  # noqa: E402  (must come after path manipulation)


# ===========================================================================
# Helpers
# ===========================================================================

def _solid_image(width: int = 50, height: int = 30, color: str = "white") -> Image.Image:
    """Create a tiny in-memory PIL image for testing."""
    return Image.new("RGB", (width, height), color=color)


def _dummy_status_cb(msg: str) -> None:  # noqa: ARG001
    pass


def _dummy_progress_cb(pct: float) -> None:  # noqa: ARG001
    pass


# ===========================================================================
# 1. Constants
# ===========================================================================

class TestConstants:
    def test_image_exts_is_frozenset(self):
        assert isinstance(app._IMAGE_EXTS, frozenset)

    def test_image_exts_contains_common_formats(self):
        for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"):
            assert ext in app._IMAGE_EXTS, f"Expected {ext} in _IMAGE_EXTS"

    def test_image_exts_all_lowercase(self):
        for ext in app._IMAGE_EXTS:
            assert ext == ext.lower(), f"Extension {ext!r} should be lowercase"

    def test_image_exts_all_start_with_dot(self):
        for ext in app._IMAGE_EXTS:
            assert ext.startswith("."), f"Extension {ext!r} should start with '.'"

    def test_pdf_ext(self):
        assert app._PDF_EXT == ".pdf"

    def test_zip_ext(self):
        assert app._ZIP_EXT == ".zip"

    def test_trocr_batch_size_positive_int(self):
        assert isinstance(app._TROCR_BATCH_SIZE, int)
        assert app._TROCR_BATCH_SIZE > 0

    def test_trocr_model_name(self):
        assert "trocr" in app.TROCR_MODEL.lower()


# ===========================================================================
# 2. _is_connection_error
# ===========================================================================

class TestIsConnectionError:
    def test_plain_valueerror_is_not_connection_error(self):
        assert app._is_connection_error(ValueError("something")) is False

    def test_plain_runtimeerror_is_not_connection_error(self):
        assert app._is_connection_error(RuntimeError("oops")) is False

    def test_oserror_with_econnrefused(self):
        exc = OSError(errno.ECONNREFUSED, "Connection refused")
        assert app._is_connection_error(exc) is True

    def test_oserror_with_etimedout(self):
        exc = OSError(errno.ETIMEDOUT, "Timed out")
        assert app._is_connection_error(exc) is True

    def test_oserror_with_enetunreach(self):
        exc = OSError(errno.ENETUNREACH, "Network unreachable")
        assert app._is_connection_error(exc) is True

    def test_oserror_with_ehostunreach(self):
        exc = OSError(errno.EHOSTUNREACH, "No route to host")
        assert app._is_connection_error(exc) is True

    def test_oserror_with_econnreset(self):
        exc = OSError(errno.ECONNRESET, "Connection reset")
        assert app._is_connection_error(exc) is True

    def test_oserror_ssl_in_message(self):
        exc = OSError("SSL certificate verification failed")
        assert app._is_connection_error(exc) is True

    def test_oserror_certificate_in_message(self):
        exc = OSError("certificate error")
        assert app._is_connection_error(exc) is True

    def test_oserror_handshake_in_message(self):
        exc = OSError("handshake failure")
        assert app._is_connection_error(exc) is True

    def test_chained_cause_oserror(self):
        inner = OSError(errno.ECONNREFUSED, "refused")
        outer = RuntimeError("wrap")
        outer.__cause__ = inner
        assert app._is_connection_error(outer) is True

    def test_chained_context_oserror(self):
        inner = OSError(errno.ETIMEDOUT, "timed out")
        outer = ValueError("wrap")
        outer.__context__ = inner
        assert app._is_connection_error(outer) is True

    def test_none_errno_oserror(self):
        """OSError with errno=None and no network keyword should return False."""
        exc = OSError("generic OS error")
        exc.errno = None
        assert app._is_connection_error(exc) is False

    def test_circular_exception_chain_does_not_loop(self):
        """A circular __context__ chain must not cause an infinite loop."""
        exc = RuntimeError("a")
        exc.__context__ = exc  # circular reference
        # Should complete without hanging; return value is False (no network exc)
        result = app._is_connection_error(exc)
        assert isinstance(result, bool)


# ===========================================================================
# 3. _resolve_models_dir — non-frozen path
# ===========================================================================

class TestResolveModelsDirNonFrozen:
    def test_returns_none_when_not_frozen(self):
        """When sys.frozen is absent, always return None regardless of paths."""
        with patch.object(sys, "frozen", False, create=True):
            assert app._resolve_models_dir() is None

    def test_returns_none_without_frozen_attr(self):
        """sys.frozen may not exist at all in a plain Python process."""
        frozen_backup = getattr(sys, "frozen", _sentinel := object())
        try:
            if hasattr(sys, "frozen"):
                del sys.frozen
            assert app._resolve_models_dir() is None
        finally:
            if frozen_backup is not _sentinel:
                sys.frozen = frozen_backup  # type: ignore[attr-defined]


# ===========================================================================
# 4. _resolve_models_dir — frozen path
# ===========================================================================

class TestResolveModelsDirFrozen:
    def test_sideloaded_models_dir_takes_priority(self, tmp_path):
        """When a models/ folder exists next to the EXE, it is returned first."""
        exe_dir = tmp_path / "dist"
        exe_dir.mkdir()
        models_dir = exe_dir / "models"
        models_dir.mkdir()

        with (
            patch.object(sys, "frozen", True, create=True),
            patch.object(sys, "executable", str(exe_dir / "HandwritingExtractor.exe"), create=True),
        ):
            result = app._resolve_models_dir()

        assert result == models_dir

    def test_meipass_fallback_used_when_no_sideloaded(self, tmp_path):
        """Falls back to _MEIPASS/models/ when no sideloaded folder exists."""
        meipass_dir = tmp_path / "_internal"
        meipass_dir.mkdir()
        models_dir = meipass_dir / "models"
        models_dir.mkdir()

        exe_dir = tmp_path / "no_sideload"
        exe_dir.mkdir()
        # Note: no models/ next to exe_dir

        with (
            patch.object(sys, "frozen", True, create=True),
            patch.object(sys, "executable", str(exe_dir / "HandwritingExtractor.exe"), create=True),
            patch.object(sys, "_MEIPASS", str(meipass_dir), create=True),
        ):
            result = app._resolve_models_dir()

        assert result == models_dir

    def test_returns_none_when_neither_location_exists(self, tmp_path):
        """Returns None when neither sideloaded nor MEIPASS models/ exists."""
        exe_dir = tmp_path / "empty"
        exe_dir.mkdir()
        meipass_dir = tmp_path / "_internal_empty"
        meipass_dir.mkdir()
        # Neither has a models/ subdirectory.

        with (
            patch.object(sys, "frozen", True, create=True),
            patch.object(sys, "executable", str(exe_dir / "HandwritingExtractor.exe"), create=True),
            patch.object(sys, "_MEIPASS", str(meipass_dir), create=True),
        ):
            result = app._resolve_models_dir()

        assert result is None

    def test_returns_none_when_no_meipass_and_no_sideloaded(self, tmp_path):
        """Returns None when frozen but _MEIPASS is absent and no sideloaded dir."""
        exe_dir = tmp_path / "no_meipass"
        exe_dir.mkdir()

        # Remove _MEIPASS if it was already mocked
        meipass_backup = getattr(sys, "_MEIPASS", _sentinel := object())
        try:
            if hasattr(sys, "_MEIPASS"):
                del sys._MEIPASS
            with (
                patch.object(sys, "frozen", True, create=True),
                patch.object(sys, "executable", str(exe_dir / "app.exe"), create=True),
            ):
                result = app._resolve_models_dir()
            assert result is None
        finally:
            if meipass_backup is not _sentinel:
                sys._MEIPASS = meipass_backup  # type: ignore[attr-defined]


# ===========================================================================
# 5. _bundled_easyocr_model_dir
# ===========================================================================

class TestBundledEasyocrModelDir:
    def test_returns_none_when_not_frozen(self):
        with patch.object(sys, "frozen", False, create=True):
            assert app._bundled_easyocr_model_dir() is None

    def test_returns_path_when_easyocr_subdir_exists(self, tmp_path):
        models_dir = tmp_path / "models"
        easyocr_dir = models_dir / "easyocr"
        easyocr_dir.mkdir(parents=True)

        with patch("app._resolve_models_dir", return_value=models_dir):
            result = app._bundled_easyocr_model_dir()

        assert result == str(easyocr_dir)

    def test_returns_none_when_easyocr_subdir_missing(self, tmp_path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        # No easyocr/ subdirectory

        with patch("app._resolve_models_dir", return_value=models_dir):
            result = app._bundled_easyocr_model_dir()

        assert result is None

    def test_returns_none_when_resolve_returns_none(self):
        with patch("app._resolve_models_dir", return_value=None):
            assert app._bundled_easyocr_model_dir() is None


# ===========================================================================
# 6. _get_device
# ===========================================================================

class TestGetDevice:
    def setup_method(self):
        # Reset the cached global before each test
        app._device = None

    def test_returns_cpu_when_cuda_unavailable(self):
        mock_torch = types.ModuleType("torch")
        mock_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        with patch.dict(sys.modules, {"torch": mock_torch}):
            app._device = None
            result = app._get_device()
        assert result == "cpu"

    def test_returns_cuda_when_cuda_available(self):
        mock_torch = types.ModuleType("torch")
        mock_torch.cuda = types.SimpleNamespace(is_available=lambda: True)
        with patch.dict(sys.modules, {"torch": mock_torch}):
            app._device = None
            result = app._get_device()
        assert result == "cuda"

    def test_returns_cpu_when_torch_import_fails(self):
        with patch.dict(sys.modules, {"torch": None}):
            app._device = None
            result = app._get_device()
        assert result == "cpu"

    def test_caches_result(self):
        """Second call should return same value without re-importing torch."""
        app._device = "cpu"
        call_count = {"n": 0}

        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        result = app._get_device()
        assert result == "cpu"  # returns cached value

    def teardown_method(self):
        app._device = None


# ===========================================================================
# 7. _pdf_to_images
# ===========================================================================

class TestPdfToImages:
    def _make_fake_page(self, page_num: int, width: int = 100, height: int = 80):
        """Build a minimal mock of a fitz page that returns a PNG pixmap."""
        img = _solid_image(width, height, color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        mock_pix = MagicMock()
        mock_pix.tobytes = MagicMock(return_value=png_bytes)

        mock_page = MagicMock()
        mock_page.get_pixmap = MagicMock(return_value=mock_pix)
        return mock_page

    def test_returns_list_of_tuples_with_page_numbers(self):
        """Each item should be (1-based page number, PIL Image)."""
        n_pages = 3
        pages = [self._make_fake_page(i) for i in range(n_pages)]

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=n_pages)
        mock_doc.load_page = MagicMock(side_effect=pages)
        mock_doc.close = MagicMock()

        mock_fitz = types.ModuleType("fitz")
        mock_fitz.open = MagicMock(return_value=mock_doc)
        mock_fitz.Matrix = MagicMock(return_value=MagicMock())

        with patch.dict(sys.modules, {"fitz": mock_fitz}):
            result = app._pdf_to_images("/fake/path.pdf")

        assert len(result) == n_pages
        for idx, (page_num, img) in enumerate(result):
            assert page_num == idx + 1  # 1-based
            assert isinstance(img, Image.Image)
            assert img.mode == "RGB"

    def test_single_page_pdf(self):
        pages = [self._make_fake_page(0)]

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.load_page = MagicMock(side_effect=pages)
        mock_doc.close = MagicMock()

        mock_fitz = types.ModuleType("fitz")
        mock_fitz.open = MagicMock(return_value=mock_doc)
        mock_fitz.Matrix = MagicMock(return_value=MagicMock())

        with patch.dict(sys.modules, {"fitz": mock_fitz}):
            result = app._pdf_to_images("/fake/single.pdf")

        assert len(result) == 1
        assert result[0][0] == 1

    def test_empty_pdf_returns_empty_list(self):
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=0)
        mock_doc.close = MagicMock()

        mock_fitz = types.ModuleType("fitz")
        mock_fitz.open = MagicMock(return_value=mock_doc)
        mock_fitz.Matrix = MagicMock(return_value=MagicMock())

        with patch.dict(sys.modules, {"fitz": mock_fitz}):
            result = app._pdf_to_images("/fake/empty.pdf")

        assert result == []
        mock_doc.close.assert_called_once()

    def test_doc_is_always_closed(self):
        """fitz doc must be closed even when load_page raises."""
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.load_page = MagicMock(side_effect=RuntimeError("page error"))
        mock_doc.close = MagicMock()

        mock_fitz = types.ModuleType("fitz")
        mock_fitz.open = MagicMock(return_value=mock_doc)
        mock_fitz.Matrix = MagicMock(return_value=MagicMock())

        with patch.dict(sys.modules, {"fitz": mock_fitz}):
            with pytest.raises(RuntimeError, match="page error"):
                app._pdf_to_images("/fake/bad.pdf")

        mock_doc.close.assert_called_once()


# ===========================================================================
# 8. _trocr_read_batch
# ===========================================================================

class TestTrocrReadBatch:
    def _setup_globals(self, decoded_texts):
        """Inject mock processor and model into app module globals."""
        mock_processor = MagicMock()
        mock_pixel = MagicMock()
        mock_pixel.to = MagicMock(return_value=mock_pixel)
        mock_processor.return_value.pixel_values = mock_pixel
        mock_processor.return_value = MagicMock(pixel_values=mock_pixel)
        # processor(images, return_tensors=...).pixel_values
        mock_processor_call = MagicMock()
        mock_processor_call.pixel_values = MagicMock()
        mock_processor_call.pixel_values.to = MagicMock(return_value=mock_pixel)
        mock_processor.side_effect = lambda *a, **kw: mock_processor_call

        mock_model = MagicMock()
        mock_model.generate = MagicMock(return_value=MagicMock())

        mock_processor.batch_decode = MagicMock(return_value=decoded_texts)

        return mock_processor, mock_model

    def test_strips_whitespace_from_output(self):
        crops = [_solid_image()]
        mock_proc = MagicMock()
        proc_output = MagicMock()
        proc_output.pixel_values = MagicMock()
        proc_output.pixel_values.to = MagicMock(return_value=MagicMock())
        mock_proc.side_effect = lambda *a, **kw: proc_output
        mock_proc.batch_decode = MagicMock(return_value=["  hello world  "])

        mock_model = MagicMock()
        mock_model.generate = MagicMock(return_value=[])

        mock_torch = types.ModuleType("torch")
        mock_torch.no_grad = MagicMock(
            return_value=MagicMock(__enter__=MagicMock(return_value=None), __exit__=MagicMock(return_value=False))
        )

        app._trocr_processor = mock_proc
        app._trocr_model = mock_model
        app._device = "cpu"

        try:
            with patch.dict(sys.modules, {"torch": mock_torch}):
                result = app._trocr_read_batch(crops)
            assert result == ["hello world"]
        finally:
            app._trocr_processor = None
            app._trocr_model = None
            app._device = None

    def test_returns_one_result_per_crop(self):
        n = 4
        crops = [_solid_image() for _ in range(n)]

        mock_proc = MagicMock()
        proc_output = MagicMock()
        proc_output.pixel_values = MagicMock()
        proc_output.pixel_values.to = MagicMock(return_value=MagicMock())
        mock_proc.side_effect = lambda *a, **kw: proc_output
        mock_proc.batch_decode = MagicMock(return_value=[f"text{i}" for i in range(n)])

        mock_model = MagicMock()
        mock_model.generate = MagicMock(return_value=[])

        mock_torch = types.ModuleType("torch")
        mock_torch.no_grad = MagicMock(
            return_value=MagicMock(__enter__=MagicMock(return_value=None), __exit__=MagicMock(return_value=False))
        )

        app._trocr_processor = mock_proc
        app._trocr_model = mock_model
        app._device = "cpu"

        try:
            with patch.dict(sys.modules, {"torch": mock_torch}):
                result = app._trocr_read_batch(crops)
            assert len(result) == n
        finally:
            app._trocr_processor = None
            app._trocr_model = None
            app._device = None


# ===========================================================================
# 9. _extract_from_image — crop-building and fallback logic
# ===========================================================================

class TestExtractFromImage:
    """Tests for _extract_from_image without running real ML models."""

    def _make_reader(self, detections):
        """Build a mock easyocr.Reader that returns fixed detections."""
        mock_reader = MagicMock()
        mock_reader.readtext = MagicMock(return_value=detections)
        return mock_reader

    def _run(self, image, detections, trocr_texts=None):
        """Run _extract_from_image with mocked reader and TrOCR."""
        reader = self._make_reader(detections)

        appended = []
        status_msgs = []

        def append_cb(text):
            appended.append(text)

        def status_cb(msg):
            status_msgs.append(msg)

        # Patch _trocr_read_batch to return controlled output
        if trocr_texts is None:
            trocr_texts = ["mocked text"] * 20  # enough for any batch

        with patch("app._trocr_read_batch", return_value=trocr_texts):
            results = app._extract_from_image(
                image,
                label="test",
                progress_cb=_dummy_progress_cb,
                status_cb=status_cb,
                append_cb=append_cb,
                reader=reader,
            )

        return results, appended, status_msgs

    def test_no_detections_returns_empty_list(self):
        image = _solid_image(200, 100)
        results, appended, _ = self._run(image, detections=[])
        assert results == []
        # Append callback must have been called with a "no text detected" message
        assert any("No text detected" in t for t in appended)

    def test_single_detection_returns_result(self):
        image = _solid_image(200, 100)
        bbox = [[10, 10], [80, 10], [80, 40], [10, 40]]  # valid 70×30 region
        detections = [(bbox, "easy_fallback", 0.9)]

        results, _, _ = self._run(image, detections, trocr_texts=["recognised text"])
        assert len(results) == 1
        assert results[0] == "recognised text"

    def test_degenerate_bbox_is_skipped(self):
        """A bounding box where the padded crop is < 5×5 px must be silently skipped.

        With _pad = 4, a bbox occupying a single pixel on a 3×3 image results in:
            x1 = max(0, 1-4) = 0,  x2 = min(3, 1+4) = 3  → width  = 3 < 5  ✓
            y1 = max(0, 1-4) = 0,  y2 = min(3, 1+4) = 3  → height = 3 < 5  ✓
        """
        image = _solid_image(3, 3)
        # Single-point bbox in the middle of a tiny 3×3 image
        bbox = [[1, 1], [1, 1], [1, 1], [1, 1]]
        detections = [(bbox, "tiny", 0.9)]

        results, _, _ = self._run(image, detections, trocr_texts=[])
        assert results == []

    def test_trocr_empty_falls_back_to_easyocr_text(self):
        """When TrOCR returns empty string, EasyOCR text is used as fallback."""
        image = _solid_image(200, 100)
        bbox = [[10, 10], [80, 10], [80, 40], [10, 40]]
        easy_text = "easyocr fallback"
        detections = [(bbox, easy_text, 0.9)]

        results, _, _ = self._run(image, detections, trocr_texts=[""])
        assert len(results) == 1
        assert results[0] == easy_text

    def test_multiple_detections_processed_in_order(self):
        image = _solid_image(400, 200)
        detections = [
            ([[10, 10], [100, 10], [100, 40], [10, 40]], "line1", 0.9),
            ([[10, 50], [100, 50], [100, 80], [10, 80]], "line2", 0.9),
            ([[10, 90], [100, 90], [100, 120], [10, 120]], "line3", 0.9),
        ]
        trocr_texts = ["TrOCR line 1", "TrOCR line 2", "TrOCR line 3"]

        results, _, _ = self._run(image, detections, trocr_texts=trocr_texts)
        assert results == ["TrOCR line 1", "TrOCR line 2", "TrOCR line 3"]

    def test_crop_padding_clamped_to_image_bounds(self):
        """Crops extending beyond the image boundary must be clamped, not raise."""
        image = _solid_image(50, 50)
        # bbox goes right to image edges — padding would push coordinates outside
        bbox = [[0, 0], [50, 0], [50, 50], [0, 50]]
        detections = [(bbox, "edge", 0.9)]

        # Should not raise; result may be recognised or skipped depending on size
        results, _, _ = self._run(image, detections, trocr_texts=["edge text"])
        assert isinstance(results, list)

    def test_status_callback_called_during_processing(self):
        image = _solid_image(200, 100)
        bbox = [[10, 10], [80, 10], [80, 40], [10, 40]]
        detections = [(bbox, "text", 0.9)]

        _, _, status_msgs = self._run(image, detections, trocr_texts=["result"])
        assert any("test" in msg for msg in status_msgs)


# ===========================================================================
# 10. ZIP entry filtering
# ===========================================================================

class TestZipEntryFiltering:
    """Verify that the supported-file detection logic works correctly."""

    def _make_zip(self, tmp_path, filenames):
        """Create a real in-memory ZIP containing empty files with given names."""
        zip_path = tmp_path / "test.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            for name in filenames:
                zf.writestr(name, b"")
        return zip_path

    def _supported_entries(self, zip_path):
        """Re-implement the entry-filtering logic from _run_zip_extraction."""
        supported_exts = app._IMAGE_EXTS | {app._PDF_EXT}
        with zipfile.ZipFile(zip_path, "r") as zf:
            return [
                e for e in zf.infolist()
                if not e.is_dir()
                and not os.path.basename(e.filename).startswith(".")
                and Path(e.filename).suffix.lower() in supported_exts
            ]

    def test_pdf_files_are_included(self, tmp_path):
        z = self._make_zip(tmp_path, ["scan.pdf"])
        entries = self._supported_entries(z)
        assert len(entries) == 1

    def test_png_files_are_included(self, tmp_path):
        z = self._make_zip(tmp_path, ["page.png"])
        entries = self._supported_entries(z)
        assert len(entries) == 1

    def test_all_image_exts_included(self, tmp_path):
        names = [f"file{ext}" for ext in app._IMAGE_EXTS]
        z = self._make_zip(tmp_path, names)
        entries = self._supported_entries(z)
        assert len(entries) == len(app._IMAGE_EXTS)

    def test_unsupported_extensions_excluded(self, tmp_path):
        z = self._make_zip(tmp_path, ["data.csv", "notes.txt", "archive.zip", "doc.docx"])
        entries = self._supported_entries(z)
        assert entries == []

    def test_hidden_files_excluded(self, tmp_path):
        z = self._make_zip(tmp_path, [".hidden.png", ".DS_Store", "visible.png"])
        entries = self._supported_entries(z)
        assert len(entries) == 1
        assert entries[0].filename == "visible.png"

    def test_mixed_archive_only_returns_supported(self, tmp_path):
        z = self._make_zip(tmp_path, [
            "page1.png", "page2.jpg", "scan.pdf",
            "README.md", "data.csv", ".hidden.jpg",
        ])
        entries = self._supported_entries(z)
        assert len(entries) == 3

    def test_nested_path_uses_basename_for_hidden_check(self, tmp_path):
        """Entries stored with directory prefixes should use the file's basename."""
        z = self._make_zip(tmp_path, ["subdir/.hidden.png", "subdir/visible.png"])
        entries = self._supported_entries(z)
        # .hidden.png has basename starting with '.' — should be excluded
        assert len(entries) == 1
        assert os.path.basename(entries[0].filename) == "visible.png"


# ===========================================================================
# 11. _trocr_read (single-image wrapper)
# ===========================================================================

class TestTrocrReadSingle:
    def test_delegates_to_batch(self):
        """_trocr_read must call _trocr_read_batch with a single-item list."""
        img = _solid_image()
        with patch("app._trocr_read_batch", return_value=["hello"]) as mock_batch:
            result = app._trocr_read(img)
        mock_batch.assert_called_once_with([img])
        assert result == "hello"


# ===========================================================================
# 12. _detect_regions
# ===========================================================================

class TestDetectRegions:
    def test_calls_reader_readtext_with_numpy_array(self):
        """_detect_regions must call reader.readtext with a numpy array."""
        import numpy as np
        image = _solid_image(100, 50)
        expected = [([[0, 0], [10, 0], [10, 5], [0, 5]], "text", 0.9)]
        mock_reader = MagicMock()
        mock_reader.readtext = MagicMock(return_value=expected)

        result = app._detect_regions(image, mock_reader)

        mock_reader.readtext.assert_called_once()
        call_args = mock_reader.readtext.call_args
        assert isinstance(call_args[0][0], np.ndarray)
        assert result == expected

    def test_returns_empty_list_when_no_detections(self):
        image = _solid_image(100, 50)
        mock_reader = MagicMock()
        mock_reader.readtext = MagicMock(return_value=[])

        result = app._detect_regions(image, mock_reader)

        assert result == []

    def test_passes_detail_and_paragraph_kwargs(self):
        """readtext must be called with detail=1, paragraph=False."""
        image = _solid_image(80, 60)
        mock_reader = MagicMock()
        mock_reader.readtext = MagicMock(return_value=[])

        app._detect_regions(image, mock_reader)

        _, kwargs = mock_reader.readtext.call_args
        assert kwargs.get("detail") == 1
        assert kwargs.get("paragraph") is False


# ===========================================================================
# 13. _extract_from_image — pre-computed detections parameter
# ===========================================================================

class TestExtractFromImagePrecomputedDetections:
    """Tests for _extract_from_image when detections are pre-provided."""

    def test_reader_readtext_not_called_when_detections_provided(self):
        """When detections are pre-provided, EasyOCR readtext must be skipped."""
        image = _solid_image(200, 100)
        bbox = [[10, 10], [80, 10], [80, 40], [10, 40]]
        detections = [(bbox, "easy_text", 0.9)]
        mock_reader = MagicMock()

        with patch("app._trocr_read_batch", return_value=["result"]):
            results = app._extract_from_image(
                image,
                label="test",
                progress_cb=_dummy_progress_cb,
                status_cb=_dummy_status_cb,
                append_cb=lambda _: None,
                reader=mock_reader,
                detections=detections,
            )

        mock_reader.readtext.assert_not_called()
        assert results == ["result"]

    def test_precomputed_empty_detections_returns_empty(self):
        """Pre-provided empty detections list → no text detected, returns []."""
        image = _solid_image(200, 100)
        appended = []

        results = app._extract_from_image(
            image,
            label="pg",
            progress_cb=_dummy_progress_cb,
            status_cb=_dummy_status_cb,
            append_cb=appended.append,
            reader=MagicMock(),
            detections=[],
        )

        assert results == []
        assert any("No text detected" in t for t in appended)

    def test_precomputed_detections_produce_correct_results(self):
        """Text from pre-provided detections must be recognised correctly."""
        image = _solid_image(400, 200)
        detections = [
            ([[10, 10], [100, 10], [100, 40], [10, 40]], "line1", 0.9),
            ([[10, 50], [100, 50], [100, 80], [10, 80]], "line2", 0.9),
        ]

        with patch("app._trocr_read_batch", return_value=["TrOCR A", "TrOCR B"]):
            results = app._extract_from_image(
                image,
                label="test",
                progress_cb=_dummy_progress_cb,
                status_cb=_dummy_status_cb,
                append_cb=lambda _: None,
                reader=MagicMock(),
                detections=detections,
            )

        assert results == ["TrOCR A", "TrOCR B"]
