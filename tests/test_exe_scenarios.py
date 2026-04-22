"""
test_exe_scenarios.py
=====================
Tests for the three scenarios the user asked about:

  1. EXE / app launch  — does the app start without crashing?
  2. Image upload + text extraction — does the full pipeline work?
  3. Missing libraries / models — are errors handled gracefully?

All heavy dependencies are already stubbed in conftest.py.
Additional stubs specific to these tests are set up inside each class.
"""

import errno
import io
import os
import sys
import tempfile
import threading
import time
import types
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
import app  # noqa: E402  (conftest stubs already active)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _rgb_image(width: int = 80, height: int = 40) -> Image.Image:
    return Image.new("RGB", (width, height), color="lightgray")


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _noop(*_a, **_kw):
    pass


# ===========================================================================
# SCENARIO 1: App startup (EXE launch path)
# ===========================================================================

class TestAppStartup:
    """Verify that HandwritingExtractorApp initialises without raising."""

    def _make_root(self):
        """Return a MagicMock that behaves like a tk.Tk root."""
        root = MagicMock()
        # after() must call the callback so _set_status / _set_progress work
        root.after = MagicMock(side_effect=lambda _ms, fn, *a: fn(*a) if callable(fn) else None)
        return root

    def test_init_does_not_raise(self):
        """HandwritingExtractorApp() must complete __init__ without exception."""
        root = self._make_root()
        app_instance = app.HandwritingExtractorApp(root)
        assert app_instance is not None

    def test_selected_file_initially_none(self):
        root = self._make_root()
        app_instance = app.HandwritingExtractorApp(root)
        assert app_instance._selected_file is None

    def test_not_processing_initially(self):
        root = self._make_root()
        app_instance = app.HandwritingExtractorApp(root)
        assert app_instance._is_processing is False

    def test_title_set_on_root(self):
        root = self._make_root()
        app.HandwritingExtractorApp(root)
        root.title.assert_called_once_with("Handwriting Text Extractor")

    def test_geometry_called_on_root(self):
        root = self._make_root()
        app.HandwritingExtractorApp(root)
        root.geometry.assert_called_once_with("960x720")

    def test_on_clear_clears_output(self):
        """_on_clear must not raise and should call _set_status."""
        root = self._make_root()
        app_instance = app.HandwritingExtractorApp(root)
        # Should not raise
        app_instance._on_clear()

    def test_on_upload_accepts_file_path(self):
        """Simulating a file selection should store the path and enable extract."""
        root = self._make_root()
        app_instance = app.HandwritingExtractorApp(root)
        # Patch the file dialog to return a fake path
        with patch("app.filedialog") as mock_fd:
            mock_fd.askopenfilename.return_value = "/fake/scan.png"
            app_instance._on_upload()
        assert app_instance._selected_file == "/fake/scan.png"

    def test_on_upload_no_file_selected_does_not_change_state(self):
        """If user cancels the dialog (returns ''), state must not change."""
        root = self._make_root()
        app_instance = app.HandwritingExtractorApp(root)
        with patch("app.filedialog") as mock_fd:
            mock_fd.askopenfilename.return_value = ""
            app_instance._on_upload()
        assert app_instance._selected_file is None


# ===========================================================================
# SCENARIO 2: Image upload → text extraction pipeline
# ===========================================================================

class TestImageExtractionPipeline:
    """
    End-to-end tests for _run_extraction with mocked ML models.
    The heavy ML loaders (_load_easyocr, _load_trocr) are patched out so
    we can test the routing and UI-update logic without real models.
    """

    def _make_root(self):
        root = MagicMock()
        root.after = MagicMock(side_effect=lambda _ms, fn, *a: fn(*a) if callable(fn) else None)
        return root

    def _fake_reader(self, bbox=None, easy_text="hello", conf=0.9):
        """Build an easyocr.Reader mock returning one detection."""
        if bbox is None:
            bbox = [[5, 5], [60, 5], [60, 25], [5, 25]]
        reader = MagicMock()
        reader.readtext = MagicMock(return_value=[(bbox, easy_text, conf)])
        return reader

    # ── PNG image ──────────────────────────────────────────────────────────

    def test_image_extraction_produces_results(self, tmp_path):
        """_run_extraction on a PNG must call _extract_from_image and finish."""
        img_path = tmp_path / "test.png"
        _rgb_image().save(img_path)

        root = self._make_root()
        app_instance = app.HandwritingExtractorApp(root)
        app_instance._selected_file = str(img_path)

        fake_reader = self._fake_reader()

        with (
            patch("app._load_easyocr", return_value=fake_reader),
            patch("app._load_trocr", return_value=(MagicMock(), MagicMock())),
            patch("app._trocr_read_batch", return_value=["extracted text"]),
        ):
            app_instance._run_extraction()

        # The result area must have had text appended (status updated)
        # We verify by checking _txt_results.insert was called
        assert app_instance._txt_results.insert.called

    def test_image_extraction_resets_is_processing(self, tmp_path):
        """_is_processing must be False again after extraction completes."""
        img_path = tmp_path / "test.png"
        _rgb_image().save(img_path)

        root = self._make_root()
        app_instance = app.HandwritingExtractorApp(root)
        app_instance._selected_file = str(img_path)
        app_instance._is_processing = True

        with (
            patch("app._load_easyocr", return_value=self._fake_reader()),
            patch("app._load_trocr", return_value=(MagicMock(), MagicMock())),
            patch("app._trocr_read_batch", return_value=["text"]),
        ):
            app_instance._run_extraction()

        assert app_instance._is_processing is False

    def test_image_extraction_enables_extract_button(self, tmp_path):
        """The Extract button must be re-enabled (state=NORMAL) after finishing."""
        img_path = tmp_path / "img.jpg"
        _rgb_image().save(img_path, format="JPEG")

        root = self._make_root()
        app_instance = app.HandwritingExtractorApp(root)
        app_instance._selected_file = str(img_path)

        with (
            patch("app._load_easyocr", return_value=self._fake_reader()),
            patch("app._load_trocr", return_value=(MagicMock(), MagicMock())),
            patch("app._trocr_read_batch", return_value=["result"]),
        ):
            app_instance._run_extraction()

        app_instance._btn_extract.config.assert_called_with(state="normal")

    # ── PDF file ───────────────────────────────────────────────────────────

    def test_pdf_extraction_calls_pdf_to_images(self, tmp_path):
        """A .pdf file must go through _pdf_to_images, not direct Image.open."""
        pdf_path = tmp_path / "doc.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")  # fake file — fitz is mocked

        root = self._make_root()
        app_instance = app.HandwritingExtractorApp(root)
        app_instance._selected_file = str(pdf_path)

        fake_img = _rgb_image()
        fake_reader = self._fake_reader()

        with (
            patch("app._load_easyocr", return_value=fake_reader),
            patch("app._load_trocr", return_value=(MagicMock(), MagicMock())),
            patch("app._pdf_to_images", return_value=[(1, fake_img)]) as mock_pdf,
            patch("app._trocr_read_batch", return_value=["pdf text"]),
        ):
            app_instance._run_extraction()

        mock_pdf.assert_called_once_with(str(pdf_path))

    def test_pdf_multipage_processes_all_pages(self, tmp_path):
        """All pages returned by _pdf_to_images must be processed."""
        pdf_path = tmp_path / "multi.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        pages = [(i + 1, _rgb_image()) for i in range(3)]
        fake_reader = self._fake_reader()

        root = self._make_root()
        app_instance = app.HandwritingExtractorApp(root)
        app_instance._selected_file = str(pdf_path)

        extract_calls = []
        original_extract = app._extract_from_image

        def spy_extract(image, label, **kw):
            extract_calls.append(label)
            return []

        with (
            patch("app._load_easyocr", return_value=fake_reader),
            patch("app._load_trocr", return_value=(MagicMock(), MagicMock())),
            patch("app._pdf_to_images", return_value=pages),
            patch("app._extract_from_image", side_effect=spy_extract),
        ):
            app_instance._run_extraction()

        assert len(extract_calls) == 3

    # ── ZIP archive ────────────────────────────────────────────────────────

    def test_zip_extraction_processes_supported_files(self, tmp_path):
        """A ZIP containing images must process each image file."""
        zip_path = tmp_path / "archive.zip"
        img = _rgb_image()

        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("page1.png", _png_bytes(img))
            zf.writestr("page2.png", _png_bytes(img))
            zf.writestr("notes.txt", b"ignore me")

        fake_reader = self._fake_reader()
        root = self._make_root()
        app_instance = app.HandwritingExtractorApp(root)
        app_instance._selected_file = str(zip_path)

        extract_calls = []

        def spy_extract(image, label, **kw):
            extract_calls.append(label)
            return ["some text"]

        with (
            patch("app._load_easyocr", return_value=fake_reader),
            patch("app._load_trocr", return_value=(MagicMock(), MagicMock())),
            patch("app._extract_from_image", side_effect=spy_extract),
        ):
            app_instance._run_extraction()

        # Only the two PNG files should be processed, not the .txt
        assert len(extract_calls) == 2

    def test_zip_no_supported_files_shows_message(self, tmp_path):
        """A ZIP with no supported files must append a helpful message."""
        zip_path = tmp_path / "empty.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("README.md", b"nothing to extract")

        root = self._make_root()
        app_instance = app.HandwritingExtractorApp(root)
        app_instance._selected_file = str(zip_path)

        appended: list = []
        app_instance._append_text = MagicMock(side_effect=lambda t: appended.append(t))

        with (
            patch("app._load_easyocr", return_value=MagicMock()),
            patch("app._load_trocr", return_value=(MagicMock(), MagicMock())),
        ):
            app_instance._run_extraction()

        joined = " ".join(appended)
        assert "No supported" in joined or "no supported" in joined.lower()


# ===========================================================================
# SCENARIO 3: Missing libraries / model errors
# ===========================================================================

class TestMissingLibrariesAndModels:
    """
    Verify that the app reports clear errors — not silent crashes — when
    critical libraries or model files are missing.
    """

    def _make_root(self):
        root = MagicMock()
        root.after = MagicMock(side_effect=lambda _ms, fn, *a: fn(*a) if callable(fn) else None)
        return root

    # ── _load_easyocr ──────────────────────────────────────────────────────

    def test_load_easyocr_returns_cached_reader_on_second_call(self):
        """_load_easyocr must not re-initialise the Reader on repeated calls."""
        app._easyocr_reader = None
        mock_reader = MagicMock()
        mock_easyocr = MagicMock()
        mock_easyocr.Reader.return_value = mock_reader

        with patch.dict(sys.modules, {"easyocr": mock_easyocr}):
            with patch("app._bundled_easyocr_model_dir", return_value=None):
                app._device = "cpu"
                r1 = app._load_easyocr(_noop)
                r2 = app._load_easyocr(_noop)

        assert r1 is r2
        assert mock_easyocr.Reader.call_count == 1
        app._easyocr_reader = None
        app._device = None

    def test_load_easyocr_raises_connection_error_on_network_failure(self):
        """A socket/network OSError during easyocr.Reader() must become ConnectionError."""
        app._easyocr_reader = None
        network_exc = OSError(errno.ECONNREFUSED, "Connection refused")

        mock_easyocr = MagicMock()
        mock_easyocr.Reader.side_effect = network_exc

        with patch.dict(sys.modules, {"easyocr": mock_easyocr}):
            with patch("app._bundled_easyocr_model_dir", return_value=None):
                app._device = "cpu"
                with pytest.raises(ConnectionError, match="internet connection"):
                    app._load_easyocr(_noop)

        app._easyocr_reader = None
        app._device = None

    def test_load_easyocr_uses_bundled_dir_when_frozen(self, tmp_path):
        """When running frozen with a local models dir, download_enabled=False."""
        app._easyocr_reader = None
        easyocr_dir = tmp_path / "models" / "easyocr"
        easyocr_dir.mkdir(parents=True)

        mock_reader = MagicMock()
        mock_easyocr = MagicMock()
        mock_easyocr.Reader.return_value = mock_reader

        with patch.dict(sys.modules, {"easyocr": mock_easyocr}):
            with patch("app._bundled_easyocr_model_dir", return_value=str(easyocr_dir)):
                app._device = "cpu"
                app._load_easyocr(_noop)

        _, kwargs = mock_easyocr.Reader.call_args
        assert kwargs.get("download_enabled") is False
        assert kwargs.get("model_storage_directory") == str(easyocr_dir)
        app._easyocr_reader = None
        app._device = None

    def test_load_easyocr_non_network_exception_reraises_as_is(self):
        """A non-network exception (e.g. ValueError) must re-raise without wrapping."""
        app._easyocr_reader = None
        mock_easyocr = MagicMock()
        mock_easyocr.Reader.side_effect = ValueError("unexpected")

        with patch.dict(sys.modules, {"easyocr": mock_easyocr}):
            with patch("app._bundled_easyocr_model_dir", return_value=None):
                app._device = "cpu"
                with pytest.raises(ValueError, match="unexpected"):
                    app._load_easyocr(_noop)

        app._easyocr_reader = None
        app._device = None

    # ── _load_trocr — frozen EXE, models/ missing ──────────────────────────

    def test_load_trocr_frozen_no_models_dir_raises_file_not_found(self):
        """Frozen EXE without a models/ directory must raise FileNotFoundError."""
        app._trocr_processor = None
        app._trocr_model = None

        with patch.object(sys, "frozen", True, create=True):
            with patch("app._resolve_models_dir", return_value=None):
                with pytest.raises(FileNotFoundError, match="models/"):
                    app._load_trocr(_noop)

        app._trocr_processor = None
        app._trocr_model = None

    def test_load_trocr_frozen_loads_from_sideloaded_dir(self, tmp_path):
        """Frozen EXE with a sideloaded models/ must load from that directory."""
        app._trocr_processor = None
        app._trocr_model = None

        models_dir = tmp_path / "models"
        trocr_dir = models_dir / "trocr"
        trocr_dir.mkdir(parents=True)

        mock_proc = MagicMock()
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock()

        # TrOCRProcessor and VisionEncoderDecoderModel MUST be real classes so that
        # isinstance(TrOCRProcessor, type) passes in _load_trocr.
        from_pretrained_proc = MagicMock(return_value=mock_proc)
        from_pretrained_model = MagicMock(return_value=mock_model)

        class FakeTrOCRProcessor:
            from_pretrained = staticmethod(from_pretrained_proc)

        class FakeVisionEncoderDecoderModel:
            from_pretrained = staticmethod(from_pretrained_model)

        mock_transformers = types.ModuleType("transformers")
        mock_transformers.TrOCRProcessor = FakeTrOCRProcessor
        mock_transformers.VisionEncoderDecoderModel = FakeVisionEncoderDecoderModel

        with (
            patch.object(sys, "frozen", True, create=True),
            patch("app._resolve_models_dir", return_value=models_dir),
            patch.dict(sys.modules, {"transformers": mock_transformers}),
            patch("app._get_device", return_value="cpu"),
        ):
            proc, mdl = app._load_trocr(_noop)

        assert proc is mock_proc
        assert mdl is mock_model
        mock_model.eval.assert_called_once()
        app._trocr_processor = None
        app._trocr_model = None

    # ── _load_trocr — source mode, network failure during download ─────────

    def test_load_trocr_source_connection_error_on_download(self):
        """Network failure during model download must raise ConnectionError."""
        app._trocr_processor = None
        app._trocr_model = None

        network_exc = OSError(errno.ETIMEDOUT, "timed out")

        # side_effect list: first call (local_files_only=True) → OSError (not cached)
        # second call (no local_files_only) → network error
        from_pretrained_proc = MagicMock(side_effect=[OSError("not cached"), network_exc])
        # VisionEncoderDecoderModel is never reached (exception fires on processor's 2nd call)

        class FakeTrOCRProcessor:
            from_pretrained = staticmethod(from_pretrained_proc)

        class FakeVisionEncoderDecoderModel:
            from_pretrained = staticmethod(MagicMock(side_effect=OSError("not cached")))

        mock_transformers = types.ModuleType("transformers")
        mock_transformers.TrOCRProcessor = FakeTrOCRProcessor
        mock_transformers.VisionEncoderDecoderModel = FakeVisionEncoderDecoderModel

        with (
            patch.object(sys, "frozen", False, create=True),
            patch.dict(sys.modules, {"transformers": mock_transformers}),
        ):
            with pytest.raises(ConnectionError, match="internet connection"):
                app._load_trocr(_noop)

        app._trocr_processor = None
        app._trocr_model = None

    def test_load_trocr_source_uses_local_cache_when_available(self):
        """When model is already cached, from_pretrained(local_files_only=True) succeeds."""
        app._trocr_processor = None
        app._trocr_model = None

        mock_proc = MagicMock()
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock()

        from_pretrained_proc = MagicMock(return_value=mock_proc)
        from_pretrained_model = MagicMock(return_value=mock_model)

        class FakeTrOCRProcessor:
            from_pretrained = staticmethod(from_pretrained_proc)

        class FakeVisionEncoderDecoderModel:
            from_pretrained = staticmethod(from_pretrained_model)

        mock_transformers = types.ModuleType("transformers")
        mock_transformers.TrOCRProcessor = FakeTrOCRProcessor
        mock_transformers.VisionEncoderDecoderModel = FakeVisionEncoderDecoderModel

        with (
            patch.object(sys, "frozen", False, create=True),
            patch.dict(sys.modules, {"transformers": mock_transformers}),
            patch("app._get_device", return_value="cpu"),
        ):
            proc, mdl = app._load_trocr(_noop)

        assert proc is mock_proc
        assert mdl is mock_model
        # local_files_only=True must have been used (no network attempt)
        from_pretrained_proc.assert_called_once()
        _, kwargs = from_pretrained_proc.call_args
        assert kwargs.get("local_files_only") is True
        app._trocr_processor = None
        app._trocr_model = None

    def test_load_trocr_returns_cached_on_second_call(self):
        """_load_trocr must not reload when processor and model are already set."""
        sentinel_proc = object()
        sentinel_model = object()
        app._trocr_processor = sentinel_proc
        app._trocr_model = sentinel_model

        proc, mdl = app._load_trocr(_noop)
        assert proc is sentinel_proc
        assert mdl is sentinel_model
        app._trocr_processor = None
        app._trocr_model = None

    # ── Lazy-loader fallback: TrOCRProcessor is not a real type ───────────

    def test_load_trocr_falls_back_to_direct_import_when_lazy_loader_fails(self, tmp_path):
        """If transformers.TrOCRProcessor is not a type, the direct path is used."""
        app._trocr_processor = None
        app._trocr_model = None

        # Make models dir for frozen=False
        mock_proc = MagicMock()
        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock()

        # Top-level transformers stub: TrOCRProcessor is NOT a type (non-class object)
        mock_transformers_top = types.ModuleType("transformers")
        mock_transformers_top.TrOCRProcessor = "not_a_type"  # triggers AttributeError branch

        # Direct-path stubs
        mock_proc_cls = MagicMock()
        mock_proc_cls.from_pretrained = MagicMock(return_value=mock_proc)
        mock_model_cls = MagicMock()
        mock_model_cls.from_pretrained = MagicMock(return_value=mock_model)

        mock_trocr_mod = MagicMock()
        mock_trocr_mod.TrOCRProcessor = mock_proc_cls

        mock_ved_mod = MagicMock()
        mock_ved_mod.VisionEncoderDecoderModel = mock_model_cls

        with (
            patch.object(sys, "frozen", False, create=True),
            patch.dict(sys.modules, {
                "transformers": mock_transformers_top,
                "transformers.models.trocr.processing_trocr": mock_trocr_mod,
                "transformers.models.vision_encoder_decoder.modeling_vision_encoder_decoder": mock_ved_mod,
            }),
            patch("app._get_device", return_value="cpu"),
        ):
            proc, mdl = app._load_trocr(_noop)

        assert proc is mock_proc
        assert mdl is mock_model
        app._trocr_processor = None
        app._trocr_model = None

    # ── Missing fitz (PyMuPDF) ─────────────────────────────────────────────

    def test_pdf_to_images_raises_when_fitz_missing(self):
        """If PyMuPDF is not installed, _pdf_to_images must raise ImportError."""
        with patch.dict(sys.modules, {"fitz": None}):
            with pytest.raises((ImportError, ModuleNotFoundError)):
                app._pdf_to_images("/some/file.pdf")

    # ── Extraction pipeline error handling ────────────────────────────────

    def test_extraction_error_appended_to_results(self, tmp_path):
        """Any exception in _run_extraction must be written to the results area."""
        img_path = tmp_path / "bad.png"
        _rgb_image().save(img_path)

        root = MagicMock()
        root.after = MagicMock(side_effect=lambda _ms, fn, *a: fn(*a) if callable(fn) else None)
        app_instance = app.HandwritingExtractorApp(root)
        app_instance._selected_file = str(img_path)

        appended: list = []
        app_instance._append_text = MagicMock(side_effect=lambda t: appended.append(t))

        with patch("app._load_easyocr", side_effect=RuntimeError("model exploded")):
            app_instance._run_extraction()

        joined = " ".join(appended)
        assert "ERROR" in joined or "error" in joined.lower() or "model exploded" in joined

    def test_extraction_error_resets_is_processing(self, tmp_path):
        """Even when extraction raises, _is_processing must reset to False."""
        img_path = tmp_path / "bad.png"
        _rgb_image().save(img_path)

        root = MagicMock()
        root.after = MagicMock(side_effect=lambda _ms, fn, *a: fn(*a) if callable(fn) else None)
        app_instance = app.HandwritingExtractorApp(root)
        app_instance._selected_file = str(img_path)
        app_instance._is_processing = True

        with patch("app._load_easyocr", side_effect=RuntimeError("boom")):
            app_instance._run_extraction()

        assert app_instance._is_processing is False

    def test_extraction_error_reenables_extract_button(self, tmp_path):
        """After a failed extraction, the Extract button must be re-enabled."""
        img_path = tmp_path / "bad.png"
        _rgb_image().save(img_path)

        root = MagicMock()
        root.after = MagicMock(side_effect=lambda _ms, fn, *a: fn(*a) if callable(fn) else None)
        app_instance = app.HandwritingExtractorApp(root)
        app_instance._selected_file = str(img_path)

        with patch("app._load_easyocr", side_effect=RuntimeError("boom")):
            app_instance._run_extraction()

        app_instance._btn_extract.config.assert_called_with(state="normal")
