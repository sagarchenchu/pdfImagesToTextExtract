# PyInstaller runtime hook for transformers
# ==========================================
# transformers >= 4.30 uses a LazyModule in its __init__.py.  When downstream
# code does "from transformers import TrOCRProcessor" the lazy loader calls
# LazyModule._get_module("models.trocr.processing_trocr").  If that submodule
# has not yet been imported, importlib.import_module() runs it from scratch.
# In a frozen EXE this can fail with:
#
#   RuntimeError: Failed to import transformers.models.trocr.processing_trocr …
#     ModuleNotFoundError: No module named 'transformers.models.vit.image_processing_vit'
#
# Pre-importing the required submodules here (before any app code runs) puts
# them into sys.modules so the lazy loader's _get_module() returns them
# immediately on the first access rather than re-running the import machinery.
#
# importlib.import_module is used (not bare "import") so that an individual
# submodule failure is silently swallowed and never aborts the EXE before the
# GUI can open.  Real errors are surfaced by app.py when the model is loaded.

import importlib

for _submod in (
    # ViT image processor – TrOCRProcessor wraps this; missing it is the
    # primary cause of LazyModule._get_module raising RuntimeError/ImportError
    "transformers.models.vit.image_processing_vit",
    # TrOCR processor and its direct dependencies
    "transformers.models.trocr.processing_trocr",
    "transformers.models.trocr.configuration_trocr",
    "transformers.models.trocr.modeling_trocr",
    # VisionEncoderDecoder
    "transformers.models.vision_encoder_decoder.modeling_vision_encoder_decoder",
    "transformers.models.vision_encoder_decoder.configuration_vision_encoder_decoder",
    # Roberta tokenizer (decoder for trocr-large-handwritten)
    "transformers.models.roberta.tokenization_roberta",
    "transformers.models.roberta.tokenization_roberta_fast",
    # Shared transformers utilities referenced by the modules above
    "transformers.processing_utils",
    "transformers.image_processing_utils",
    "transformers.image_utils",
    "transformers.tokenization_utils_base",
):
    try:
        importlib.import_module(_submod)
    except Exception as _e:
        # Swallow to avoid aborting the EXE before the GUI opens; real import
        # failures will surface in app.py when the model is first used.
        import sys as _sys
        if _sys.stderr is not None:
            print(f"[rthook_transformers] WARNING: could not pre-import {_submod}: {_e}", file=_sys.stderr)
