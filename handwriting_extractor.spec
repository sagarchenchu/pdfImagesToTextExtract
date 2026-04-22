# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for HandwritingExtractor.exe
=============================================
Build command (from repo root, on Windows with Python 3.10/3.11):

    pip install pyinstaller
    pyinstaller handwriting_extractor.spec

The resulting exe is written to  dist\HandwritingExtractor\HandwritingExtractor.exe
(or dist\HandwritingExtractor.exe when onefile=True – see note below).

Notes
-----
* ML models (TrOCR ~1 GB, EasyOCR ~250 MB) are NOT bundled into the EXE.
  Instead, place a  models\  folder next to HandwritingExtractor.exe after
  extracting the distribution:

      HandwritingExtractor\
          HandwritingExtractor.exe
          models\
              trocr\          <- microsoft/trocr-large-handwritten files
              easyocr\        <- craft_mlt_25k.pth + english_g2.pth

  See README.md for download links.  At runtime the EXE checks for the
  sideloaded  models\  folder first and falls back to any bundled copy
  inside _MEIPASS (kept for backwards compatibility).

* The --onedir layout is used here because PyTorch DLLs expand to ~3 GB
  which makes --onefile extremely slow on startup (it must unpack to %TEMP%
  every launch).  Zip the dist\HandwritingExtractor folder for distribution.
"""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_data_files

# ── Collect all files for heavy packages ──────────────────────────────────
datas = []
binaries = []
hiddenimports = []

for pkg in [
    "torch",
    "torchvision",
    "transformers",
    "tokenizers",
    "easyocr",
    "timm",
    "PIL",
    "cv2",
    "numpy",
    "fitz",        # PyMuPDF
    "huggingface_hub",
    "safetensors",
    "sentencepiece",
    "tqdm",
    "filelock",
    "packaging",
    "regex",
    "requests",
    "certifi",
    "charset_normalizer",
    "urllib3",
    "idna",
]:
    d, b, h = collect_all(pkg)
    datas    += d
    binaries += b
    hiddenimports += h

# ── Ensure transformers/models/ directory exists on disk at runtime ────────
# transformers uses __file__-relative os.path lookups (e.g.
#   os.path.join(os.path.dirname(__file__), "models"))
# at runtime.  PyInstaller normally compiles .py files into the .pyz archive
# which makes the physical _internal/transformers/models/ directory absent,
# causing WinError 3 (ERROR_PATH_NOT_FOUND).  Collecting the transformers
# sources as data files (include_py_files=True) copies them next to the EXE
# so the directory tree exists on disk.
datas += collect_data_files("transformers", include_py_files=True)

# ── ML models are NOT bundled — they are sideloaded at runtime ────────────
# Place a  models\  folder next to HandwritingExtractor.exe after extracting.
# See README.md for download links and the expected folder layout.
_spec_root = Path(SPECPATH)  # noqa: F821 – PyInstaller built-in

# Extra hidden imports that static analysis sometimes misses
hiddenimports += [
    "tkinter",
    "tkinter.ttk",
    "tkinter.filedialog",
    "tkinter.messagebox",
    "tkinter.scrolledtext",
    "PIL._tkinter_finder",
    "PIL.Image",
    "PIL.ImageOps",
    "torchvision.transforms",
    "torchvision.transforms.functional",
    "torchvision.models",
    "torchvision.models.resnet",
    "torchvision.models.vgg",
    "torchvision.models.densenet",
    "torchvision.models.inception",
    "torchvision.models.googlenet",
    "torchvision.models.alexnet",
    "torchvision.models.squeezenet",
    "torchvision.models.mobilenet",
    "torchvision.models.mnasnet",
    "torchvision.models.shufflenetv2",
    "torchvision.ops",
    "torchvision.ops.feature_pyramid_network",
    "torch.nn.modules.activation",
    "torch.nn.modules.container",
    "torch.nn.modules.conv",
    "torch.nn.modules.linear",
    "torch.nn.modules.normalization",
    "transformers.models.trocr",
    "transformers.models.trocr.modeling_trocr",
    "transformers.models.trocr.configuration_trocr",
    "transformers.models.trocr.processing_trocr",
    "transformers.models.vision_encoder_decoder",
    "transformers.models.vision_encoder_decoder.modeling_vision_encoder_decoder",
    "transformers.models.vision_encoder_decoder.configuration_vision_encoder_decoder",
    "transformers.models.deit",
    "transformers.models.deit.modeling_deit",
    "transformers.models.deit.configuration_deit",
    "transformers.models.vit",
    "transformers.models.vit.modeling_vit",
    "transformers.models.vit.configuration_vit",
    "transformers.models.vit.feature_extraction_vit",
    "transformers.models.roberta",
    "transformers.models.roberta.modeling_roberta",
    "transformers.models.roberta.configuration_roberta",
    "transformers.models.roberta.tokenization_roberta",
    "transformers.models.roberta.tokenization_roberta_fast",
    "transformers.models.auto",
    "transformers.models.auto.configuration_auto",
    "transformers.models.auto.modeling_auto",
    "transformers.models.auto.tokenization_auto",
    "transformers.models.auto.processing_auto",
    "transformers.modeling_utils",
    "transformers.tokenization_utils",
    "transformers.tokenization_utils_fast",
    "transformers.processing_utils",
    "transformers.feature_extraction_utils",
    "transformers.image_processing_utils",
    "transformers.image_transforms",
    "easyocr.detection",
    "easyocr.recognition",
    "easyocr.utils",
    "easyocr.config",
    "scipy",
    "scipy.special",
    "scipy.special._ufuncs",
    "sklearn",
    "sklearn.utils",
]

a = Analysis(
    ["app.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=["rthooks/rthook_torchvision.py"],
    excludes=[
        # Cut down size by removing things we don't need
        "matplotlib",
        "IPython",
        "jupyter",
        "notebook",
        "pandas",
        "scipy.spatial.transform._rotation_groups",
    ],
    noarchive=True,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,   # onedir – binaries go to COLLECT
    name="HandwritingExtractor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,           # no black console window behind the GUI
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,               # replace with an .ico path if you have one
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="HandwritingExtractor",
)
