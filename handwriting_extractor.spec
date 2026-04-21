# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for HandwritingExtractor.exe
=============================================
Build command (from repo root, on Windows with Python 3.10/3.11):

    python download_models.py          # one-time: pre-download ML models
    pip install pyinstaller
    pyinstaller handwriting_extractor.spec

The resulting exe is written to  dist\HandwritingExtractor\HandwritingExtractor.exe
(or dist\HandwritingExtractor.exe when onefile=True – see note below).

Notes
-----
* ML models (TrOCR ~1 GB, EasyOCR ~250 MB) are bundled inside the distribution
  folder so the application works fully offline with no internet connection needed.
  Run  download_models.py  before this spec to populate the  models/  directory.
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

# ── Bundle pre-downloaded ML models ───────────────────────────────────────
# Run  download_models.py  before this spec to populate these directories.
_spec_root = Path(SPECPATH)  # noqa: F821 – PyInstaller built-in

_trocr_dir = _spec_root / "models" / "trocr"
if _trocr_dir.exists():
    datas += [(str(_trocr_dir), "models/trocr")]
    print(f"INFO: Bundling TrOCR model from {_trocr_dir}")
else:
    print("WARNING: models/trocr not found – run  python download_models.py  first.")
    print("         The EXE will require internet access on first launch to download models.")

_easyocr_dir = _spec_root / "models" / "easyocr"
if _easyocr_dir.exists():
    datas += [(str(_easyocr_dir), "models/easyocr")]
    print(f"INFO: Bundling EasyOCR models from {_easyocr_dir}")
else:
    print("WARNING: models/easyocr not found – run  python download_models.py  first.")
    print("         The EXE will require internet access on first launch to download models.")

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
    "torch.nn.modules.activation",
    "torch.nn.modules.container",
    "torch.nn.modules.conv",
    "torch.nn.modules.linear",
    "torch.nn.modules.normalization",
    "transformers.models.trocr",
    "transformers.models.trocr.modeling_trocr",
    "transformers.models.vision_encoder_decoder",
    "transformers.models.vision_encoder_decoder.modeling_vision_encoder_decoder",
    "transformers.models.deit",
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
    runtime_hooks=[],
    excludes=[
        # Cut down size by removing things we don't need
        "matplotlib",
        "IPython",
        "jupyter",
        "notebook",
        "pandas",
        "scipy.spatial.transform._rotation_groups",
    ],
    noarchive=False,
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
