# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for TextExtractAPI.exe
========================================
Build command (from repo root, on Windows with Python 3.10/3.11):

    pip install pyinstaller
    pyinstaller api_server.spec

The resulting exe is written to  dist\TextExtractAPI\TextExtractAPI.exe

Notes
-----
* ML models (TrOCR ~1 GB, EasyOCR ~250 MB) are NOT bundled into the EXE.
  Instead, place a  models\  folder next to TextExtractAPI.exe after
  extracting the distribution:

      TextExtractAPI\
          TextExtractAPI.exe
          models\
              trocr\          <- microsoft/trocr-large-handwritten files
              easyocr\        <- craft_mlt_25k.pth + english_g2.pth

  See README.md for download links.  At runtime the EXE checks for the
  sideloaded  models\  folder first and falls back to any bundled copy
  inside _MEIPASS (kept for backwards compatibility).

* The --onedir layout is used here because PyTorch DLLs expand to ~3 GB
  which makes --onefile extremely slow on startup.  Zip the
  dist\TextExtractAPI folder for distribution.

* CUDA/GPU DLLs are stripped from the bundle by default to keep the
  distribution ZIP manageable (~500 MB–1 GB instead of ~3 GB).  The app
  falls back to CPU inference automatically.  To re-enable GPU support,
  remove or comment out the _CUDA_DLL_PREFIXES filter block and rebuild.
"""

import os
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
    "scipy",
    "sklearn",          # scikit-learn
    "yaml",             # PyYAML
    "shapely",
    "pyclipper",
    "bidi",
    "fsspec",
    "six",
    "flask",            # Flask web framework
    "werkzeug",         # Flask's WSGI toolkit (required by Flask)
    "click",            # Flask's CLI dependency
    "jinja2",           # Flask's template engine
    "markupsafe",       # Required by Jinja2
    "itsdangerous",     # Flask session signing
]:
    d, b, h = collect_all(pkg)
    datas    += d
    binaries += b
    hiddenimports += h

# ── Ensure transformers/models/ directory exists on disk at runtime ────────
for _tf_pkg in [
    "transformers.models.trocr",
    "transformers.models.vision_encoder_decoder",
    "transformers.models.vit",
    "transformers.models.deit",
    "transformers.models.roberta",
    "transformers.models.auto",
]:
    datas += collect_data_files(_tf_pkg, include_py_files=True)

# Non-.py data files (JSON configs, vocab files, etc.) for the full transformers
datas += collect_data_files("transformers", include_py_files=False)

# ── ML models are NOT bundled — they are sideloaded at runtime ────────────
_spec_root = Path(SPECPATH)  # noqa: F821 – PyInstaller built-in

# Extra hidden imports
hiddenimports += [
    # Flask and WSGI
    "flask",
    "flask.cli",
    "flask.globals",
    "flask.helpers",
    "flask.json",
    "flask.json.provider",
    "flask.logging",
    "flask.sansio.scaffold",
    "flask.sansio.app",
    "flask.sansio.blueprints",
    "werkzeug",
    "werkzeug.serving",
    "werkzeug.routing",
    "werkzeug.routing.rules",
    "werkzeug.routing.map",
    "werkzeug.routing.matcher",
    "werkzeug.routing.converters",
    "werkzeug.routing.exceptions",
    "werkzeug.exceptions",
    "werkzeug.datastructures",
    "werkzeug.datastructures.file_storage",
    "werkzeug.datastructures.headers",
    "werkzeug.datastructures.structures",
    "werkzeug.middleware",
    "werkzeug.middleware.proxy_fix",
    "werkzeug.wrappers",
    "werkzeug.wrappers.request",
    "werkzeug.wrappers.response",
    "werkzeug.formparser",
    "werkzeug.http",
    "werkzeug.urls",
    "werkzeug.utils",
    "werkzeug.wsgi",
    "werkzeug.sansio",
    "werkzeug.sansio.utils",
    "werkzeug.sansio.http",
    "jinja2",
    "jinja2.ext",
    "jinja2.loaders",
    "markupsafe",
    "click",
    "itsdangerous",
    # ML / OCR
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
    "transformers.models.vit.image_processing_vit",
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
    "transformers.image_processing_base",
    "transformers.image_transforms",
    "transformers.image_utils",
    "transformers.tokenization_utils_base",
    "easyocr.detection",
    "easyocr.recognition",
    "easyocr.utils",
    "easyocr.config",
    "easyocr.craft_utils",
    "easyocr.imgproc",
    "easyocr.model.modules",
    "easyocr.model.vgg_model",
    "easyocr.DBNet_utils",
    "timm.layers",
    "timm.models.layers",
    "scipy",
    "scipy.special",
    "scipy.special._ufuncs",
    "scipy.spatial",
    "scipy.spatial.transform",
    "scipy.spatial.transform._rotation_groups",
    "scipy.sparse",
    "scipy.sparse.csgraph",
    "scipy.sparse._compressed",
    "scipy.sparse._csr",
    "scipy.sparse._csc",
    "scipy.sparse._coo",
    "scipy.linalg",
    "scipy.linalg.blas",
    "scipy.linalg.lapack",
    "scipy.optimize",
    "scipy.ndimage",
    "scipy.stats",
    "scipy.io",
    "scipy.interpolate",
    "scipy.signal",
    "scipy.fft",
    "scipy.integrate",
    "sklearn",
    "sklearn.utils",
    "sklearn.utils._cython_blas",
    "sklearn.utils._sorting",
    "sklearn.utils._heap",
    "sklearn.utils.murmurhash",
    "sklearn.neighbors",
    "sklearn.neighbors._partition_nodes",
    "sklearn.cluster",
    "sklearn.decomposition",
    "sklearn.preprocessing",
    "sklearn.metrics",
    "sklearn.linear_model",
    "sklearn.svm",
    "sklearn.tree",
    "sklearn.ensemble",
    "PIL.Image",
    "PIL.ImageOps",
    "PIL.ImageFilter",
    "PIL.ImageDraw",
    "PIL.ImageFont",
    "PIL.ImageEnhance",
    "PIL.ImageChops",
    "PIL.ImageColor",
    "PIL.ImageStat",
    "PIL.ImageSequence",
    "PIL.JpegImagePlugin",
    "PIL.PngImagePlugin",
    "PIL.BmpImagePlugin",
    "PIL.TiffImagePlugin",
    "yaml",
    "multiprocessing.pool",
    "multiprocessing.managers",
    "zipfile",
    "html",
    "html.parser",
    "html.entities",
]

a = Analysis(
    ["api_server.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[
        "rthooks/rthook_stdio.py",
        "rthooks/rthook_torchvision.py",
        "rthooks/rthook_transformers.py",
    ],
    excludes=[
        "matplotlib",
        "IPython",
        "jupyter",
        "notebook",
        "pandas",
        "torch.testing",
        "torch.utils.bottleneck",
        "torch.utils.tensorboard",
        "torch.distributed",
        "torch.onnx",
        "torch.jit",
        "torchvision.datasets",
        "torchvision.io",
        # GUI — not needed in the API server
        "tkinter",
        "tkinter.ttk",
        "tkinter.filedialog",
        "tkinter.messagebox",
        "tkinter.scrolledtext",
        "_tkinter",
    ],
    noarchive=True,
    optimize=0,
)

pyz = PYZ(a.pure)

# ── Strip CUDA/GPU-specific DLLs from the bundle ──────────────────────────
_CUDA_DLL_PREFIXES = (
    "cublas",
    "cublaslt",
    "cudnn",
    "cufft",
    "curand",
    "cusolver",
    "cusparse",
    "nccl",
    "nvrtc",
    "nvtoolsext",
    "c10_cuda",
    "caffe2_nvrtc",
    "cudart",
    "torch_cuda",
)


def _is_cuda_dll(dest_name: str) -> bool:
    base = os.path.basename(dest_name).lower()
    return any(base.startswith(p) for p in _CUDA_DLL_PREFIXES)


a.binaries = [(d, s, t) for d, s, t in a.binaries if not _is_cuda_dll(d)]

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="TextExtractAPI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,           # console=True — server logs go to stdout
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="TextExtractAPI",
)
