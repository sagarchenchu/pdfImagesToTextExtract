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

* CUDA/GPU DLLs are stripped from the bundle by default (see the
  _CUDA_DLL_PREFIXES filter below) to keep the distribution ZIP to a
  manageable size (~500 MB–1 GB instead of ~3 GB).  The app falls back to
  CPU inference automatically.  To re-enable GPU support, remove or comment
  out that filter block and rebuild.
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
# causing WinError 3 (ERROR_PATH_NOT_FOUND).
#
# Collecting .py sources for only the model families actually used at runtime
# (TrOCR, VisionEncoderDecoder, ViT, DeiT, RoBERTa, Auto) provides the
# required on-disk directory tree without bundling hundreds of unused model
# .py files that bloat the distribution ZIP.
# Collecting ALL transformers .py files via
#   collect_data_files("transformers", include_py_files=True)
# previously added ~300–500 MB of unused Python source to the archive.
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
# package — these are small and needed for model configuration at runtime.
datas += collect_data_files("transformers", include_py_files=False)

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
    # feature_extraction_vit is a deprecated shim; image_processing_vit is the modern replacement
    # (listed explicitly further below as it is the primary module needed at runtime)
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
    # ViT image processor — the primary missing module that causes the
    # LazyModule._get_module RuntimeError when importing TrOCRProcessor
    "transformers.models.vit.image_processing_vit",
    "transformers.models.trocr.processing_trocr",  # replaces deprecated feature_extraction_trocr
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
    # timm.layers is the modern API (replaces deprecated timm.models.layers).
    # EasyOCR currently imports via timm.models.layers which is a compat shim
    # in timm 1.x that re-exports from timm.layers, so both must be bundled.
    "timm.layers",
    "timm.models.layers",  # compat shim — needed by EasyOCR until it migrates
    "scipy",
    "scipy.special",
    "scipy.special._ufuncs",
    "sklearn",
    "sklearn.utils",
    # stdlib modules used by app.py at runtime — listed explicitly in case
    # a future PyInstaller version stops auto-collecting them
    "zipfile",
    # html.parser and related stdlib HTML modules — required by requests,
    # huggingface_hub, and transformers for HTML response parsing; PyInstaller
    # occasionally misses these standard-library modules in the frozen bundle.
    "html",
    "html.parser",
    "html.entities",
]

a = Analysis(
    ["app.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=["rthooks/rthook_torchvision.py", "rthooks/rthook_transformers.py"],
    excludes=[
        # Unused UI / data-science packages that sneak in as transitive deps
        "matplotlib",
        "IPython",
        "jupyter",
        "notebook",
        "pandas",
        "scipy.spatial.transform._rotation_groups",
        # Test / benchmark infrastructure (large, never used at runtime)
        "torch.testing",
        "torch.utils.bottleneck",
        "torch.utils.tensorboard",
        "torch.distributed",
        "torch.onnx",
        "torch.jit",           # TorchScript — not used
        "torchvision.datasets",
        "torchvision.io",      # video I/O — not needed
    ],
    noarchive=True,
    optimize=0,
)

pyz = PYZ(a.pure)

# ── Strip CUDA/GPU-specific DLLs from the bundle ──────────────────────────
# PyTorch ships a full CUDA toolkit (~1–2 GB of DLLs: cublas, cudnn, cufft,
# curand, cusolver, cusparse, nvrtc, etc.) even in a plain "pip install torch"
# on Windows.  These binaries are ONLY needed for NVIDIA GPU inference; they
# are unused on systems without a CUDA-capable GPU and represent the single
# largest contributor to distribution ZIP size and extraction time.
#
# Filtering them out here produces a CPU-only EXE that is ~1–2 GB lighter.
# If you need GPU acceleration, remove or comment out this block and rebuild.
#
# DLL name prefixes that identify CUDA / GPU-only libraries:
_CUDA_DLL_PREFIXES = (
    "cublas",        # cuBLAS: GPU BLAS routines
    "cublaslt",      # cuBLAS-Lt: lightweight cuBLAS
    "cudnn",         # cuDNN: deep neural network primitives
    "cufft",         # cuFFT: GPU fast Fourier transform
    "curand",        # cuRAND: GPU random number generation
    "cusolver",      # cuSOLVER: dense/sparse linear algebra
    "cusparse",      # cuSPARSE: sparse matrix routines
    "nccl",          # NCCL: multi-GPU / multi-node communication
    "nvrtc",         # NVRTC: runtime compilation
    "nvtoolsext",    # NVTX: profiling markers
    "c10_cuda",      # PyTorch CUDA C10 library
    "caffe2_nvrtc",  # Caffe2 NVRTC bridge
    "cudart",        # CUDA runtime (not needed without any CUDA kernels)
    "torch_cuda",    # PyTorch CUDA extension entry points
)  # already lowercase — avoids repeated .lower() calls during filtering

def _is_cuda_dll(dest_name: str) -> bool:
    """Return True if the binary is a CUDA/GPU-only library."""
    base = os.path.basename(dest_name).lower()
    return any(base.startswith(p) for p in _CUDA_DLL_PREFIXES)

a.binaries = [(d, s, t) for d, s, t in a.binaries if not _is_cuda_dll(d)]

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
