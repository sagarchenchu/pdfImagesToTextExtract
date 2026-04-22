"""
conftest.py
===========
Install lightweight stubs for heavy/GUI dependencies (tkinter, torch,
easyocr, transformers) into sys.modules **before** app.py is imported by any
test.  This lets us run the full pure-logic test suite without a display,
without CUDA, and without gigabytes of ML models.
"""

import sys
import types
from unittest.mock import MagicMock


def _make_module(name: str, **attrs) -> types.ModuleType:
    """Create a minimal stub module with the given attributes."""
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


# ── tkinter stubs ────────────────────────────────────────────────────────────
# tkinter is not available in the headless CI/test environment.
_tk_root_mock = MagicMock()
_tk_stub = _make_module(
    "tkinter",
    Tk=MagicMock(return_value=_tk_root_mock),
    Frame=MagicMock,
    Label=MagicMock,
    Button=MagicMock,
    Text=MagicMock,
    LabelFrame=MagicMock,
    X="x",
    Y="y",
    BOTH="both",
    LEFT="left",
    RIGHT="right",
    END="end",
    NORMAL="normal",
    DISABLED="disabled",
    WORD="word",
    FLAT="flat",
    GROOVE="groove",
    N="n",
    S="s",
    W="w",
    E="e",
)
sys.modules.setdefault("tkinter", _tk_stub)

for _sub in ("ttk", "filedialog", "messagebox", "scrolledtext"):
    sys.modules.setdefault(f"tkinter.{_sub}", MagicMock())

# ── torch stub ───────────────────────────────────────────────────────────────
_torch_stub = _make_module(
    "torch",
    cuda=_make_module("torch.cuda", is_available=MagicMock(return_value=False)),
    no_grad=MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())),
    Tensor=MagicMock,
)
sys.modules.setdefault("torch", _torch_stub)
sys.modules.setdefault("torch.cuda", _torch_stub.cuda)

# ── transformers stub ────────────────────────────────────────────────────────
_transformers_stub = _make_module(
    "transformers",
    TrOCRProcessor=MagicMock,
    VisionEncoderDecoderModel=MagicMock,
)
sys.modules.setdefault("transformers", _transformers_stub)
for _sub in (
    "transformers.models",
    "transformers.models.trocr",
    "transformers.models.trocr.processing_trocr",
    "transformers.models.vision_encoder_decoder",
    "transformers.models.vision_encoder_decoder.modeling_vision_encoder_decoder",
):
    sys.modules.setdefault(_sub, MagicMock())

# ── easyocr stub ─────────────────────────────────────────────────────────────
sys.modules.setdefault("easyocr", MagicMock())

# ── torchvision stub ─────────────────────────────────────────────────────────
for _tv in (
    "torchvision",
    "torchvision.models",
    "torchvision.ops",
    "torchvision.transforms",
    "torchvision.transforms.functional",
):
    sys.modules.setdefault(_tv, MagicMock())
