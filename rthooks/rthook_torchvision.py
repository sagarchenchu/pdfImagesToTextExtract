# PyInstaller runtime hook for torchvision
# ==========================================
# torchvision >= 0.16 uses lazy sub-module loading via __init__.py.
# In a frozen (PyInstaller) environment the lazy loader can encounter
# a partially-initialised torchvision module when something tries to do
# "from torchvision import models" before torchvision is fully loaded,
# causing:
#   ImportError: cannot import name 'models' from partially initialized
#   module 'torchvision' (most likely due to a circular import)
#
# Importing the sub-packages explicitly here, before any app code runs,
# ensures the module registry is fully populated and avoids the race.
#
# importlib.import_module is used instead of bare "import" statements so
# that a failure in any individual sub-module does not raise an *unhandled*
# exception that aborts the EXE before the GUI can even open.  All errors
# are silently swallowed here; app.py's _warm_torchvision() will surface
# any real import failure when the model is actually used at runtime.
#
# NOTE: app.py also contains _warm_torchvision() which performs the same
# imports for the non-frozen (python app.py) path.  Keep both lists in sync
# if new torchvision sub-packages need to be pre-imported.

import importlib

try:
    import torchvision  # noqa: F401 – full package __init__ must run first
except Exception:
    pass
else:
    for _submod in (
        "torchvision.models",
        "torchvision.ops",
        "torchvision.transforms",
        "torchvision.transforms.functional",
    ):
        try:
            importlib.import_module(_submod)
        except Exception:
            pass
