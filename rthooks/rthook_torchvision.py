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

import torchvision          # noqa: F401 – must be first
import torchvision.models   # noqa: F401 – pre-populate the lazy attribute
import torchvision.ops      # noqa: F401 – used internally by several models
import torchvision.transforms        # noqa: F401
import torchvision.transforms.functional  # noqa: F401
