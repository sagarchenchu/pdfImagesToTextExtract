# PyInstaller runtime hook — redirect NULL stdout/stderr
# ======================================================
# When PyInstaller builds with console=False (windowed GUI) the bootloader sets
# sys.stdout and sys.stderr to None because no console is attached.  Several
# third-party libraries (pip, rich, tqdm, huggingface_hub, …) call methods such
# as sys.stdout.isatty() or sys.stdout.write() without first checking for None,
# which raises:
#
#   AttributeError: 'NoneType' object has no attribute 'isatty'
#
# Replace None streams with a no-op sink backed by os.devnull BEFORE any other
# runtime hooks or application code runs, so all library code works without
# modification.  The sink is opened in text mode with UTF-8 encoding to match
# what the rest of the application expects.

import os
import sys

if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8", errors="replace")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8", errors="replace")
