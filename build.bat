@echo off
REM ============================================================
REM  build.bat  –  Build HandwritingExtractor.exe on Windows
REM ============================================================
REM Requirements:
REM   * Python 3.10 or 3.11 (64-bit) installed and on PATH
REM
REM ML models are NOT downloaded or bundled during the build.
REM After building, place a  models\  folder next to the EXE:
REM
REM   dist\HandwritingExtractor\
REM       HandwritingExtractor.exe
REM       models\
REM           trocr\          <- microsoft/trocr-large-handwritten files
REM           easyocr\        <- craft_mlt_25k.pth + english_g2.pth
REM
REM See README.md for model download links.
REM ============================================================

echo.
echo ============================================================
echo  Handwriting Text Extractor – Windows EXE Builder
echo ============================================================
echo.

REM ── 1. Create / activate a virtual environment ────────────────
if not exist ".venv" (
    echo [1/4] Creating virtual environment…
    python -m venv .venv
) else (
    echo [1/4] Virtual environment already exists.
)

call .venv\Scripts\activate.bat

REM ── 2. Install / upgrade dependencies ────────────────────────
echo.
echo [2/4] Installing dependencies (this may take several minutes)…
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
pip install pyinstaller --quiet

REM ── 3. Run PyInstaller ────────────────────────────────────────
echo.
echo [3/4] Building EXE with PyInstaller…
pyinstaller handwriting_extractor.spec --noconfirm

REM ── 4. Done ───────────────────────────────────────────────────
echo.
if exist "dist\HandwritingExtractor\HandwritingExtractor.exe" (
    echo [4/4] SUCCESS!
    echo.
    echo   EXE location:
    echo     dist\HandwritingExtractor\HandwritingExtractor.exe
    echo.
    echo   NEXT STEP: place your models\ folder next to the EXE:
    echo     dist\HandwritingExtractor\models\trocr\       (TrOCR files)
    echo     dist\HandwritingExtractor\models\easyocr\     (EasyOCR .pth files)
    echo.
    echo   Then zip and distribute the dist\HandwritingExtractor\ folder.
) else (
    echo [4/4] Build may have failed – check output above for errors.
)

echo.
pause
