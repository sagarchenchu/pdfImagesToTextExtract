@echo off
REM ============================================================
REM  build.bat  –  Build HandwritingExtractor.exe on Windows
REM ============================================================
REM Requirements:
REM   * Python 3.10 or 3.11 (64-bit) installed and on PATH
REM   * Internet access for first-time pip and model downloads
REM
REM Usage:
REM   Double-click build.bat   OR   run from a Command Prompt
REM ============================================================

echo.
echo ============================================================
echo  Handwriting Text Extractor – Windows EXE Builder
echo ============================================================
echo.

REM ── 1. Create / activate a virtual environment ────────────────
if not exist ".venv" (
    echo [1/5] Creating virtual environment…
    python -m venv .venv
) else (
    echo [1/5] Virtual environment already exists.
)

call .venv\Scripts\activate.bat

REM ── 2. Install / upgrade dependencies ────────────────────────
echo.
echo [2/5] Installing dependencies (this may take several minutes)…
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
pip install pyinstaller --quiet

REM ── 3. Download ML models for bundling ───────────────────────
echo.
echo [3/5] Downloading ML models for offline bundling…
echo        (TrOCR ~1 GB + EasyOCR ~250 MB – skipped if already downloaded)
python download_models.py
if errorlevel 1 (
    echo.
    echo [3/5] WARNING: Model download failed. The EXE may require internet
    echo        access on first launch. Check your connection and retry.
)

REM ── 4. Run PyInstaller ────────────────────────────────────────
echo.
echo [4/5] Building EXE with PyInstaller…
pyinstaller handwriting_extractor.spec --noconfirm

REM ── 5. Done ───────────────────────────────────────────────────
echo.
if exist "dist\HandwritingExtractor\HandwritingExtractor.exe" (
    echo [5/5] SUCCESS!
    echo.
    echo   EXE location:
    echo     dist\HandwritingExtractor\HandwritingExtractor.exe
    echo.
    echo   To distribute: zip the entire  dist\HandwritingExtractor\  folder.
) else (
    echo [5/5] Build may have failed – check output above for errors.
)

echo.
pause
