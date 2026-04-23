@echo off
REM ============================================================
REM  build_api.bat  –  Build TextExtractAPI.exe on Windows
REM ============================================================
REM Requirements:
REM   * Python 3.10 or 3.11 (64-bit) installed and on PATH
REM
REM Builds a headless HTTP API server EXE that listens on
REM   http://127.0.0.1:9102/extractText
REM and accepts PDF or image file uploads, returning extracted text.
REM
REM ML models are NOT downloaded or bundled during the build.
REM After building, place a  models\  folder next to the EXE:
REM
REM   dist\TextExtractAPI\
REM       TextExtractAPI.exe
REM       models\
REM           trocr\          <- microsoft/trocr-large-handwritten files
REM           easyocr\        <- craft_mlt_25k.pth + english_g2.pth
REM
REM See README.md for model download links.
REM ============================================================

echo.
echo ============================================================
echo  Text Extract API Server – Windows EXE Builder
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
set "PYTHONWARNINGS=ignore::FutureWarning:timm.models.layers,ignore::DeprecationWarning:torch.distributed"
pyinstaller api_server.spec --noconfirm

REM ── 4. Done ───────────────────────────────────────────────────
echo.
if exist "dist\TextExtractAPI\TextExtractAPI.exe" (
    echo [4/4] SUCCESS!
    echo.
    echo   EXE location:
    echo     dist\TextExtractAPI\TextExtractAPI.exe
    echo.
    echo   NEXT STEP: place your models\ folder next to the EXE:
    echo     dist\TextExtractAPI\models\trocr\       (TrOCR files)
    echo     dist\TextExtractAPI\models\easyocr\     (EasyOCR .pth files)
    echo.
    echo   Run the EXE then send requests:
    echo     curl -X POST http://127.0.0.1:9102/extractText -F "file=@your_file.pdf"
    echo.
    echo   Then zip and distribute the dist\TextExtractAPI\ folder.
) else (
    echo [4/4] Build may have failed – check output above for errors.
)

echo.
pause
