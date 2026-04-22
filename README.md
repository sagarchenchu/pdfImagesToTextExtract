# Handwriting Text Extractor

> Extract **handwritten text** from PDF files and images using a modern
> **EasyOCR + TrOCR** pipeline — packaged as a Windows `.exe`.

---

## ✨ Features

| | |
|---|---|
| 📄 **PDF & Image support** | Upload a multi-page PDF or any image (PNG, JPG, TIFF, BMP, WebP) |
| 🔍 **EasyOCR layout detection** | Locates individual text lines/regions on each page |
| ✍️ **TrOCR handwriting recognition** | Microsoft `trocr-large-handwritten` reads each detected region |
| 📊 **Live progress bar** | Shows per-region progress during extraction |
| 💾 **Save results** | Export the full extracted text to a `.txt` file |
| 🖥️ **Windows EXE** | No Python installation needed on the target machine |

---

## 🚀 Quick Start (pre-built EXE)

1. Go to the **[Releases](../../releases)** tab and download
   `HandwritingExtractor-Windows-x64.zip`
2. Extract the zip to any folder, e.g. `C:\HandwritingExtractor\`
3. Download the ML models (see table below) and place them in a `models\` folder
   **next to** `HandwritingExtractor.exe`
4. Run **`HandwritingExtractor.exe`**

> **Works fully offline after models are placed next to the EXE.**
> No internet connection is needed at runtime.

### 📦 Required model files

Download these files once and keep them in the folder layout shown below.

#### TrOCR — `microsoft/trocr-large-handwritten` (~1.35 GB total)

Place all files in `models\trocr\`:

| File | Download |
|------|----------|
| `config.json` | [↓ download](https://huggingface.co/microsoft/trocr-large-handwritten/resolve/main/config.json) |
| `generation_config.json` | [↓ download](https://huggingface.co/microsoft/trocr-large-handwritten/resolve/main/generation_config.json) |
| `preprocessor_config.json` | [↓ download](https://huggingface.co/microsoft/trocr-large-handwritten/resolve/main/preprocessor_config.json) |
| `tokenizer_config.json` | [↓ download](https://huggingface.co/microsoft/trocr-large-handwritten/resolve/main/tokenizer_config.json) |
| `vocab.json` | [↓ download](https://huggingface.co/microsoft/trocr-large-handwritten/resolve/main/vocab.json) |
| `merges.txt` | [↓ download](https://huggingface.co/microsoft/trocr-large-handwritten/resolve/main/merges.txt) |
| `special_tokens_map.json` | [↓ download](https://huggingface.co/microsoft/trocr-large-handwritten/resolve/main/special_tokens_map.json) |
| `pytorch_model.bin` (~1.35 GB) | [↓ download](https://huggingface.co/microsoft/trocr-large-handwritten/resolve/main/pytorch_model.bin) |

#### EasyOCR (~250 MB total)

Place the `.pth` files (unzipped) in `models\easyocr\`:

| File | Download |
|------|----------|
| `craft_mlt_25k.pth` (~90 MB) | [↓ download (zip)](https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip) |
| `english_g2.pth` (~160 MB) | [↓ download (zip)](https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip) |

> Unzip each archive and place the `.pth` file inside `models\easyocr\`.

### 📁 Required folder layout

```
HandwritingExtractor\
    HandwritingExtractor.exe
    models\
        trocr\
            config.json
            generation_config.json
            preprocessor_config.json
            tokenizer_config.json
            vocab.json
            merges.txt
            special_tokens_map.json
            pytorch_model.bin
        easyocr\
            craft_mlt_25k.pth
            english_g2.pth
    (other EXE support files…)
```

---

## 🛠️ Build from source (Windows)

### Prerequisites

* Windows 10/11 (64-bit)
* Python **3.10 or 3.11** (64-bit) – [python.org](https://www.python.org/downloads/)
* Git

### Steps

```bat
git clone https://github.com/sagarchenchu/pdfImagesToTextExtract.git
cd pdfImagesToTextExtract

REM Double-click build.bat  OR  run the commands below:
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install pyinstaller
pyinstaller handwriting_extractor.spec --noconfirm
```

The finished application is in `dist\HandwritingExtractor\`.
Place the `models\` folder (see layout above) next to the EXE, then zip and distribute.

> **Proxy / offline environments:** the build step requires no internet access.
> Download the model files on any machine with internet access, copy them over,
> and place them in `models\` next to the EXE.

### Run without building

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

---

## ⚙️ Automatic CI/CD build

Every push to `main` that changes `app.py`, `requirements.txt`, or the spec file
triggers the **Build Windows EXE** GitHub Actions workflow
(`.github/workflows/build-exe.yml`).

* The compiled application zip (without models) is uploaded as a **workflow artifact** (retained 30 days).
* Push a `v*.*.*` tag to automatically create a **GitHub Release** with the zip attached.
* After downloading, add the `models\` folder next to the EXE (see layout above).

---

## 🏗️ Architecture

```
PDF / Image
    │
    ▼
PyMuPDF  ──►  page images (2× zoom for quality)
    │
    ▼
EasyOCR  ──►  bounding boxes for each text region
    │
    ▼  (crop each region)
TrOCR    ──►  handwritten text string
    │
    ▼
tkinter GUI  ──►  live display + save to .txt
```

### Key libraries

| Library | Role |
|---------|------|
| `easyocr` | Text **region / line detection** (CRAFT detector) |
| `transformers` (TrOCR) | **Handwriting recognition** per region |
| `PyMuPDF` (`fitz`) | PDF → image conversion |
| `Pillow` | Image loading & cropping |
| `torch` | Inference backend for both models |
| `tkinter` | Cross-platform GUI (ships with Python) |
| `PyInstaller` | Packaging to a Windows `.exe` |

---

## 📋 Requirements

```
Python 3.10 / 3.11 (64-bit)
torch >= 2.1
torchvision >= 0.16
transformers >= 4.37
easyocr >= 1.7.1
PyMuPDF >= 1.23
Pillow >= 10.0
numpy >= 1.24
opencv-python-headless >= 4.8
```

See [`requirements.txt`](requirements.txt) for exact versions.

---

## 📝 License

MIT

