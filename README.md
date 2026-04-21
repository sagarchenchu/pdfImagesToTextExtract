# Handwriting Text Extractor

> Extract **handwritten text** from PDF files and images using a modern
> **EasyOCR + TrOCR** pipeline — packaged as a single Windows `.exe`.

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
2. Extract the zip to any folder
3. Run **`HandwritingExtractor.exe`**
4. On **first launch** the app downloads ML models (~1.5 GB) — a one-time
   operation that requires internet access.  Subsequent launches are instant.

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
Zip that folder and distribute it — `HandwritingExtractor.exe` is inside.

### Run without building

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

---

## ⚙️ Automatic CI/CD build

Every push to `main` that changes `app.py`, `requirements.txt` or the spec
file triggers the **Build Windows EXE** GitHub Actions workflow
(`.github/workflows/build-exe.yml`).

* The compiled zip is uploaded as a **workflow artifact** (retained 30 days).
* Push a `v*.*.*` tag to automatically create a **GitHub Release** with the
  zip attached.

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

