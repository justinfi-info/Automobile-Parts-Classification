# Automobile Parts Classification

Streamlit demo for classifying 40 automobile parts with a TensorFlow Lite model.

## Features
- Upload a single image for real-time Top-K predictions.
- Batch upload for tabular results (CSV download when `pandas` is installed).
- Interactive gallery that browses sample images per class folder.
- Gradient prediction card with confidence and key metrics.
- Debug pane that shows model input/output shapes, dtypes, and quantization.

## Project layout
- `Automobile_parts_classification.ipynb` - notebook used to train/export the TFLite model.
- `.streamlit/streamlit_app.py` - Streamlit UI and inference code.
- `car parts 50/` - training images organized in class-named folders (used for labels/gallery).
- `models/compressed_model.tflite` - expected TFLite model file (must be non-empty).
- `TERMINAL_USAGE.md` - quick commands to set up and run on Windows/PowerShell.
- `requirement.txt` - minimal runtime dependencies.

## Prerequisites
- Python 3.12 (or 3.11) recommended. TensorFlow wheels are not yet provided for 3.13+.
- pip and a virtual environment (recommended).
- A valid TFLite model saved to `models/compressed_model.tflite`.

## Setup (Windows/PowerShell)
```powershell
cd "c:\Users\Mujee\Downloads\ML Projects\Car parts"
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirement.txt
```

## Run the app
```powershell
streamlit run .\Automobile-Parts-Classification\.streamlit\streamlit_app.py
```
If `streamlit` is not on PATH, use `python -m streamlit run ...`.

## Using the UI
- **Single Image** tab: upload a PNG/JPG/WebP; see Top-K predictions and confidence.
- **Batch Processing** tab: upload multiple images; view/download a results table.
- **Gallery** tab: pick a class to browse sample images from `car parts 50/train/<class>/`.
- Sidebar lets you change model path, labels directory, Top-K, and confidence threshold.

## Troubleshooting
- *No matching distribution found for tensorflow*: install Python 3.12/3.11 and recreate the venv.
- *Model file is missing or empty*: export your model from the notebook and place it at `models/compressed_model.tflite`.
- *Runtime missing*: install `tensorflow` (full) or `tflite-runtime` (lightweight; platform availability varies).

## Notes
- If `pandas` is installed, batch results can be downloaded as CSV from the app.
- Class labels are derived from folder names under `car parts 50/train/`; ensure they match your training set.
