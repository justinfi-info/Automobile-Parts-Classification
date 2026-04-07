# Terminal usage (Windows / PowerShell)

## 1) Go to the project folder

From your workspace root:

```powershell
cd "c:\Users\Mujee\Downloads\ML Projects\Car parts"
```

## 2) (Recommended) Create + activate a venv

TensorFlow often does not publish wheels for the newest Python versions immediately. If you see
`No matching distribution found for tensorflow`, install Python 3.11 or 3.12 and create the venv
with that interpreter (on Windows, the `py` launcher makes this easy).

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 3) Install dependencies

```powershell
python -m pip install -r requirement.txt
```

## 4) Run the Streamlit app

Note: the command is `streamlit run ...` (not `rum ...`).

```powershell
streamlit run .\Automobile-Parts-Classification\.streamlit\streamlit_app.py
```

## 5) Model file requirement

The app expects a non-empty TFLite model at:

`Automobile-Parts-Classification\models\compressed_model.tflite`

If that file is empty (0 bytes), export/build your model again and replace it.
