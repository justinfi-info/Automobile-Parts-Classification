from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import streamlit as st
from PIL import Image


APP_TITLE = "Automobile Parts Classification"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "compressed_model.tflite"
DEFAULT_DATASET_TRAIN_DIR = Path(__file__).resolve().parents[1] / "car parts 50" / "train"


@dataclass(frozen=True)
class ModelInfo:
    input_size: tuple[int, int]
    input_dtype: np.dtype
    input_quant: tuple[float, int] | None
    output_dtype: np.dtype
    output_quant: tuple[float, int] | None


@dataclass(frozen=True)
class InferenceEnv:
    interpreter: object | None
    model_info: ModelInfo | None
    error: str | None


def _list_class_names_from_train_dir(train_dir: Path) -> list[str]:
    if not train_dir.exists() or not train_dir.is_dir():
        return []
    names = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
    return names


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.max(x)
    exps = np.exp(x)
    return exps / (np.sum(exps) + 1e-12)


def _human_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def _inject_css() -> None:
    st.markdown(
        """
        <style>
          :root {
            --card-bg: rgba(255,255,255,0.7);
            --card-border: rgba(0,0,0,0.08);
            --muted: rgba(0,0,0,0.55);
          }
          .block-container { padding-top: 1.2rem; }
          .app-subtle { color: var(--muted); font-size: 0.92rem; }
          .kpi-card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 14px;
            padding: 14px 14px 12px 14px;
            backdrop-filter: blur(6px);
          }
          .kpi-title { font-size: 0.78rem; color: var(--muted); margin-bottom: 4px; }
          .kpi-value { font-size: 1.35rem; font-weight: 700; line-height: 1.1; }
          .kpi-sub { font-size: 0.82rem; color: var(--muted); margin-top: 6px; }
          .badge {
            display: inline-block;
            font-size: 0.75rem;
            padding: 2px 8px;
            border-radius: 999px;
            border: 1px solid var(--card-border);
            background: rgba(255,255,255,0.55);
            margin-left: 6px;
          }
          .result-card {
            border-radius: 16px;
            padding: 22px 18px;
            border: 1px solid rgba(255,255,255,0.25);
            background: linear-gradient(135deg, rgba(148,70,247,0.95), rgba(255,92,125,0.95));
            color: white;
          }
          .result-label { font-size: 2.0rem; font-weight: 800; letter-spacing: 0.02em; }
          .result-conf { font-size: 1.15rem; opacity: 0.95; margin-top: 6px; font-weight: 600; }
          .section-title { font-size: 1.18rem; font-weight: 750; margin: 10px 0 6px 0; }
          .thin-rule { border-top: 1px solid rgba(0,0,0,0.08); margin: 10px 0 10px 0; }
          .metric-grid {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 10px;
          }
          .metric-item {
            background: rgba(255,255,255,0.7);
            border: 1px solid var(--card-border);
            border-radius: 14px;
            padding: 10px 12px;
          }
          .metric-k { font-size: 0.78rem; color: var(--muted); }
          .metric-v { font-size: 1.05rem; font-weight: 750; margin-top: 2px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _kpi_card(title: str, value: str, sub: str | None = None, badge: str | None = None) -> None:
    badge_html = f'<span class="badge">{badge}</span>' if badge else ""
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-title">{title}{badge_html}</div>
          <div class="kpi-value">{value}</div>
          {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _find_sample_image(train_dir: Path) -> Optional[Path]:
    if not train_dir.exists():
        return None
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    for pat in patterns:
        candidate = next(train_dir.rglob(pat), None)
        if candidate is not None and candidate.is_file():
            return candidate
    return None


@st.cache_resource(show_spinner=False)
def _load_tflite_interpreter(model_path: str):
    # Prefer the lightweight runtime if available, else fall back to TensorFlow.
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore

        interpreter = Interpreter(model_path=model_path)
    except ModuleNotFoundError:
        import tensorflow as tf

        interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def _get_model_info(interpreter) -> ModelInfo:
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_shape = input_details["shape"]
    # Expected: [1, H, W, C]
    input_h = int(input_shape[1])
    input_w = int(input_shape[2])

    input_dtype = np.dtype(input_details["dtype"])
    output_dtype = np.dtype(output_details["dtype"])

    input_quant = input_details.get("quantization")
    output_quant = output_details.get("quantization")
    input_quant_tuple = None
    output_quant_tuple = None
    if isinstance(input_quant, (tuple, list)) and len(input_quant) == 2:
        scale, zero_point = float(input_quant[0]), int(input_quant[1])
        if scale != 0.0:
            input_quant_tuple = (scale, zero_point)
    if isinstance(output_quant, (tuple, list)) and len(output_quant) == 2:
        scale, zero_point = float(output_quant[0]), int(output_quant[1])
        if scale != 0.0:
            output_quant_tuple = (scale, zero_point)

    return ModelInfo(
        input_size=(input_w, input_h),
        input_dtype=input_dtype,
        input_quant=input_quant_tuple,
        output_dtype=output_dtype,
        output_quant=output_quant_tuple,
    )


def _preprocess_image(
    image: Image.Image,
    model_info: ModelInfo,
) -> np.ndarray:
    image = image.convert("RGB").resize(model_info.input_size)
    arr = np.asarray(image)
    arr = np.expand_dims(arr, axis=0)

    if model_info.input_dtype == np.float32:
        arr = arr.astype(np.float32) / 255.0
        return arr

    if model_info.input_dtype in (np.uint8, np.int8):
        if model_info.input_quant is None:
            # Best-effort fallback for quantized inputs without quant params.
            return arr.astype(model_info.input_dtype)

        scale, zero_point = model_info.input_quant
        arr = arr.astype(np.float32) / 255.0
        arr = np.round(arr / scale + zero_point).astype(model_info.input_dtype)
        return arr

    return arr.astype(model_info.input_dtype)


def _predict(interpreter, input_tensor: np.ndarray) -> np.ndarray:
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    interpreter.set_tensor(input_index, input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_index)
    return np.asarray(output).squeeze()


def _to_probabilities(raw_output: np.ndarray, model_info: ModelInfo) -> np.ndarray:
    if raw_output.ndim != 1:
        raw_output = raw_output.reshape(-1)

    if model_info.output_dtype in (np.uint8, np.int8) and model_info.output_quant is not None:
        scale, zero_point = model_info.output_quant
        deq = (raw_output.astype(np.float32) - float(zero_point)) * float(scale)
        return _softmax(deq)

    if model_info.output_dtype == np.float32:
        values = raw_output.astype(np.float32)
        # If it already looks like probabilities, keep it; else softmax.
        if np.all(values >= 0.0) and np.all(values <= 1.0) and 0.8 <= float(np.sum(values)) <= 1.2:
            return values / (np.sum(values) + 1e-12)
        return _softmax(values)

    return _softmax(raw_output.astype(np.float32))


def _topk_indices(probs: np.ndarray, k: int) -> list[int]:
    k = max(1, min(int(k), int(probs.size)))
    return list(np.argsort(probs)[::-1][:k])


def _render_missing_model(model_path: Path) -> None:
    st.error("Model file is missing or empty.")
    st.write("Export your trained model to a non-empty `.tflite` file and place it here:")
    st.code(
        "\n".join(
            [
                "# Example (from your notebook)",
                "model.save('compressed_model.tflite')",
                "",
                "# Then copy it to:",
                str(model_path),
            ]
        )
    )


def _safe_path_from_text(text: str) -> Path:
    # Streamlit text input gives a string; keep it permissive but deterministic.
    try:
        return Path(text).expanduser().resolve()
    except Exception:
        return Path(text)


def _format_topk(
    class_names: list[str],
    probs: np.ndarray,
    topk: Iterable[int],
) -> list[tuple[str, float]]:
    results: list[tuple[str, float]] = []
    for idx in topk:
        label = class_names[idx] if 0 <= idx < len(class_names) else f"class_{idx}"
        results.append((label, float(probs[idx])))
    return results


def _try_prepare_inference(model_path: Path) -> InferenceEnv:
    if (not model_path.exists()) or (model_path.is_file() and model_path.stat().st_size == 0):
        return InferenceEnv(interpreter=None, model_info=None, error="missing_model")

    try:
        interpreter = _load_tflite_interpreter(str(model_path))
    except ModuleNotFoundError:
        return InferenceEnv(interpreter=None, model_info=None, error="missing_runtime")
    except Exception as exc:  # noqa: BLE001
        return InferenceEnv(interpreter=None, model_info=None, error=f"load_failed: {exc}")

    try:
        model_info = _get_model_info(interpreter)
    except Exception as exc:  # noqa: BLE001
        return InferenceEnv(interpreter=None, model_info=None, error=f"model_info_failed: {exc}")

    return InferenceEnv(interpreter=interpreter, model_info=model_info, error=None)


def _render_prediction_panel(
    *,
    env: InferenceEnv,
    model_path: Path,
    class_names: list[str],
    image: Image.Image,
    top_k: int,
    confidence_threshold: float,
    show_debug: bool,
) -> None:
    st.subheader("Prediction Results")

    if env.error == "missing_model":
        _render_missing_model(model_path)
        return

    if env.error == "missing_runtime":
        st.error(
            "No TFLite runtime is installed. Install either `tensorflow` (most common) or `tflite-runtime` "
            "(lightweight; not available for all OS/Python versions)."
        )
        st.code("\n".join(["python -m pip install tensorflow", "python -m pip install tflite-runtime"]))
        return

    if env.error is not None:
        st.error("Failed to initialize inference.")
        st.caption(env.error)
        return

    assert env.interpreter is not None
    assert env.model_info is not None

    input_tensor = _preprocess_image(image, env.model_info)
    raw_output = _predict(env.interpreter, input_tensor)
    probs = _to_probabilities(raw_output, env.model_info)

    top_indices = _topk_indices(probs, top_k)
    top_rows = _format_topk(class_names, probs, top_indices)
    best_label, best_prob = top_rows[0]

    if best_prob >= confidence_threshold:
        st.success("Prediction successful!")
    else:
        st.warning(
            f"Low confidence ({best_prob * 100:.2f}%) below threshold ({confidence_threshold * 100:.0f}%). "
            "Try a clearer image."
        )

    st.markdown(
        f"""
        <div class="result-card">
          <div class="result-label">{best_label}</div>
          <div class="result-conf">{best_prob * 100:.2f}% Confidence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Key Metrics</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="metric-grid">
          <div class="metric-item"><div class="metric-k">Predicted Class</div><div class="metric-v">{best_label}</div></div>
          <div class="metric-item"><div class="metric-k">Confidence</div><div class="metric-v">{best_prob * 100:.2f}%</div></div>
          <div class="metric-item"><div class="metric-k">Top Match</div><div class="metric-v">{best_label}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Top 5 Predictions</div>', unsafe_allow_html=True)
    for label, prob in top_rows[:5]:
        left, right = st.columns([3, 1])
        with left:
            st.caption(label)
            st.progress(min(max(float(prob), 0.0), 1.0))
        with right:
            st.caption(f"{prob * 100:.2f}%")

    if show_debug:
        st.divider()
        st.caption("Debug")
        st.write(
            {
                "model_path": str(model_path),
                "model_size": _human_bytes(int(model_path.stat().st_size)),
                "input_size": env.model_info.input_size,
                "input_dtype": str(env.model_info.input_dtype),
                "input_quant": env.model_info.input_quant,
                "output_dtype": str(env.model_info.output_dtype),
                "output_quant": env.model_info.output_quant,
                "num_classes_guess": int(probs.size),
                "labels_found": len(class_names),
            }
        )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    _inject_css()

    with st.sidebar:
        st.subheader("About This App")
        st.markdown("**Auto Parts Image Classifier**")
        st.markdown('<div class="app-subtle">Upload an image and get Top‑5 predictions.</div>', unsafe_allow_html=True)

        st.divider()
        st.subheader("Model Information")
        model_name = st.text_input("Model name", value="MobileNetV2")
        reported_accuracy = st.text_input("Reported accuracy", value="98.0%")

        st.divider()
        st.subheader("Settings")
        model_path_text = st.text_input("Model path", value=str(DEFAULT_MODEL_PATH))
        dataset_train_text = st.text_input("Train folder (for labels)", value=str(DEFAULT_DATASET_TRAIN_DIR))
        confidence_threshold = st.slider("Confidence threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        top_k = st.slider("Top‑K predictions", min_value=1, max_value=10, value=5)
        show_debug = st.checkbox("Show debug info", value=False)

    model_path = _safe_path_from_text(model_path_text)
    train_dir = _safe_path_from_text(dataset_train_text)

    class_names = _list_class_names_from_train_dir(train_dir)
    classes_count = len(class_names)

    st.markdown(
        f"""
        <div class="app-subtle">
          Primary model: <b>{model_name}</b> • Classes: <b>{classes_count if classes_count else "?"}</b> • Speed: <b>TFLite</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    kpi1, kpi2, kpi3 = st.columns([1, 1, 1])
    with kpi1:
        _kpi_card("Primary Model", model_name, sub=f"{reported_accuracy} accuracy", badge="Model")
    with kpi2:
        _kpi_card("Classes", str(classes_count if classes_count else "—"), sub="Auto Parts", badge="Dataset")
    with kpi3:
        _kpi_card("Speed", "TFLite", sub="Optimized inference", badge="Runtime")

    if not class_names:
        st.warning(f"No class folders found in `{train_dir}`. Labels will display as `class_<idx>`.")

    env = _try_prepare_inference(model_path)

    tab_single, tab_batch, tab_gallery = st.tabs(["Single Image", "Batch Processing", "Gallery"])

    with tab_single:
        st.markdown('<div class="section-title">Single Image Classification</div>', unsafe_allow_html=True)
        left, right = st.columns([1.15, 1.0])
        image: Image.Image | None = None
        with left:
            st.subheader("Upload Image")
            uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "webp"], key="single_upload")
            if uploaded is None:
                sample_path = _find_sample_image(train_dir)
                if sample_path is not None:
                    st.caption("No upload yet — showing a sample from your dataset.")
                    st.image(Image.open(sample_path), use_container_width=True)
                else:
                    st.info("Upload an image to start.")
            else:
                image = Image.open(uploaded)
                st.image(image, use_container_width=True)

        with right:
            if image is None:
                st.subheader("Prediction Results")
                st.info("Upload an image to see predictions.")
            else:
                _render_prediction_panel(
                    env=env,
                    model_path=model_path,
                    class_names=class_names,
                    image=image,
                    top_k=top_k,
                    confidence_threshold=confidence_threshold,
                    show_debug=show_debug,
                )

    with tab_batch:
        st.markdown('<div class="section-title">Batch Processing</div>', unsafe_allow_html=True)
        files = st.file_uploader(
            "Upload multiple images",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            key="batch_upload",
        )

        if not files:
            st.info("Upload multiple images to get a table of predictions.")
        elif env.error is not None:
            if env.error == "missing_model":
                _render_missing_model(model_path)
            elif env.error == "missing_runtime":
                st.error("Install `tensorflow` (or `tflite-runtime`) to enable batch inference.")
            else:
                st.error("Inference is not available.")
                st.caption(env.error)
        else:
            assert env.interpreter is not None
            assert env.model_info is not None

            rows: list[dict[str, object]] = []
            for f in files:
                try:
                    img = Image.open(f)
                    input_tensor = _preprocess_image(img, env.model_info)
                    raw_output = _predict(env.interpreter, input_tensor)
                    probs = _to_probabilities(raw_output, env.model_info)
                    top_idx = int(np.argmax(probs))
                    label = class_names[top_idx] if 0 <= top_idx < len(class_names) else f"class_{top_idx}"
                    conf = float(probs[top_idx])
                    rows.append(
                        {
                            "file": getattr(f, "name", "image"),
                            "predicted": label,
                            "confidence": round(conf, 6),
                            "pass_threshold": conf >= confidence_threshold,
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    rows.append({"file": getattr(f, "name", "image"), "error": str(exc)})

            st.dataframe(rows, use_container_width=True)

            # Download as CSV
            try:
                import pandas as pd  # type: ignore

                df = pd.DataFrame(rows)
                st.download_button(
                    "Download results CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="batch_predictions.csv",
                    mime="text/csv",
                )
            except ModuleNotFoundError:
                st.caption("Tip: install `pandas` to enable CSV download from the app.")

    with tab_gallery:
        st.markdown('<div class="section-title">Interactive Gallery</div>', unsafe_allow_html=True)
        if not class_names:
            st.info("Gallery needs class folders under your train directory.")
        else:
            selected = st.selectbox("Select a class", class_names)
            class_dir = train_dir / selected
            images = []
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
                images.extend(list(class_dir.glob(ext)))
            images = images[:12]

            if not images:
                st.info("No images found for this class.")
            else:
                cols = st.columns(4)
                for i, img_path in enumerate(images):
                    with cols[i % 4]:
                        st.image(Image.open(img_path), use_container_width=True, caption=img_path.name)


if __name__ == "__main__":
    main()
