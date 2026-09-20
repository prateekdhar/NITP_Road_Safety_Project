"""Image detection page registered by app.py."""
import io
import json
import math
from collections import Counter
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from road_damage import DESCRIPTIONS
from road_damage.inference import load_detector, detections

ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(page_title="Road Damage Inspector", page_icon="🛣️", layout="wide")
st.title("Road Damage Inspector")
st.write("Locate cracks and potholes in road images and review their predicted categories.")
st.page_link("pages/1_Model_Statistics.py", label="View model statistics and paper results", icon="📊")


@st.cache_resource
def detector(path, modified_ns):
    return load_detector(path)


with st.sidebar:
    st.header("Detection settings")
    checkpoints = sorted((p for p in (ROOT / "runs/train").glob("*/weights/best.pt")
                         if not p.parent.parent.name.startswith("smoke_")), key=lambda p: p.stat().st_mtime_ns)
    default = str(checkpoints[-1].relative_to(ROOT)) if checkpoints else "runs/train/baseline/weights/best.pt"
    weights = st.text_input("Trained model path", value=default)
    conf = st.slider("Confidence threshold", .05, .95, .25, .05)
    st.caption("Higher thresholds show fewer, more confident predictions.")
    st.divider()
    st.write("**Damage categories**")
    for code, description in DESCRIPTIONS.items():
        st.write(f"**{code}** · {description}")

checkpoint = Path(weights)
if not checkpoint.is_absolute():
    checkpoint = ROOT / checkpoint
if not checkpoint.is_file():
    st.info("No trained road-damage model found. Prepare RDD2022 and train your first model using the commands below.")
    st.code('python -m road_damage.dataset --source "data/raw/RDD2022"\n'
            'python -m road_damage.train --variant baseline --epochs 300 --device cpu', language="powershell")
    st.caption("For a compatible CUDA installation, use --device 0. See README.md for dataset and setup instructions.")
    st.stop()

st.subheader("Model performance (saved validation results)")
try:
    path = checkpoint.resolve()
    model = detector(str(path), path.stat().st_mtime_ns)
    saved_metrics = (model.ckpt or {}).get("train_metrics") or {}
    metric_fields = [
        ("mAP@50", "metrics/mAP50(B)"),
        ("Precision", "metrics/precision(B)"),
        ("Recall", "metrics/recall(B)"),
        ("mAP@50-95", "metrics/mAP50-95(B)"),
    ]
    for column, (label, key) in zip(st.columns(4), metric_fields):
        value = saved_metrics.get(key)
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = float("nan")
        column.metric(label, f"{value:.2%}" if math.isfinite(value) and 0 <= value <= 1 else "Unavailable")
    st.caption(
        "Object detection uses mAP rather than a single classification accuracy. "
        "These are historical validation metrics stored in the selected checkpoint, "
        "not a new test evaluation or the reference paper's results. "
        "Prediction confidence is separate from model performance."
    )
    st.caption(f"Selected checkpoint: {checkpoint.name} | Run: {checkpoint.parent.parent.name}")
except (ValueError, OSError, RuntimeError) as exc:
    st.warning(f"Saved model performance could not be loaded: {exc}")

uploaded = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.write("Upload an image to begin. Results can be downloaded as an annotated image and a JSON report.")
    st.stop()

try:
    source = Image.open(uploaded)
    if source.width * source.height > 25_000_000:
        st.error("Please use an image with at most 25 million pixels.")
        st.stop()
    original = ImageOps.exif_transpose(source).convert("RGB")
except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
    st.error("This file could not be read as an image. Upload a valid JPG or PNG.")
    st.stop()

left, right = st.columns(2)
left.image(original, caption="Original image", use_container_width=True)
if st.button("Detect road damage", type="primary"):
    try:
        path = checkpoint.resolve()
        with st.spinner("Inspecting the road image…"):
            model = detector(str(path), path.stat().st_mtime_ns)
            result = model.predict(original, conf=conf, device="cpu", verbose=False)[0]
        rows = detections(result)
        annotated = Image.fromarray(result.plot()[..., ::-1])
        right.image(annotated, caption="Detected damage", use_container_width=True)
        counts = Counter(row["code"] for row in rows)
        for column, (code, description) in zip(st.columns(4), DESCRIPTIONS.items()):
            column.metric(description, counts[code])
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.info("No damage was detected at this confidence threshold. This does not establish that the road is damage-free.")
        buffer = io.BytesIO()
        annotated.save(buffer, format="PNG")
        st.download_button("Download annotated image", buffer.getvalue(), "road-detections.png", "image/png", on_click="ignore")
        report = dict(image=uploaded.name, confidence_threshold=conf, detections=rows)
        st.download_button("Download detection report", json.dumps(report, indent=2), "detections.json", "application/json", on_click="ignore")
    except (ValueError, OSError, RuntimeError) as exc:
        st.error(f"Detection could not run: {exc}")
