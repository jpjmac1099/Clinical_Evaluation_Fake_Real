from __future__ import annotations

import base64
import io
import random
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image


APP_TITLE = "Echo Synthetic Real/Fake Demo"

DEMO_DIR = Path("demo_cases")
DEMO_LABELS = DEMO_DIR / "demo_labels.tsv"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def init_state():
    defaults = {
        "dataset": None,
        "current_idx": 0,
        "responses": [],
        "started": False,
        "mode": "Embedded demo cases",
        "session_uid": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_demo():
    st.session_state.dataset = None
    st.session_state.current_idx = 0
    st.session_state.responses = []
    st.session_state.started = False
    st.session_state.session_uid = datetime.now().strftime("%Y%m%d_%H%M%S")


def find_media_path(base_dir: Path, mixed_name: str) -> Path:
    stem = Path(mixed_name).stem

    matches = [
        p for p in base_dir.iterdir()
        if p.is_file()
        and p.stem == stem
        and p.suffix.lower() in IMAGE_EXTS.union(VIDEO_EXTS)
    ]

    if not matches:
        raise FileNotFoundError(f"Missing media file for {mixed_name}")

    return matches[0]


def load_embedded_demo() -> pd.DataFrame:
    if not DEMO_LABELS.exists():
        raise FileNotFoundError(
            f"Missing {DEMO_LABELS}. Create demo_cases/demo_labels.tsv first."
        )

    df = pd.read_csv(DEMO_LABELS, sep="\t")
    rows = []

    for _, row in df.iterrows():
        media_path = find_media_path(DEMO_DIR, row["mixed_name"])

        new_row = row.copy()
        new_row["media_path"] = str(media_path)
        new_row["displayed_file"] = media_path.name
        rows.append(new_row)

    out = pd.DataFrame(rows)
    out = out.sample(frac=1, random_state=8).reset_index(drop=True)
    return out


def extract_zip_to_temp(zip_file) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="demo_upload_"))
    zip_path = temp_dir / "upload.zip"

    with open(zip_path, "wb") as f:
        f.write(zip_file.getbuffer())

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(temp_dir)

    return temp_dir


def load_zip_demo(zip_file) -> pd.DataFrame:
    temp_dir = extract_zip_to_temp(zip_file)

    label_files = list(temp_dir.rglob("demo_labels.tsv")) + list(temp_dir.rglob("mixed_labels.txt"))

    if not label_files:
        raise FileNotFoundError(
            "ZIP must contain demo_labels.tsv or mixed_labels.txt."
        )

    labels_path = label_files[0]
    media_dir = labels_path.parent

    df = pd.read_csv(labels_path, sep="\t")
    rows = []

    for _, row in df.iterrows():
        media_path = find_media_path(media_dir, row["mixed_name"])

        new_row = row.copy()
        new_row["media_path"] = str(media_path)
        new_row["displayed_file"] = media_path.name
        rows.append(new_row)

    out = pd.DataFrame(rows)
    out = out.sample(frac=1, random_state=8).reset_index(drop=True)
    return out


def load_single_upload(uploaded_file) -> pd.DataFrame:
    temp_dir = Path(tempfile.mkdtemp(prefix="single_demo_"))
    media_path = temp_dir / uploaded_file.name

    with open(media_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return pd.DataFrame(
        [
            {
                "mixed_name": uploaded_file.name,
                "true_label": "unknown",
                "method": "uploaded",
                "view_group": "unknown",
                "view_label": "unknown",
                "original_patient": "uploaded_case",
                "media_path": str(media_path),
                "displayed_file": uploaded_file.name,
            }
        ]
    )


def show_media(media_path: Path):
    suffix = media_path.suffix.lower()
    display_width = 280

    if suffix in IMAGE_EXTS:
        image = Image.open(media_path)
        st.image(image, width=display_width)

    elif suffix in VIDEO_EXTS:
        with open(media_path, "rb") as f:
            video_bytes = f.read()

        video_base64 = base64.b64encode(video_bytes).decode()

        components.html(
            f"""
            <html>
            <body style="margin:0; padding:0; overflow:hidden;">
                <video
                    width="{display_width}"
                    controls
                    muted
                    loop
                    style="width:{display_width}px; height:auto; display:block;"
                >
                    <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
                </video>
            </body>
            </html>
            """,
            width=display_width + 30,
            height=display_width + 80,
            scrolling=False,
        )

    else:
        st.error(f"Unsupported media type: {suffix}")


def record_answer(prediction: str):
    df = st.session_state.dataset
    idx = st.session_state.current_idx
    row = df.iloc[idx]

    true_label = str(row.get("true_label", "unknown")).lower().strip()

    correct = None
    if true_label in {"real", "fake"}:
        correct = prediction == true_label

    response = {
        "sample_idx": idx,
        "mixed_name": row.get("mixed_name", ""),
        "displayed_file": row.get("displayed_file", ""),
        "original_patient": row.get("original_patient", ""),
        "method": row.get("method", ""),
        "view_group": row.get("view_group", ""),
        "view_label": row.get("view_label", ""),
        "prediction": prediction,
        "true_label": true_label,
        "correct": correct,
        "timestamp": datetime.now().isoformat(),
    }

    st.session_state.responses.append(response)
    st.session_state.current_idx += 1


def results_df() -> pd.DataFrame:
    return pd.DataFrame(st.session_state.responses)


def show_results():
    df = results_df()

    st.subheader("Live Results")

    if df.empty:
        st.info("No answers yet.")
        return

    known = df[df["correct"].notna()].copy()

    if not known.empty:
        accuracy = known["correct"].mean()
        st.metric("Accuracy", f"{accuracy:.1%}")

        summary = (
            known.groupby(["view_label", "method"], dropna=False)
            .agg(
                n=("correct", "size"),
                correct=("correct", "sum"),
                accuracy=("correct", "mean"),
            )
            .reset_index()
        )

        summary["accuracy"] = (summary["accuracy"] * 100).round(1)

        st.dataframe(summary, use_container_width=True)
    else:
        st.info("Ground truth is unknown for uploaded-only cases.")

    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download results CSV",
        data=csv,
        file_name=f"demo_results_{st.session_state.session_uid}.csv",
        mime="text/csv",
    )


st.set_page_config(page_title=APP_TITLE, layout="wide")
init_state()

st.title(APP_TITLE)
st.caption("Presentation demo app with embedded echocardiography cases.")

with st.sidebar:
    st.header("Demo settings")

    st.session_state.mode = st.radio(
        "Data source",
        [
            "Embedded demo cases",
            "Upload ZIP with labels",
            "Upload one media file",
        ],
    )

    if st.button("Reset demo"):
        reset_demo()
        st.rerun()

    st.markdown("---")
    st.caption("Tip: add this app link or QR code to your presentation slide.")


st.subheader("1. Load cases")

if st.session_state.dataset is None:
    if st.session_state.mode == "Embedded demo cases":
        if st.button("Load embedded demo cases", type="primary"):
            try:
                st.session_state.dataset = load_embedded_demo()
                st.rerun()
            except Exception as e:
                st.error(f"Could not load embedded cases: {e}")

    elif st.session_state.mode == "Upload ZIP with labels":
        uploaded_zip = st.file_uploader(
            "Upload ZIP containing media files and demo_labels.tsv or mixed_labels.txt",
            type=["zip"],
        )

        if st.button("Load ZIP", type="primary"):
            if uploaded_zip is None:
                st.error("Please upload a ZIP first.")
                st.stop()

            try:
                st.session_state.dataset = load_zip_demo(uploaded_zip)
                st.rerun()
            except Exception as e:
                st.error(f"Could not load ZIP: {e}")

    elif st.session_state.mode == "Upload one media file":
        uploaded_media = st.file_uploader(
            "Upload one image or video",
            type=list(IMAGE_EXTS.union(VIDEO_EXTS)),
        )

        if st.button("Load media", type="primary"):
            if uploaded_media is None:
                st.error("Please upload a media file first.")
                st.stop()

            try:
                st.session_state.dataset = load_single_upload(uploaded_media)
                st.rerun()
            except Exception as e:
                st.error(f"Could not load media: {e}")

    st.stop()


df = st.session_state.dataset
n_total = len(df)
answered = len(st.session_state.responses)

top1, top2, top3 = st.columns(3)
top1.metric("Total cases", n_total)
top2.metric("Answered", answered)
top3.metric("Remaining", n_total - answered)

st.progress(answered / n_total if n_total else 0)

if not st.session_state.started:
    st.subheader("2. Start live test")

    if st.button("Start", type="primary"):
        st.session_state.started = True
        st.rerun()

    st.stop()


idx = st.session_state.current_idx

if idx < n_total:
    row = df.iloc[idx]
    media_path = Path(row["media_path"])

    left, right = st.columns([1, 1.2])

    with left:
        st.subheader(f"Case {idx + 1} / {n_total}")
        show_media(media_path)

    with right:
        st.subheader("Your guess")

        st.write(f"View: **{row.get('view_label', 'unknown')}**")
        st.write(f"Group: **{row.get('view_group', 'unknown')}**")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Real", use_container_width=True, type="primary"):
                record_answer("real")
                st.rerun()

        with col2:
            if st.button("Fake", use_container_width=True):
                record_answer("fake")
                st.rerun()

        with st.expander("Demo metadata"):
            st.write(f"Method: `{row.get('method', '')}`")
            st.write(f"Original patient: `{row.get('original_patient', '')}`")
            st.write(f"Displayed file: `{row.get('displayed_file', '')}`")

    st.stop()


st.success("Demo complete.")
show_results()
