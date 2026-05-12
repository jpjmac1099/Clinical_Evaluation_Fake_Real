from __future__ import annotations

import base64
import random
import secrets
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image


APP_TITLE = "Clinician Fake/Real Classification Demo"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

DEMO_DATA_DIR = Path("demo_data")
DEMO_GT_PATH = DEMO_DATA_DIR / "demo_gt.tsv"


def init_state():
    defaults = {
        "dataset": None,
        "reader_name": "",
        "seed": secrets.randbelow(10**9),
        "started": False,
        "current_idx": 0,
        "responses": [],
        "submitted": False,
        "evaluation_type": "frames",
        "session_uid": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_session():
    st.session_state.dataset = None
    st.session_state.reader_name = ""
    st.session_state.seed = secrets.randbelow(10**9)
    st.session_state.started = False
    st.session_state.current_idx = 0
    st.session_state.responses = []
    st.session_state.submitted = False
    st.session_state.evaluation_type = "frames"
    st.session_state.session_uid = datetime.now().strftime("%Y%m%d_%H%M%S")


def infer_group_from_view(view_label: str) -> str:
    view_label = str(view_label).upper()

    if view_label in {"A4C", "A5C", "A3C", "A2C"}:
        return "apical"

    if view_label in {"PLAX", "PSAX"}:
        return "parasternal"

    if "SUB" in view_label:
        return "subcostal"

    return "unknown_group"


def load_gt() -> pd.DataFrame:
    if not DEMO_GT_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DEMO_GT_PATH}. "
            "Create demo_data/demo_gt.tsv."
        )

    gt_df = pd.read_csv(DEMO_GT_PATH, sep="\t")

    required_cols = {
        "mixed_name",
        "true_label",
        "original_file",
        "method",
        "view_label",
        "original_patient",
    }

    missing = required_cols - set(gt_df.columns)
    if missing:
        raise ValueError(f"Missing columns in demo_gt.tsv: {sorted(missing)}")

    gt_df = gt_df.copy()
    gt_df["mixed_name"] = gt_df["mixed_name"].astype(str).str.strip()
    gt_df["true_label"] = gt_df["true_label"].astype(str).str.lower().str.strip()
    gt_df["original_file"] = gt_df["original_file"].astype(str).str.strip()
    gt_df["method"] = gt_df["method"].astype(str).str.strip()
    gt_df["view_label"] = gt_df["view_label"].astype(str).str.strip()
    gt_df["original_patient"] = gt_df["original_patient"].astype(str).str.strip()

    if "view_group" not in gt_df.columns:
        gt_df["view_group"] = gt_df["view_label"].apply(infer_group_from_view)
    else:
        gt_df["view_group"] = gt_df["view_group"].astype(str).str.strip()
        gt_df["view_group"] = gt_df.apply(
            lambda r: infer_group_from_view(r["view_label"])
            if r["view_group"] in {"", "unknown_group", "nan"}
            else r["view_group"],
            axis=1,
        )

    if "source_folder" not in gt_df.columns:
        gt_df["source_folder"] = ""

    if "source_frame" not in gt_df.columns:
        gt_df["source_frame"] = ""

    gt_df["mixed_stem"] = gt_df["mixed_name"].apply(lambda x: Path(str(x)).stem)

    return gt_df


def find_demo_media_folder(evaluation_type: str) -> Path:
    folder = DEMO_DATA_DIR / ("frames" if evaluation_type == "frames" else "videos")

    if not folder.exists():
        raise FileNotFoundError(f"Could not find media folder: {folder}")

    return folder


def load_dataset(evaluation_type: str) -> pd.DataFrame:
    gt_df = load_gt()
    media_folder = find_demo_media_folder(evaluation_type)

    allowed_exts = IMAGE_EXTS if evaluation_type == "frames" else VIDEO_EXTS

    files = [
        p for p in media_folder.iterdir()
        if p.is_file() and p.suffix.lower() in allowed_exts
    ]

    if not files:
        raise RuntimeError(f"No {evaluation_type} files found in {media_folder}")

    rows = []

    for p in sorted(files):
        stem = p.stem
        match = gt_df[gt_df["mixed_stem"] == stem]

        if len(match) == 0:
            continue

        row = match.iloc[0].copy()
        row["media_path"] = str(p)
        row["displayed_file"] = p.name
        row["label"] = row["view_label"]

        rows.append(row)

    if not rows:
        raise RuntimeError("No matching GT entries found for the demo files.")

    df = pd.DataFrame(rows)

    rng = random.Random(int(st.session_state.seed))
    df = df.sample(frac=1, random_state=rng.randint(0, 10**6)).reset_index(drop=True)

    return df


def record_answer(prediction: str):
    df = st.session_state.dataset
    idx = st.session_state.current_idx
    row = df.iloc[idx]

    true_label = str(row.get("true_label", "")).lower().strip()
    correct = prediction == true_label

    response = {
        "session_uid": st.session_state.session_uid,
        "reader_name": st.session_state.reader_name,
        "evaluation_type": st.session_state.evaluation_type,
        "sample_idx": idx,
        "mixed_name": str(row.get("mixed_name", "")),
        "displayed_file": str(row.get("displayed_file", "")),
        "original_file": str(row.get("original_file", "")),
        "original_patient": str(row.get("original_patient", "")),
        "method": str(row.get("method", "")),
        "view_group": str(row.get("view_group", "")),
        "view_label": str(row.get("view_label", "")),
        "prediction": prediction,
        "true_label": true_label,
        "correct": bool(correct),
        "timestamp": datetime.now().isoformat(),
    }

    st.session_state.responses.append(response)
    st.session_state.current_idx += 1


def results_df() -> pd.DataFrame:
    return pd.DataFrame(st.session_state.responses)


def show_media(media_path: Path):
    display_width = 240

    if st.session_state.evaluation_type == "frames":
        image = Image.open(media_path)
        st.image(image, width=display_width)

    else:
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


def show_results():
    df = results_df()

    st.subheader("Your Results")

    if df.empty:
        st.info("No answers yet.")
        return

    total = len(df)
    correct = int(df["correct"].sum())
    accuracy = correct / total if total else 0.0

    st.success(f"Your accuracy: {correct}/{total} = {accuracy:.1%}")

    by_view_method = (
        df.groupby(["view_label", "method"], dropna=False)
        .agg(
            n=("correct", "size"),
            correct=("correct", "sum"),
            accuracy=("correct", "mean"),
        )
        .reset_index()
    )

    by_view_method["accuracy"] = (by_view_method["accuracy"] * 100).round(1)

    st.subheader("Accuracy by view and method")
    st.dataframe(by_view_method, use_container_width=True)

    by_method = (
        df.groupby("method", dropna=False)
        .agg(
            n=("correct", "size"),
            correct=("correct", "sum"),
            accuracy=("correct", "mean"),
        )
        .reset_index()
    )

    by_method["accuracy"] = (by_method["accuracy"] * 100).round(1)

    st.subheader("Accuracy by method")
    st.dataframe(by_method, use_container_width=True)

    by_view = (
        df.groupby("view_label", dropna=False)
        .agg(
            n=("correct", "size"),
            correct=("correct", "sum"),
            accuracy=("correct", "mean"),
        )
        .reset_index()
    )

    by_view["accuracy"] = (by_view["accuracy"] * 100).round(1)

    st.subheader("Accuracy by view")
    st.dataframe(by_view, use_container_width=True)

    with st.expander("See all answers"):
        st.dataframe(df, use_container_width=True)


st.set_page_config(page_title=APP_TITLE, layout="wide")
init_state()

st.title(APP_TITLE)
st.caption("Demo version: embedded cases, no email, no password. Accuracy appears after submission.")

with st.sidebar:
    st.header("Demo settings")

    st.session_state.reader_name = st.text_input(
        "Your name",
        value=st.session_state.reader_name,
        placeholder="Optional",
    )

    st.session_state.evaluation_type = st.radio(
        "Evaluation type",
        options=["frames", "videos"],
        format_func=lambda x: "Frames" if x == "frames" else "Videos",
        disabled=st.session_state.started,
    )

    st.caption(f"Session seed: {st.session_state.seed}")

    if st.button("Reset demo"):
        reset_session()
        st.rerun()


if st.session_state.dataset is None:
    try:
        st.session_state.dataset = load_dataset(st.session_state.evaluation_type)
    except Exception as e:
        st.error(f"Could not load demo data: {e}")
        st.stop()


df = st.session_state.dataset
n_total = len(df)
answered = len(st.session_state.responses)

top1, top2, top3 = st.columns(3)
top1.metric("Total samples", n_total)
top2.metric("Answered", answered)
top3.metric("Remaining", n_total - answered)

st.progress(answered / n_total if n_total else 0.0)

if not st.session_state.started:
    st.subheader("Start demo")

    if st.button("Start classification", type="primary"):
        st.session_state.started = True
        st.rerun()

    st.stop()


idx = st.session_state.current_idx

if idx < n_total:
    row = df.iloc[idx]
    media_path = Path(row["media_path"])

    left, spacer, right = st.columns([0.8, 0.1, 1.2])

    with left:
        st.subheader(f"Sample {idx + 1} / {n_total}")
        show_media(media_path)

    with right:
        st.subheader("Classification")

        col_real, col_fake = st.columns(2)

        with col_real:
            if st.button("Real", use_container_width=True, type="primary"):
                record_answer("real")
                st.rerun()

        with col_fake:
            if st.button("Fake", use_container_width=True):
                record_answer("fake")
                st.rerun()

        st.markdown("---")

        if st.button(
            "Submit now",
            use_container_width=True,
            disabled=answered == 0,
        ):
            st.session_state.current_idx = n_total
            st.session_state.submitted = True
            st.rerun()

    st.stop()


st.success("Demo complete.")
show_results()

if st.button("Start new demo"):
    reset_session()
    st.rerun()
