from __future__ import annotations

import io
import random
import secrets
import smtplib
import tempfile
import zipfile
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image


APP_TITLE = "Clinician Fake/Real Classification"
RESULTS_EMAIL = "jpav.freitas@gmail.com"

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def init_state():
    defaults = {
        "dataset": None,
        "working_dir": None,
        "reader_id": "",
        "reader_name": "",
        "notes": "",
        "seed": secrets.randbelow(10**9),
        "started": False,
        "current_idx": 0,
        "responses": [],
        "session_uid": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "setup_done": False,
        "submitted": False,
        "evaluation_type": "frames",
        "uploaded_zip_name": "",
        "detected_view": "unknown",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_session():
    st.session_state.dataset = None
    st.session_state.working_dir = None
    st.session_state.notes = ""
    st.session_state.seed = secrets.randbelow(10**9)
    st.session_state.started = False
    st.session_state.current_idx = 0
    st.session_state.responses = []
    st.session_state.session_uid = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state.setup_done = False
    st.session_state.submitted = False
    st.session_state.evaluation_type = "frames"
    st.session_state.uploaded_zip_name = ""
    st.session_state.detected_view = "unknown"


def detect_view_from_name(name: str) -> str:
    name = str(name).lower()

    view_patterns = {
        "A4C": ["a4c", "4ch", "4_ch", "4-ch"],
        "A3C": ["a3c", "3ch", "3_ch", "3-ch"],
        "A5C": ["a5c", "5ch", "5_ch", "5-ch"],
        "A2C": ["a2c", "2ch", "2_ch", "2-ch"],
        "PSAX": ["psax"],
        "PLAX": ["plax"],
    }

    for view, patterns in view_patterns.items():
        if any(p in name for p in patterns):
            return view

    return "unknown"


def extract_zip_to_temp(zip_file) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="study_upload_"))
    zip_path = temp_dir / "study_package.zip"

    with open(zip_path, "wb") as f:
        f.write(zip_file.getbuffer())

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(temp_dir)

    return temp_dir


def find_mixed_dir(base_dir: Path) -> Path:
    mixed_dirs = [p for p in base_dir.rglob("mixed") if p.is_dir()]
    if not mixed_dirs:
        raise FileNotFoundError("Could not find a 'mixed' folder inside the ZIP.")
    return mixed_dirs[0]


def load_hidden_gt_from_secrets() -> pd.DataFrame:
    """
    GT is stored in Streamlit secrets and never exposed to the clinician.

    Required columns:
        mixed_name, true_label, original_file

    Optional columns:
        method

    View/label is detected automatically from uploaded ZIP/folder name.
    """
    if "gt" not in st.secrets or "tsv" not in st.secrets["gt"]:
        raise RuntimeError("Missing [gt].tsv in Streamlit secrets.")

    gt_tsv = st.secrets["gt"]["tsv"]
    gt_df = pd.read_csv(io.StringIO(gt_tsv), sep="\t")

    required_cols = {"mixed_name", "true_label", "original_file"}
    if not required_cols.issubset(gt_df.columns):
        raise ValueError(
            f"GT in secrets must contain columns: {sorted(required_cols)}"
        )

    gt_df = gt_df.copy()
    gt_df["mixed_name"] = gt_df["mixed_name"].astype(str).str.strip()
    gt_df["true_label"] = gt_df["true_label"].astype(str).str.lower().str.strip()
    gt_df["original_file"] = gt_df["original_file"].astype(str).str.strip()

    if "method" not in gt_df.columns:
        gt_df["method"] = gt_df["true_label"]
    else:
        gt_df["method"] = gt_df["method"].astype(str).str.strip()

    return gt_df


def find_media_file_by_stem(mixed_dir: Path, mixed_name: str, evaluation_type: str) -> Path:
    """
    Same GT works for frames/videos:
      sample_0000.png -> sample_0000.png in frame mode
      sample_0000.png -> sample_0000.mp4 in video mode
    """
    stem = Path(str(mixed_name)).stem
    allowed_exts = IMAGE_EXTS if evaluation_type == "frames" else VIDEO_EXTS

    matches = [
        p for p in mixed_dir.iterdir()
        if p.is_file()
        and p.stem == stem
        and p.suffix.lower() in allowed_exts
    ]

    if len(matches) == 0:
        raise FileNotFoundError(
            f"Could not find media for {mixed_name} in {evaluation_type} mode. "
            f"Expected stem: {stem}"
        )

    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple files found for stem {stem}: {[m.name for m in matches]}"
        )

    return matches[0]


def load_dataset(
    mixed_dir: Path,
    evaluation_type: str,
    uploaded_zip_name: str = "",
) -> pd.DataFrame:
    gt_df = load_hidden_gt_from_secrets()

    detected_view = detect_view_from_name(uploaded_zip_name)
    if detected_view == "unknown":
        detected_view = detect_view_from_name(str(mixed_dir))

    gt_df["label"] = detected_view

    media_paths = []
    displayed_files = []

    for mixed_name in gt_df["mixed_name"]:
        media_path = find_media_file_by_stem(
            mixed_dir=mixed_dir,
            mixed_name=mixed_name,
            evaluation_type=evaluation_type,
        )
        media_paths.append(str(media_path))
        displayed_files.append(media_path.name)

    gt_df["media_path"] = media_paths
    gt_df["displayed_file"] = displayed_files

    rng = random.Random(int(st.session_state.seed))
    indices = list(gt_df.index)
    rng.shuffle(indices)
    gt_df = gt_df.loc[indices].reset_index(drop=True)

    st.session_state.detected_view = detected_view

    return gt_df


def responses_to_df() -> pd.DataFrame:
    if not st.session_state.responses:
        return pd.DataFrame(
            columns=[
                "session_uid",
                "reader_id",
                "reader_name",
                "evaluation_type",
                "detected_view",
                "sample_idx",
                "mixed_name",
                "displayed_file",
                "original_file",
                "label",
                "method",
                "prediction",
                "timestamp",
                "true_label",
                "correct",
            ]
        )

    return pd.DataFrame(st.session_state.responses)


def save_session_csv() -> Path:
    safe_reader = (st.session_state.reader_id or "reader").replace(" ", "_")
    mode = st.session_state.evaluation_type
    view = st.session_state.detected_view
    out = RESULTS_DIR / f"responses_{mode}_{view}_{safe_reader}_{st.session_state.session_uid}.csv"
    responses_to_df().to_csv(out, index=False)
    return out


def compute_scores(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "overall_accuracy": 0.0,
            "total": 0,
            "correct": 0,
            "by_label": pd.DataFrame(columns=["label", "n", "correct", "accuracy"]),
            "by_method": pd.DataFrame(columns=["method", "n", "correct", "accuracy"]),
        }

    total = len(df)
    correct = int(df["correct"].sum())
    overall_accuracy = correct / total if total else 0.0

    by_label = (
        df.groupby("label", dropna=False)
        .agg(n=("label", "size"), correct=("correct", "sum"))
        .reset_index()
    )
    by_label["accuracy"] = by_label["correct"] / by_label["n"]

    by_method = (
        df.groupby("method", dropna=False)
        .agg(n=("method", "size"), correct=("correct", "sum"))
        .reset_index()
    )
    by_method["accuracy"] = by_method["correct"] / by_method["n"]

    return {
        "overall_accuracy": overall_accuracy,
        "total": total,
        "correct": correct,
        "by_label": by_label,
        "by_method": by_method,
    }


def send_email_with_csv(csv_path: Path, scores: dict):
    smtp_host = st.secrets["smtp"]["host"]
    smtp_port = int(st.secrets["smtp"]["port"])
    smtp_user = st.secrets["smtp"]["username"]
    smtp_password = st.secrets["smtp"]["password"]
    sender_email = st.secrets["smtp"]["sender_email"]

    msg = EmailMessage()
    msg["From"] = sender_email
    msg["To"] = RESULTS_EMAIL
    msg["Subject"] = (
        f"Clinician Study Results - "
        f"{st.session_state.evaluation_type} - "
        f"{st.session_state.detected_view} - "
        f"{st.session_state.reader_id} - "
        f"{st.session_state.session_uid}"
    )

    body = (
        f"Evaluation type: {st.session_state.evaluation_type}\n"
        f"Detected view: {st.session_state.detected_view}\n"
        f"Reader ID: {st.session_state.reader_id}\n"
        f"Reader name: {st.session_state.reader_name}\n"
        f"Session UID: {st.session_state.session_uid}\n"
        f"Answered: {scores['total']}\n"
        f"Correct: {scores['correct']}\n"
        f"Accuracy: {scores['overall_accuracy']:.4f}\n"
        f"Notes: {st.session_state.notes}\n\n"
        f"Accuracy by view/label:\n"
    )

    if not scores["by_label"].empty:
        for _, row in scores["by_label"].iterrows():
            body += (
                f"- {row['label']}: "
                f"{int(row['correct'])}/{int(row['n'])} "
                f"({row['accuracy']:.4f})\n"
            )

    body += "\nAccuracy by method:\n"
    if not scores["by_method"].empty:
        for _, row in scores["by_method"].iterrows():
            body += (
                f"- {row['method']}: "
                f"{int(row['correct'])}/{int(row['n'])} "
                f"({row['accuracy']:.4f})\n"
            )

    msg.set_content(body)

    with open(csv_path, "rb") as f:
        csv_data = f.read()

    msg.add_attachment(
        csv_data,
        maintype="text",
        subtype="csv",
        filename=csv_path.name,
    )

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)


def submit_results():
    final_df = responses_to_df()
    scores = compute_scores(final_df)
    csv_path = save_session_csv()
    send_email_with_csv(csv_path, scores)
    st.session_state.submitted = True


def record_answer(prediction: str):
    df = st.session_state.dataset
    idx = st.session_state.current_idx
    row = df.iloc[idx]

    true_label = str(row["true_label"]).strip().lower()
    correct = prediction == true_label

    response = {
        "session_uid": st.session_state.session_uid,
        "reader_id": st.session_state.reader_id,
        "reader_name": st.session_state.reader_name,
        "evaluation_type": st.session_state.evaluation_type,
        "detected_view": st.session_state.detected_view,
        "sample_idx": idx,
        "mixed_name": row["mixed_name"],
        "displayed_file": row["displayed_file"],
        "original_file": row["original_file"],
        "label": row["label"],
        "method": row["method"],
        "prediction": prediction,
        "timestamp": datetime.now().isoformat(),
        "true_label": true_label,
        "correct": bool(correct),
    }

    st.session_state.responses.append(response)
    st.session_state.current_idx += 1


st.set_page_config(page_title=APP_TITLE, layout="wide")
init_state()

st.title(APP_TITLE)
st.caption("Upload one ZIP containing only the mixed/ folder.")

with st.sidebar:
    st.header("Reader")

    st.session_state.reader_id = st.text_input(
        "Reader ID",
        value=st.session_state.reader_id,
        placeholder="reader_01",
    )

    st.session_state.reader_name = st.text_input(
        "Reader name",
        value=st.session_state.reader_name,
        placeholder="Dr. Name",
    )

    st.caption(f"Automatic session seed: {st.session_state.seed}")
    st.caption(f"Detected view: {st.session_state.detected_view}")
    st.caption(f"Results destination: {RESULTS_EMAIL}")

    st.markdown("---")

    if st.button("Reset app"):
        reset_session()
        st.rerun()


st.subheader("0. Select evaluation type")

st.session_state.evaluation_type = st.radio(
    "Evaluation type",
    options=["frames", "videos"],
    format_func=lambda x: "Frames evaluation" if x == "frames" else "Video evaluation",
    horizontal=True,
    disabled=st.session_state.setup_done,
)

st.subheader("1. Upload study package")

uploaded_zip = st.file_uploader(
    "Upload one ZIP with the mixed/ folder only",
    type=["zip"],
)

if not st.session_state.setup_done:
    if st.button("Load uploaded ZIP", type="primary"):
        if not uploaded_zip:
            st.error("Please upload the ZIP file.")
            st.stop()

        try:
            st.session_state.uploaded_zip_name = uploaded_zip.name

            temp_dir = extract_zip_to_temp(uploaded_zip)
            mixed_dir = find_mixed_dir(temp_dir)

            dataset = load_dataset(
                mixed_dir=mixed_dir,
                evaluation_type=st.session_state.evaluation_type,
                uploaded_zip_name=uploaded_zip.name,
            )

            st.session_state.working_dir = str(temp_dir)
            st.session_state.dataset = dataset
            st.session_state.setup_done = True

            st.success(
                f"Loaded {len(dataset)} samples. "
                f"Detected view: {st.session_state.detected_view}"
            )
            st.rerun()

        except Exception as e:
            st.error(f"Failed to load uploaded ZIP: {e}")
            st.stop()

if not st.session_state.setup_done:
    st.info("Choose the evaluation type, upload the ZIP, then click 'Load uploaded ZIP'.")
    st.stop()


df = st.session_state.dataset
n_total = len(df)
answered = len(st.session_state.responses)

top1, top2, top3 = st.columns([1, 1, 1])
top1.metric("Total samples", n_total)
top2.metric("Answered", answered)
top3.metric("Remaining", n_total - answered)

st.progress(answered / n_total if n_total else 0.0)

if not st.session_state.started:
    st.subheader("2. Start session")

    if st.button(
        "Start classification",
        type="primary",
        disabled=not st.session_state.reader_id.strip(),
    ):
        st.session_state.started = True
        st.rerun()

    st.stop()


idx = st.session_state.current_idx

if idx < n_total:
    row = df.iloc[idx]
    media_path = Path(row["media_path"])

    left, right = st.columns([3, 1])

    with left:
        st.subheader(f"Sample {idx + 1} / {n_total}")
        st.markdown("<div style='height:120px'></div>", unsafe_allow_html=True)

        if st.session_state.evaluation_type == "frames":
            image = Image.open(media_path)
            st.image(image, width=300)
        else:
            st.video(
                str(media_path),
                format="video/mp4",
                start_time=0,
                autoplay=False,
                loop=False,
                muted=True,
            )

    with right:
        st.subheader("Classification")
        st.write(f"Reader ID: **{st.session_state.reader_id}**")

        if st.session_state.reader_name.strip():
            st.write(f"Reader name: **{st.session_state.reader_name}**")

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
            disabled=answered == 0 or st.session_state.submitted,
        ):
            try:
                submit_results()
                st.success(f"Results sent to {RESULTS_EMAIL}")
            except Exception as e:
                st.error(f"Could not send email: {e}")

    st.stop()


st.success("Session complete.")

st.session_state.notes = st.text_area(
    "Optional notes",
    value=st.session_state.notes,
)

bottom1, bottom2 = st.columns([2, 1])

with bottom1:
    if st.button(
        "Submit",
        type="primary",
        use_container_width=True,
        disabled=answered == 0 or st.session_state.submitted,
    ):
        try:
            submit_results()
            st.success(f"Results sent to {RESULTS_EMAIL}")
        except Exception as e:
            st.error(f"Could not send email: {e}")

with bottom2:
    if st.button("Start new session", use_container_width=True):
        reset_session()
        st.rerun()
