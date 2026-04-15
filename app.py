from __future__ import annotations

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


def extract_zip_to_temp(zip_file) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="study_upload_"))
    zip_path = temp_dir / "study_package.zip"

    with open(zip_path, "wb") as f:
        f.write(zip_file.getbuffer())

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(temp_dir)

    return temp_dir


def find_required_files(base_dir: Path):
    mixed_dirs = [p for p in base_dir.rglob("mixed") if p.is_dir()]
    mixed_label_files = [p for p in base_dir.rglob("mixed_labels.txt") if p.is_file()]
    metadata_files = [p for p in base_dir.rglob("metadata_all.txt") if p.is_file()]

    if not mixed_dirs:
        raise FileNotFoundError("Could not find a 'mixed' folder inside the ZIP.")
    if not mixed_label_files:
        raise FileNotFoundError("Could not find 'mixed_labels.txt' inside the ZIP.")
    if not metadata_files:
        raise FileNotFoundError("Could not find 'metadata_all.txt' inside the ZIP.")

    return mixed_dirs[0], mixed_label_files[0], metadata_files[0]


def load_dataset(mixed_dir: Path, mixed_labels_path: Path, metadata_path: Path) -> pd.DataFrame:
    mixed_df = pd.read_csv(mixed_labels_path, sep="\t")
    meta_df = pd.read_csv(metadata_path, sep="\t")

    required_mixed = {"mixed_name", "true_label", "original_file"}
    required_meta = {"saved_path", "group", "subtype", "original_patient", "source_frame"}

    if not required_mixed.issubset(mixed_df.columns):
        raise ValueError(f"mixed_labels.txt must contain columns: {sorted(required_mixed)}")
    if not required_meta.issubset(meta_df.columns):
        raise ValueError(f"metadata_all.txt must contain columns: {sorted(required_meta)}")

    meta_df = meta_df.copy()
    meta_df["original_file"] = meta_df["saved_path"].apply(lambda x: Path(str(x)).name)

    merged = mixed_df.merge(meta_df, on="original_file", how="left")
    merged["image_path"] = merged["mixed_name"].apply(lambda x: str(mixed_dir / str(x)))

    missing = merged[~merged["image_path"].apply(lambda p: Path(p).exists())]
    if len(missing) > 0:
        raise FileNotFoundError(f"Missing image file example: {missing.iloc[0]['image_path']}")

    rng = random.Random(int(st.session_state.seed))
    indices = list(merged.index)
    rng.shuffle(indices)
    merged = merged.loc[indices].reset_index(drop=True)

    return merged


def responses_to_df() -> pd.DataFrame:
    if not st.session_state.responses:
        return pd.DataFrame(columns=[
            "session_uid",
            "reader_id",
            "reader_name",
            "sample_idx",
            "mixed_name",
            "prediction",
            "timestamp",
            "true_label",
            "subtype",
            "correct",
        ])
    return pd.DataFrame(st.session_state.responses)


def save_session_csv() -> Path:
    safe_reader = (st.session_state.reader_id or "reader").replace(" ", "_")
    out = RESULTS_DIR / f"responses_{safe_reader}_{st.session_state.session_uid}.csv"
    responses_to_df().to_csv(out, index=False)
    return out


def compute_scores(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "overall_accuracy": 0.0,
            "total": 0,
            "correct": 0,
        }

    total = len(df)
    correct = int(df["correct"].sum())
    overall_accuracy = correct / total if total else 0.0

    return {
        "overall_accuracy": overall_accuracy,
        "total": total,
        "correct": correct,
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
        f"{st.session_state.reader_id} - {st.session_state.session_uid}"
    )
    msg.set_content(
        f"Reader ID: {st.session_state.reader_id}\n"
        f"Reader name: {st.session_state.reader_name}\n"
        f"Session UID: {st.session_state.session_uid}\n"
        f"Total: {scores['total']}\n"
        f"Correct: {scores['correct']}\n"
        f"Accuracy: {scores['overall_accuracy']:.4f}\n"
        f"Notes: {st.session_state.notes}\n"
    )

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


def record_answer(prediction: str):
    df = st.session_state.dataset
    idx = st.session_state.current_idx
    row = df.iloc[idx]

    true_label = str(row["true_label"]).strip().lower()
    subtype = row.get("subtype", "unknown")
    correct = prediction == true_label

    response = {
        "session_uid": st.session_state.session_uid,
        "reader_id": st.session_state.reader_id,
        "reader_name": st.session_state.reader_name,
        "sample_idx": idx,
        "mixed_name": row["mixed_name"],
        "prediction": prediction,
        "timestamp": datetime.now().isoformat(),
        "true_label": true_label,
        "subtype": subtype,
        "correct": bool(correct),
    }

    st.session_state.responses.append(response)
    st.session_state.current_idx += 1


st.set_page_config(page_title=APP_TITLE, layout="wide")
init_state()

st.title(APP_TITLE)
st.caption("Upload one ZIP containing mixed/, mixed_labels.txt, and metadata_all.txt.")

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
    st.caption(f"Results destination: {RESULTS_EMAIL}")

    st.markdown("---")
    if st.button("Reset app"):
        reset_session()
        st.rerun()

st.subheader("1. Upload study package")

uploaded_zip = st.file_uploader(
    "Upload one ZIP with mixed/, mixed_labels.txt, and metadata_all.txt",
    type=["zip"],
)

if not st.session_state.setup_done:
    if st.button("Load uploaded ZIP", type="primary"):
        if not uploaded_zip:
            st.error("Please upload the ZIP file.")
            st.stop()

        try:
            temp_dir = extract_zip_to_temp(uploaded_zip)
            mixed_dir, mixed_labels_path, metadata_path = find_required_files(temp_dir)
            dataset = load_dataset(mixed_dir, mixed_labels_path, metadata_path)

            st.session_state.working_dir = str(temp_dir)
            st.session_state.dataset = dataset
            st.session_state.setup_done = True
            st.success(f"Loaded {len(dataset)} samples.")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to load uploaded ZIP: {e}")
            st.stop()

if not st.session_state.setup_done:
    st.info("Upload the ZIP and click 'Load uploaded ZIP'.")
    st.stop()

df = st.session_state.dataset
n_total = len(df)
answered = len(st.session_state.responses)

c1, c2 = st.columns(2)
c1.metric("Total samples", n_total)
c2.metric("Answered", answered)

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
    image_path = Path(row["image_path"])

    left, right = st.columns([3, 1])

    with left:
        st.subheader(f"Sample {idx + 1} / {n_total}")
        st.markdown("<div style='height:120px'></div>", unsafe_allow_html=True)
        image = Image.open(image_path)
        st.image(image, width=300)

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

    st.stop()

st.success("Session complete.")
st.write("Press Submit to send the results.")

st.session_state.notes = st.text_area("Optional notes", value=st.session_state.notes)

final_df = responses_to_df()
scores = compute_scores(final_df)
csv_path = save_session_csv()

if st.button("Submit", type="primary", disabled=st.session_state.submitted):
    try:
        send_email_with_csv(csv_path, scores)
        st.session_state.submitted = True
        st.success(f"Results sent to {RESULTS_EMAIL}")
    except Exception as e:
        st.error(f"Could not send email: {e}")

if st.button("Start new session"):
    reset_session()
    st.rerun()
