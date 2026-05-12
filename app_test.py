from __future__ import annotations

import base64
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
import streamlit.components.v1 as components
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
        "detected_view_group": "unknown_group",
        "detected_view": "unknown_view",
        "final_scores": None,
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
    st.session_state.detected_view_group = "unknown_group"
    st.session_state.detected_view = "unknown_view"
    st.session_state.final_scores = None


def detect_view_info_from_name(name: str) -> tuple[str, str]:
    path_str = str(name).lower()

    view_patterns = {
        "A4C": ["a4c", "4ch", "4_ch", "4-ch"],
        "A5C": ["a5c", "5ch", "5_ch", "5-ch"],
        "A3C": ["a3c", "3ch", "3_ch", "3-ch"],
        "A2C": ["a2c", "2ch", "2_ch", "2-ch"],
        "PSAX": ["psax", "pssa"],
        "PLAX": ["plax", "psla"],
    }

    view_label = "unknown_view"
    for view, patterns in view_patterns.items():
        if any(p in path_str for p in patterns):
            view_label = view
            break

    if view_label in ["A4C", "A5C", "A3C", "A2C"]:
        view_group = "apical"
    elif view_label in ["PSAX", "PLAX"]:
        view_group = "parasternal"
    else:
        view_group = "unknown_group"

    return view_group, view_label


def group_from_view(view_label: str) -> str:
    view_label = str(view_label).strip().upper()

    if view_label in ["A4C", "A5C", "A3C", "A2C"]:
        return "apical"
    if view_label in ["PSAX", "PLAX"]:
        return "parasternal"
    if view_label == "SUBCOSTAL":
        return "subcostal"

    return "unknown_group"


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
    if "gt" not in st.secrets or "tsv" not in st.secrets["gt"]:
        raise RuntimeError("Missing [gt].tsv in Streamlit secrets.")

    gt_text = str(st.secrets["gt"]["tsv"]).strip()
    required_cols = {"mixed_name", "true_label", "original_file"}

    # Try real TSV first
    gt_df = pd.read_csv(io.StringIO(gt_text), sep="\t")

    # If pasted with spaces instead of tabs, retry with whitespace separator
    if not required_cols.issubset(gt_df.columns):
        gt_df = pd.read_csv(
            io.StringIO(gt_text),
            sep=r"\s+",
            engine="python",
        )

    if not required_cols.issubset(gt_df.columns):
        raise ValueError(
            "GT in secrets must contain at least these columns: "
            f"{sorted(required_cols)}. Detected columns: {gt_df.columns.tolist()}"
        )

    gt_df = gt_df.copy()

    for col in [
        "mixed_name",
        "true_label",
        "original_file",
        "method",
        "view_group",
        "view_label",
        "original_patient",
        "source_folder",
        "source_frame",
    ]:
        if col not in gt_df.columns:
            gt_df[col] = ""
        gt_df[col] = gt_df[col].astype(str).str.strip()

    gt_df["true_label"] = gt_df["true_label"].str.lower()

    # Match using full mixed filename, not stem
    gt_df["mixed_name_norm"] = (
        gt_df["mixed_name"]
        .astype(str)
        .str.strip()
        .apply(lambda x: Path(x).name)
        .str.lower()
    )

    gt_df["view_label"] = gt_df["view_label"].replace("", "unknown_view")
    gt_df["view_group"] = gt_df["view_group"].replace("", "unknown_group")

    gt_df["view_group"] = gt_df.apply(
        lambda r: group_from_view(r["view_label"])
        if str(r["view_group"]).strip() in ["", "unknown_group"]
        else r["view_group"],
        axis=1,
    )

    gt_df["method"] = gt_df.apply(
        lambda r: r["true_label"] if str(r["method"]).strip() == "" else r["method"],
        axis=1,
    )

    return gt_df


def load_dataset(
    mixed_dir: Path,
    evaluation_type: str,
    uploaded_zip_name: str = "",
) -> pd.DataFrame:
    gt_df = load_hidden_gt_from_secrets()

    detected_group, detected_view = detect_view_info_from_name(uploaded_zip_name)

    if detected_group == "unknown_group" or detected_view == "unknown_view":
        folder_group, folder_view = detect_view_info_from_name(str(mixed_dir))

        if detected_group == "unknown_group":
            detected_group = folder_group

        if detected_view == "unknown_view":
            detected_view = folder_view

    allowed_exts = IMAGE_EXTS if evaluation_type == "frames" else VIDEO_EXTS

    files = [
        p for p in mixed_dir.iterdir()
        if p.is_file() and p.suffix.lower() in allowed_exts
    ]

    if len(files) == 0:
        raise RuntimeError(f"No {evaluation_type} files found in {mixed_dir}")

    rows = []

    for p in sorted(files):
        mixed_name = p.name.strip().lower()
        match = gt_df[gt_df["mixed_name_norm"] == mixed_name]

        if len(match) == 0:
            continue

        row = match.iloc[0].copy()

        row["media_path"] = str(p)
        row["displayed_file"] = p.name

        if not str(row.get("view_label", "")).strip() or row["view_label"] == "unknown_view":
            row["view_label"] = detected_view

        if not str(row.get("view_group", "")).strip() or row["view_group"] == "unknown_group":
            row["view_group"] = group_from_view(row["view_label"])
            if row["view_group"] == "unknown_group":
                row["view_group"] = detected_group

        row["label"] = row["view_label"]

        rows.append(row)

    if len(rows) == 0:
        raise RuntimeError("No matching GT entries found for files.")

    df = pd.DataFrame(rows)

    rng = random.Random(int(st.session_state.seed))
    df = df.sample(frac=1, random_state=rng.randint(0, 10**6)).reset_index(drop=True)

    if len(df) > 0:
        unique_groups = sorted(df["view_group"].dropna().unique())
        unique_views = sorted(df["view_label"].dropna().unique())

        st.session_state.detected_view_group = (
            unique_groups[0] if len(unique_groups) == 1 else "mixed"
        )
        st.session_state.detected_view = (
            unique_views[0] if len(unique_views) == 1 else "mixed"
        )
    else:
        st.session_state.detected_view_group = detected_group
        st.session_state.detected_view = detected_view

    return df


def responses_to_df() -> pd.DataFrame:
    if not st.session_state.responses:
        return pd.DataFrame(
            columns=[
                "session_uid",
                "reader_id",
                "reader_name",
                "evaluation_type",
                "detected_view_group",
                "detected_view",
                "sample_idx",
                "mixed_name",
                "displayed_file",
                "original_file",
                "method",
                "view_group",
                "view_label",
                "label",
                "original_patient",
                "source_folder",
                "source_frame",
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
    group = st.session_state.detected_view_group
    view = st.session_state.detected_view

    out = RESULTS_DIR / (
        f"responses_{mode}_{group}_{view}_{safe_reader}_"
        f"{st.session_state.session_uid}.csv"
    )

    responses_to_df().to_csv(out, index=False)
    return out


def compute_scores(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "overall_accuracy": 0.0,
            "total": 0,
            "correct": 0,
            "by_group": pd.DataFrame(columns=["view_group", "n", "correct", "accuracy"]),
            "by_view": pd.DataFrame(columns=["view_label", "n", "correct", "accuracy"]),
            "by_method": pd.DataFrame(columns=["method", "n", "correct", "accuracy"]),
            "by_view_method": pd.DataFrame(
                columns=["view_label", "method", "n", "correct", "accuracy"]
            ),
        }

    df = df.copy()
    df["correct"] = df["correct"].astype(bool)

    total = len(df)
    correct = int(df["correct"].sum())
    overall_accuracy = correct / total if total else 0.0

    by_group = (
        df.groupby("view_group", dropna=False)
        .agg(n=("view_group", "size"), correct=("correct", "sum"))
        .reset_index()
    )
    by_group["accuracy"] = by_group["correct"] / by_group["n"]

    by_view = (
        df.groupby("view_label", dropna=False)
        .agg(n=("view_label", "size"), correct=("correct", "sum"))
        .reset_index()
    )
    by_view["accuracy"] = by_view["correct"] / by_view["n"]

    by_method = (
        df.groupby("method", dropna=False)
        .agg(n=("method", "size"), correct=("correct", "sum"))
        .reset_index()
    )
    by_method["accuracy"] = by_method["correct"] / by_method["n"]

    by_view_method = (
        df.groupby(["view_label", "method"], dropna=False)
        .agg(n=("correct", "size"), correct=("correct", "sum"))
        .reset_index()
    )
    by_view_method["accuracy"] = by_view_method["correct"] / by_view_method["n"]

    return {
        "overall_accuracy": overall_accuracy,
        "total": total,
        "correct": correct,
        "by_group": by_group,
        "by_view": by_view,
        "by_method": by_method,
        "by_view_method": by_view_method,
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
        f"{st.session_state.detected_view_group} - "
        f"{st.session_state.detected_view} - "
        f"{st.session_state.reader_id} - "
        f"{st.session_state.session_uid}"
    )

    body = (
        f"Evaluation type: {st.session_state.evaluation_type}\n"
        f"Detected view group: {st.session_state.detected_view_group}\n"
        f"Detected view: {st.session_state.detected_view}\n"
        f"Reader ID: {st.session_state.reader_id}\n"
        f"Reader name: {st.session_state.reader_name}\n"
        f"Session UID: {st.session_state.session_uid}\n"
        f"Answered: {scores['total']}\n"
        f"Correct: {scores['correct']}\n"
        f"Accuracy: {scores['overall_accuracy']:.4f}\n"
        f"Notes: {st.session_state.notes}\n\n"
        f"Accuracy by group:\n"
    )

    if not scores["by_group"].empty:
        for _, row in scores["by_group"].iterrows():
            body += (
                f"- {row['view_group']}: "
                f"{int(row['correct'])}/{int(row['n'])} "
                f"({row['accuracy']:.4f})\n"
            )

    body += "\nAccuracy by view:\n"
    if not scores["by_view"].empty:
        for _, row in scores["by_view"].iterrows():
            body += (
                f"- {row['view_label']}: "
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

    body += "\nAccuracy by view and method:\n"
    if not scores["by_view_method"].empty:
        for _, row in scores["by_view_method"].iterrows():
            body += (
                f"- {row['view_label']} / {row['method']}: "
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
    st.session_state.final_scores = scores
    return scores


def record_answer(prediction: str):
    df = st.session_state.dataset
    idx = st.session_state.current_idx
    row = df.iloc[idx]

    true_label = str(row.get("true_label", "")).strip().lower()
    correct = prediction == true_label

    response = {
        "session_uid": st.session_state.session_uid,
        "reader_id": st.session_state.reader_id,
        "reader_name": st.session_state.reader_name,
        "evaluation_type": st.session_state.evaluation_type,
        "detected_view_group": st.session_state.detected_view_group,
        "detected_view": st.session_state.detected_view,
        "sample_idx": idx,
        "mixed_name": str(row.get("mixed_name", "")),
        "displayed_file": str(row.get("displayed_file", "")),
        "original_file": str(row.get("original_file", "")),
        "method": str(row.get("method", "unknown")),
        "view_group": str(row.get("view_group", st.session_state.detected_view_group)),
        "view_label": str(row.get("view_label", st.session_state.detected_view)),
        "label": str(row.get("label", row.get("view_label", st.session_state.detected_view))),
        "original_patient": str(row.get("original_patient", "")),
        "source_folder": str(row.get("source_folder", "")),
        "source_frame": str(row.get("source_frame", "")),
        "prediction": prediction,
        "timestamp": datetime.now().isoformat(),
        "true_label": true_label,
        "correct": bool(correct),
    }

    st.session_state.responses.append(response)
    st.session_state.current_idx += 1


def show_media(media_path: Path):
    display_width = 220
    iframe_width = display_width + 30
    iframe_height = display_width + 70

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
                    style="width:{display_width}px; height:auto; display:block;"
                >
                    <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
                </video>
            </body>
            </html>
            """,
            width=iframe_width,
            height=iframe_height,
            scrolling=False,
        )


def show_score_summary(scores: dict):
    st.subheader("Your results")

    total = int(scores["total"])
    correct = int(scores["correct"])
    wrong = total - correct
    accuracy = scores["overall_accuracy"] * 100

    c1, c2, c3 = st.columns(3)

    c1.metric("Total accuracy", f"{accuracy:.1f}%")
    c2.metric("Correct", correct)
    c3.metric("Wrong", wrong)

    st.write(f"You got **{correct} correct** and **{wrong} wrong** out of **{total}** samples.")

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
    st.caption(f"Detected group: {st.session_state.detected_view_group}")
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
                f"Detected group: {st.session_state.detected_view_group}. "
                f"Detected view: {st.session_state.detected_view}."
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

if st.session_state.submitted and st.session_state.final_scores is not None:
    show_score_summary(st.session_state.final_scores)
    st.stop()

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

    left, spacer, right = st.columns([0.7, 0.1, 1.2])

    with left:
        st.subheader(f"Sample {idx + 1} / {n_total}")
        show_media(media_path)

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
                scores = submit_results()
                st.success(f"Results sent to {RESULTS_EMAIL}")
                show_score_summary(scores)
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
            scores = submit_results()
            st.success(f"Results sent to {RESULTS_EMAIL}")
            show_score_summary(scores)
        except Exception as e:
            st.error(f"Could not send email: {e}")

with bottom2:
    if st.button("Start new session", use_container_width=True):
        reset_session()
        st.rerun()
