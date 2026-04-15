from __future__ import annotations

import json
import random
import secrets
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Clinician Fake/Real Classification", layout="wide")

if not st.user.is_logged_in:
    st.title("Clinician Fake/Real Classification")
    st.write("Please sign in to access the study.")
    if st.button("Sign in"):
        st.login()
    st.stop()

ALLOWED_EMAILS = {
    "jpav.freitas@gmail.org"}

user_email = st.user.get("email", "").lower()

if user_email not in ALLOWED_EMAILS:
    st.error("You are signed in, but you are not authorized for this study.")
    if st.button("Log out"):
        st.logout()
    st.stop()

# ============================================================
# Configuration
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MIXED_DIR = DATA_DIR / "mixed"
MIXED_LABELS_PATH = DATA_DIR / "mixed_labels.txt"
METADATA_PATH = DATA_DIR / "metadata_all.txt"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

APP_TITLE = "Clinician Fake/Real Classification"


# ============================================================
# Helpers
# ============================================================
def safe_read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def load_ground_truth() -> pd.DataFrame:
    """
    Expected files:
    - data/mixed_labels.txt with columns:
        mixed_name, true_label, original_file
    - data/metadata_all.txt with columns:
        saved_path, group, subtype, original_patient, source_frame

    Joins:
    original_file <-> basename(saved_path)
    """
    mixed_df = safe_read_table(MIXED_LABELS_PATH)
    meta_df = safe_read_table(METADATA_PATH)

    required_mixed = {"mixed_name", "true_label", "original_file"}
    required_meta = {"saved_path", "group", "subtype", "original_patient", "source_frame"}

    if mixed_df.empty:
        raise FileNotFoundError(f"Missing or empty file: {MIXED_LABELS_PATH}")
    if meta_df.empty:
        raise FileNotFoundError(f"Missing or empty file: {METADATA_PATH}")

    if not required_mixed.issubset(set(mixed_df.columns)):
        raise ValueError(
            f"{MIXED_LABELS_PATH.name} must contain columns: {sorted(required_mixed)}"
        )
    if not required_meta.issubset(set(meta_df.columns)):
        raise ValueError(
            f"{METADATA_PATH.name} must contain columns: {sorted(required_meta)}"
        )

    meta_df = meta_df.copy()
    meta_df["original_file"] = meta_df["saved_path"].apply(lambda x: Path(str(x)).name)

    merged = mixed_df.merge(meta_df, on="original_file", how="left")
    merged["image_path"] = merged["mixed_name"].apply(lambda x: str(MIXED_DIR / str(x)))

    missing = merged[merged["image_path"].apply(lambda p: not Path(p).exists())]
    if len(missing) > 0:
        sample_missing = missing.iloc[0]["image_path"]
        raise FileNotFoundError(
            f"Some mixed images are missing. Example missing file: {sample_missing}"
        )

    return merged


def shuffle_dataframe(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    indices = list(df.index)
    rng.shuffle(indices)
    return df.loc[indices].reset_index(drop=True)


def responses_to_dataframe(responses: list[dict]) -> pd.DataFrame:
    if not responses:
        return pd.DataFrame(
            columns=[
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
            ]
        )
    return pd.DataFrame(responses)


def compute_scores(df_resp: pd.DataFrame) -> dict:
    if df_resp.empty:
        return {
            "overall_accuracy": 0.0,
            "total": 0,
            "correct": 0,
            "by_label": pd.DataFrame(columns=["true_label", "n", "correct", "accuracy"]),
            "by_subtype": pd.DataFrame(columns=["subtype", "n", "correct", "accuracy"]),
        }

    total = len(df_resp)
    correct = int(df_resp["correct"].sum())
    overall_accuracy = correct / total if total > 0 else 0.0

    by_label = (
        df_resp.groupby("true_label", dropna=False)
        .agg(n=("true_label", "size"), correct=("correct", "sum"))
        .reset_index()
    )
    by_label["accuracy"] = by_label["correct"] / by_label["n"]

    by_subtype = (
        df_resp.groupby("subtype", dropna=False)
        .agg(n=("subtype", "size"), correct=("correct", "sum"))
        .reset_index()
    )
    by_subtype["accuracy"] = by_subtype["correct"] / by_subtype["n"]

    return {
        "overall_accuracy": overall_accuracy,
        "total": total,
        "correct": correct,
        "by_label": by_label,
        "by_subtype": by_subtype,
    }


def append_single_response_to_master_csv(response: dict) -> Path:
    output_path = RESULTS_DIR / "responses_master.csv"
    write_header = not output_path.exists()
    pd.DataFrame([response]).to_csv(output_path, mode="a", header=write_header, index=False)
    return output_path


def save_live_session_snapshot(reader_id: str, session_uid: str, responses: list[dict]) -> Path:
    safe_reader = (reader_id or "reader").replace(" ", "_")
    csv_path = RESULTS_DIR / f"responses_{safe_reader}_{session_uid}.csv"
    df_resp = responses_to_dataframe(responses)
    df_resp.to_csv(csv_path, index=False)
    return csv_path


def save_session_scores(
    df_resp: pd.DataFrame,
    scores: dict,
    reader_id: str,
    reader_name: str,
    session_uid: str,
    notes: str,
) -> Path:
    safe_reader = (reader_id or "reader").replace(" ", "_")
    json_path = RESULTS_DIR / f"scores_{safe_reader}_{session_uid}.json"

    payload = {
        "reader_id": reader_id,
        "reader_name": reader_name,
        "session_uid": session_uid,
        "notes": notes,
        "generated_at": datetime.now().isoformat(),
        "overall_accuracy": scores["overall_accuracy"],
        "total": scores["total"],
        "correct": scores["correct"],
        "by_label": scores["by_label"].to_dict(orient="records"),
        "by_subtype": scores["by_subtype"].to_dict(orient="records"),
    }

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return json_path


def get_session_state_defaults() -> None:
    defaults = {
        "dataset": None,
        "shuffled_dataset": None,
        "reader_id": "",
        "reader_name": "",
        "notes": "",
        "seed": secrets.randbelow(10**9),
        "started": False,
        "current_idx": 0,
        "responses": [],
        "session_uid": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "saved_scores": False,
        "saved_master_path": "",
        "saved_csv_path": "",
        "saved_json_path": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_session() -> None:
    st.session_state.started = False
    st.session_state.current_idx = 0
    st.session_state.responses = []
    st.session_state.notes = ""
    st.session_state.seed = secrets.randbelow(10**9)
    st.session_state.session_uid = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state.shuffled_dataset = None
    st.session_state.saved_scores = False
    st.session_state.saved_master_path = ""
    st.session_state.saved_csv_path = ""
    st.session_state.saved_json_path = ""


def record_answer(prediction: str) -> None:
    df = st.session_state.shuffled_dataset
    idx = st.session_state.current_idx

    if df is None or idx >= len(df):
        return

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

    master_path = append_single_response_to_master_csv(response)
    session_csv_path = save_live_session_snapshot(
        st.session_state.reader_id,
        st.session_state.session_uid,
        st.session_state.responses,
    )

    st.session_state.saved_master_path = str(master_path)
    st.session_state.saved_csv_path = str(session_csv_path)
    st.session_state.current_idx += 1


# ============================================================
# UI
# ============================================================
st.set_page_config(page_title=APP_TITLE, layout="wide")
get_session_state_defaults()

st.title(APP_TITLE)
st.caption("Classify each image as real or fake. Responses are saved automatically after every click.")

with st.sidebar:
    st.header("Setup")
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
    st.caption(f"Automatic session shuffle seed: {st.session_state.seed}")

    st.markdown("---")
    st.write("Expected folder structure:")
    st.code(
        "doctor_app/\n"
        "├── app.py\n"
        "├── data/\n"
        "│   ├── mixed/\n"
        "│   ├── mixed_labels.txt\n"
        "│   └── metadata_all.txt\n"
        "└── results/"
    )

    if st.button("Reload data"):
        st.session_state.dataset = None
        st.session_state.shuffled_dataset = None
        st.rerun()

    if st.button("Reset session"):
        reset_session()
        st.rerun()


if st.session_state.dataset is None:
    try:
        st.session_state.dataset = load_ground_truth()
    except Exception as e:
        st.error(f"Could not load dataset: {e}")
        st.stop()

if st.session_state.shuffled_dataset is None and st.session_state.dataset is not None:
    st.session_state.shuffled_dataset = shuffle_dataframe(
        st.session_state.dataset,
        int(st.session_state.seed),
    )

df_all = st.session_state.shuffled_dataset
n_total = len(df_all)
df_resp = responses_to_dataframe(st.session_state.responses)

col1, col2 = st.columns(2)
col1.metric("Total samples", n_total)
col2.metric("Answered", len(df_resp))

progress = 0.0 if n_total == 0 else len(df_resp) / n_total
st.progress(progress)

if not st.session_state.started:
    st.subheader("Start session")
    st.write("Fill the Reader ID and start the classification session.")

    if n_total == 0:
        st.warning("No samples found.")
        st.stop()

    if st.button("Start classification", type="primary", disabled=not st.session_state.reader_id.strip()):
        st.session_state.started = True
        st.rerun()

    st.stop()

current_idx = st.session_state.current_idx

if current_idx < n_total:
    row = df_all.iloc[current_idx]
    image_path = Path(row["image_path"])

    left, right = st.columns([3, 1])

    with left:
        st.subheader(f"Sample {current_idx + 1} / {n_total}")

        st.markdown("<div style='height:120px'></div>", unsafe_allow_html=True)

        image = Image.open(image_path)
        st.image(image, width=200)

    with right:
        st.subheader("Classification")
        st.write(f"Reader ID: **{st.session_state.reader_id}**")
        if st.session_state.reader_name.strip():
            st.write(f"Reader name: **{st.session_state.reader_name}**")

        st.markdown("Choose one option:")
        real_col, fake_col = st.columns(2)

        with real_col:
            if st.button("Real", use_container_width=True, type="primary"):
                record_answer("real")
                st.rerun()

        with fake_col:
            if st.button("Fake", use_container_width=True):
                record_answer("fake")
                st.rerun()

    st.stop()


# ============================================================
# End of session
# ============================================================
final_df = responses_to_dataframe(st.session_state.responses)
scores = compute_scores(final_df)

if not st.session_state.saved_scores:
    json_path = save_session_scores(
        final_df,
        scores,
        st.session_state.reader_id,
        st.session_state.reader_name,
        st.session_state.session_uid,
        st.session_state.notes,
    )
    st.session_state.saved_json_path = str(json_path)
    st.session_state.saved_scores = True

st.success("Session complete.")
st.write("Thank you. Your responses have been saved.")

st.session_state.notes = st.text_area("Optional notes", value=st.session_state.notes)

if st.button("Save notes update"):
    json_path = save_session_scores(
        final_df,
        scores,
        st.session_state.reader_id,
        st.session_state.reader_name,
        st.session_state.session_uid,
        st.session_state.notes,
    )
    st.session_state.saved_json_path = str(json_path)
    st.success("Notes updated.")

st.info(
    "Saved files:\n\n"
    f"- {st.session_state.saved_master_path or str(RESULTS_DIR / 'responses_master.csv')}\n"
    f"- {st.session_state.saved_csv_path}\n"
    f"- {st.session_state.saved_json_path}"
)

if st.button("Start new session", use_container_width=True):
    reset_session()
    st.rerun()
