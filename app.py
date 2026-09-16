#!/usr/bin/env python3
"""Blinded clinician study: ZIP upload + answer keys stored as CSV text in Streamlit Secrets.

Run: streamlit run app.py
Expected Secrets:
[study]
ed_labels_csv = '''<ENTIRE ED CSV>'''
video_labels_csv = '''<ENTIRE VIDEO CSV>'''
results_dir = "results"
[smtp]
host = "..."
port = 587
username = "..."
password = "..."
sender_email = "..."
recipient_email = "..."

Clinicians upload ONLY the appropriate media ZIP; no labels inside the archive.
"""
from __future__ import annotations

import csv
import io
import logging
import secrets
import smtplib
import ssl
import tempfile
import zipfile
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path

import pandas as pd
import streamlit as st

APP_TITLE = "Echocardiography realism study"
DISPLAY_SIZE = 192  # Change to 128 if you prefer smaller images and videos.
REQUIRED = {"sample_id", "file", "view", "source", "true_label", "original_acquisition"}
RESPONSE_COLUMNS = (
    "session_id", "reader_id", "modality", "sample_number", "sample_id",
    "view", "prediction", "true_label", "source", "correct",
    "original_acquisition", "timestamp_utc", "notes",
)
SCORE_COLUMNS = ("group", "category", "total", "correct", "accuracy")
MAX_UNCOMPRESSED_BYTES = 2 * 1024**3
MAX_MEMBERS = 2000


def load_private_manifest(modality: str) -> pd.DataFrame:
    key = "ed_labels_csv" if modality == "ED images" else "video_labels_csv"
    raw = str(st.secrets["study"][key])
    df = pd.read_csv(io.StringIO(raw), keep_default_na=False, dtype=str)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Private {modality} CSV is missing columns: {sorted(missing)}")
    if df.empty or df["file"].duplicated().any() or df["sample_id"].duplicated().any():
        raise ValueError("Private manifest must be nonempty with unique filenames and sample IDs")
    expected_ext = ".png" if modality == "ED images" else ".mp4"
    for name in df["file"]:
        if not name or Path(name).name != name or Path(name).suffix.lower() != expected_ext:
            raise ValueError("Unsafe or unexpected media filename in private manifest")
    if not set(df["true_label"]).issubset({"real", "fake"}):
        raise ValueError("true_label must be real/fake")
    if not set(df["source"]).issubset({"stage1", "stage2", "real"}):
        raise ValueError("source must be stage1/stage2/real")
    if any((src == "real") != (label == "real") for src, label in zip(df["source"], df["true_label"])):
        raise ValueError("Private manifest source/label mismatch")
    return df.reset_index(drop=True)


def unpack_verified_zip(upload, manifest: pd.DataFrame, modality: str, destination: Path) -> None:
    """Extract only expected names, rejecting extra files, duplicates, traversal and ZIP bombs."""
    expected = set(manifest["file"])
    found = {}
    try:
        with zipfile.ZipFile(upload) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_MEMBERS:
                raise ValueError("ZIP contains too many entries")
            total = 0
            for info in infos:
                if info.is_dir():
                    continue
                path = Path(info.filename.replace("\\", "/"))
                if path.is_absolute() or ".." in path.parts or not path.parts:
                    raise ValueError("Unsafe ZIP entry")
                # Prevent including hidden metadata, CSV answer keys, or other extras.
                name = path.name
                if name not in expected:
                    raise ValueError("ZIP contains files that do not belong to this experiment")
                if name in found:
                    raise ValueError("ZIP contains a duplicate media filename")
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("ZIP symlinks are not permitted")
                total += info.file_size
                if total > MAX_UNCOMPRESSED_BYTES:
                    raise ValueError("Uncompressed ZIP is too large")
                found[name] = info
            if set(found) != expected:
                raise ValueError(
                    f"Incorrect ZIP for {modality}: expected {len(expected)} media files, "
                    f"found {len(found)} matching files. Check that you uploaded the right experiment."
                )
            destination.mkdir(parents=True, exist_ok=True)
            for name, info in found.items():
                # Never preserve ZIP paths; only manifest-approved basenames.
                with archive.open(info) as src, (destination / name).open("wb") as dst:
                    remaining = info.file_size
                    while remaining:
                        chunk = src.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise ValueError("Truncated media file in ZIP")
                        dst.write(chunk)
                        remaining -= len(chunk)
    except zipfile.BadZipFile as exc:
        raise ValueError("Invalid or corrupted ZIP file") from exc


def results_folder() -> Path:
    path = Path(str(st.secrets["study"].get("results_dir", "results"))).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path: Path, columns, records) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def response_path() -> Path:
    return results_folder() / f"responses_{st.session_state.session_id}.csv"


def save_responses() -> Path:
    path = response_path()
    write_csv(path, RESPONSE_COLUMNS, st.session_state.responses)
    return path


def score_rows(responses: list[dict]) -> list[dict]:
    df = pd.DataFrame(responses)
    rows = []
    for group, category, data in [("overall", "all", df)]:
        count = len(data)
        correct = int(data["correct"].sum()) if count else 0
        rows.append(dict(group=group, category=category, total=count,
                         correct=correct, accuracy=correct / count if count else 0))
    for field in ("source", "view"):
        for name, data in df.groupby(field):
            count = len(data)
            correct = int(data["correct"].sum())
            rows.append(dict(group=field, category=name, total=count,
                             correct=correct, accuracy=correct / count))
    return rows


def save_scores() -> tuple[Path, list[dict]]:
    scores = score_rows(st.session_state.responses)
    path = results_folder() / f"scores_{st.session_state.session_id}.csv"
    write_csv(path, SCORE_COLUMNS, scores)
    return path, scores


def email_results(responses: Path, scores_file: Path, scores: list[dict]) -> None:
    smtp = st.secrets["smtp"]
    message = EmailMessage()
    message["From"] = str(smtp["sender_email"])
    message["To"] = str(smtp["recipient_email"])
    message["Subject"] = (
        f"Clinician study results: {st.session_state.modality} / {st.session_state.session_id}"
    )
    overall = scores[0]
    message.set_content(
        f"Reader ID: {st.session_state.reader}\n"
        f"Session: {st.session_state.session_id}\n"
        f"Experiment: {st.session_state.modality}\n"
        f"Answers: {overall['total']}\n"
        f"Correct: {overall['correct']}\n"
        f"Accuracy: {overall['accuracy']:.2%}\n"
        f"Notes: {st.session_state.notes}\n\n"
        "Individual responses and accuracy by source/view are attached.\n"
    )
    for path in (responses, scores_file):
        message.add_attachment(path.read_bytes(), maintype="text", subtype="csv", filename=path.name)
    host, port = str(smtp["host"]), int(smtp["port"])
    if port == 465:
        with smtplib.SMTP_SSL(host, port, context=ssl.create_default_context(), timeout=30) as server:
            server.login(str(smtp["username"]), str(smtp["password"]))
            server.send_message(message)
    else:
        with smtplib.SMTP(host, port, timeout=30) as server:
            server.starttls(context=ssl.create_default_context())
            server.login(str(smtp["username"]), str(smtp["password"]))
            server.send_message(message)


def reset_session() -> None:
    for key in ("started", "dataset", "media_dir", "reader", "modality", "order",
                "idx", "responses", "session_id", "emailed", "notes", "temp_handle", "finalized"):
        if key == "temp_handle" and key in st.session_state:
            st.session_state[key].cleanup()
        st.session_state.pop(key, None)


st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("For each echocardiogram, classify its appearance as real or synthetic.")

if "started" not in st.session_state:
    st.session_state.started = False

if not st.session_state.started:
    with st.form("study_setup"):
        modality = st.radio("Choose experiment", ["ED images", "Videos"], horizontal=True)
        reader = st.text_input("Reader ID (pseudonym)", placeholder="clinician_01")
        upload = st.file_uploader("Upload the corresponding media ZIP", type=["zip"])
        start = st.form_submit_button("Load ZIP and start classification", type="primary")
    if start:
        if not reader.strip() or upload is None:
            st.error("Enter a reader ID and upload the media ZIP for the selected experiment.")
            st.stop()
        temp_handle = None
        try:
            manifest = load_private_manifest(modality)
            temp_handle = tempfile.TemporaryDirectory(prefix="echoclinician_")
            media_dir = Path(temp_handle.name) / "media"
            unpack_verified_zip(upload, manifest, modality, media_dir)
        except Exception as exc:
            if temp_handle is not None:
                temp_handle.cleanup()
            logging.exception("Could not initialize study")
            st.error(f"Could not load this ZIP: {exc}")
            st.stop()
        order = list(range(len(manifest)))
        secrets.SystemRandom().shuffle(order)
        st.session_state.update(dict(
            started=True, dataset=manifest, media_dir=str(media_dir),
            temp_handle=temp_handle, reader=reader.strip(), modality=modality,
            order=order, idx=0, responses=[], session_id=secrets.token_hex(12),
            emailed=False, notes="", finalized=False,
        ))
        st.rerun()
    st.stop()

count = len(st.session_state.order)
idx = st.session_state.idx
st.progress(idx / count if count else 0)
st.caption(f"{st.session_state.modality} · Sample {min(idx + 1, count)} of {count}")

if idx < count and not st.session_state.finalized:
    row = st.session_state.dataset.iloc[st.session_state.order[idx]]
    path = Path(st.session_state.media_dir) / str(row["file"])
    media_col, answer_col = st.columns([4, 2], gap="large")
    with media_col:
        if st.session_state.modality == "ED images":
            st.image(str(path), width=DISPLAY_SIZE)
        else:
            st.video(str(path), format="video/mp4", autoplay=False, loop=True, width=DISPLAY_SIZE)
    with answer_col:
        st.subheader("Classification")
        st.caption("Select an answer; submitted answers cannot be changed.")
        selection = st.radio(
            "This sample appears to be:", ["Real", "Synthetic"], index=None,
            key=f"answer_{st.session_state.session_id}_{idx}",
        )
        if st.button("Submit answer", type="primary", disabled=selection is None):
            prediction = "real" if selection == "Real" else "fake"
            st.session_state.responses.append(dict(
                session_id=st.session_state.session_id,
                reader_id=st.session_state.reader,
                modality=st.session_state.modality,
                sample_number=idx + 1,
                sample_id=str(row["sample_id"]),
                view=str(row["view"]),
                prediction=prediction,
                true_label=str(row["true_label"]),
                source=str(row["source"]),
                correct=int(prediction == row["true_label"]),
                original_acquisition=str(row["original_acquisition"]),
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                notes="",
            ))
            save_responses()
            st.session_state.idx += 1
            st.rerun()

    st.divider()
    st.caption(
        f"You can finish the experiment early. Only your {len(st.session_state.responses)} "
        "previously submitted answers will be included; the current unanswered sample is excluded."
    )
    if st.button(
        "Finish now & email current results",
        disabled=not st.session_state.responses,
        help="At least one submitted answer is required to calculate scores.",
    ):
        st.session_state.finalized = True
        st.rerun()
    st.stop()

if st.session_state.finalized and idx < count:
    st.success(
        f"You finished early: {len(st.session_state.responses)} of {count} "
        "samples were classified. Thank you for participating."
    )
else:
    st.success("Experiment complete. Thank you for participating.")
# Optional notes can be entered before automatic email: email occurs after the
# final answer at first completion (notes may be added and results resent manually).
st.session_state.notes = st.text_area("Optional comments", value=st.session_state.notes)
for item in st.session_state.responses:
    item["notes"] = st.session_state.notes
response_file = save_responses()

if not st.session_state.emailed:
    try:
        scores_file, summary = save_scores()
        email_results(response_file, scores_file, summary)
    except Exception:
        logging.exception("Automatic clinician results email failed")
        st.error("Automatic email delivery failed. Results were saved locally; retry below.")
        if st.button("Retry sending results", type="primary"):
            try:
                scores_file, summary = save_scores()
                email_results(response_file, scores_file, summary)
            except Exception:
                logging.exception("Clinician results email retry failed")
                st.error("Retry failed. Please contact the study administrator.")
            else:
                st.session_state.emailed = True
                st.rerun()
    else:
        st.session_state.emailed = True
        st.rerun()
else:
    st.success("Results have been emailed to the study administrator.")

if st.button("Start another experiment"):
    reset_session()
    st.rerun()
