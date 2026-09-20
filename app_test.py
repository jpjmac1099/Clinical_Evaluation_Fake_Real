#!/usr/bin/env python3
"""Clinician classification app: repository-hosted media ZIPs, on-page scores.

Run: streamlit run app_test.py

Keep media ZIPs in the GitHub repository (or set their repo-relative paths below).
Keep the answer-key CSV TEXT in Streamlit Secrets, NOT the GitHub repository:

[study]
ed_labels_csv = '''sample_id,file,view,source,true_label,original_acquisition\n...'''
video_labels_csv = '''sample_id,file,view,source,true_label,original_acquisition\n...'''
# Optional overrides, relative to app_test.py:
# ed_zip_path = "ed_images.zip"
# video_zip_path = "videos.zip"

No media upload or SMTP/email is used. Scores appear only upon completion/early finish.
"""
from __future__ import annotations

import csv
import io
import logging
import secrets
import stat
import subprocess
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import imageio_ffmpeg
import pandas as pd
import streamlit as st

APP_TITLE = "Echocardiography realism study"
DISPLAY_SIZE = 192
REPOSITORY_DIR = Path(__file__).resolve().parent
# Edit these defaults if your GitHub ZIPs are elsewhere in the repository.
ED_ZIP_PATH = "ed_images.zip"
VIDEO_ZIP_PATH = "videos.zip"
MAX_MEMBERS = 2000
MAX_TOTAL_UNCOMPRESSED = 2 * 1024**3
MAX_SINGLE_MEDIA = 100 * 1024**2
REQUIRED = {"sample_id", "file", "view", "source", "true_label", "original_acquisition"}
RESPONSE_COLUMNS = (
    "session_id", "reader_id", "modality", "sample_number", "sample_id", "view",
    "prediction", "true_label", "source", "correct", "original_acquisition",
    "timestamp_utc", "notes",
)
SCORE_COLUMNS = ("group", "category", "total", "correct", "accuracy")


def study_settings():
    return st.secrets["study"]


def media_zip_path(modality: str) -> Path:
    setting = "ed_zip_path" if modality == "ED images" else "video_zip_path"
    default = ED_ZIP_PATH if modality == "ED images" else VIDEO_ZIP_PATH
    candidate = (REPOSITORY_DIR / str(study_settings().get(setting, default))).resolve()
    if not candidate.is_relative_to(REPOSITORY_DIR):
        raise ValueError("Media ZIP path must remain inside the GitHub repository")
    if not candidate.is_file():
        raise FileNotFoundError(
            f"Repository ZIP not found: {candidate.relative_to(REPOSITORY_DIR)}. "
            "Set the correct path in ED_ZIP_PATH / VIDEO_ZIP_PATH or [study] Secrets."
        )
    return candidate


def load_private_manifest(modality: str) -> pd.DataFrame:
    key = "ed_labels_csv" if modality == "ED images" else "video_labels_csv"
    raw = str(study_settings()[key])
    df = pd.read_csv(io.StringIO(raw), keep_default_na=False, dtype=str)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Private {modality} CSV is missing columns: {sorted(missing)}")
    if df.empty or df["file"].duplicated().any() or df["sample_id"].duplicated().any():
        raise ValueError("Manifest must be nonempty, with unique filenames and sample IDs")
    extension = ".png" if modality == "ED images" else ".mp4"
    for name in df["file"]:
        if not name or Path(name).name != name or Path(name).suffix.lower() != extension:
            raise ValueError("Manifest contains unsafe or unexpected media filename")
    if not set(df["true_label"]).issubset({"real", "fake"}):
        raise ValueError("true_label must be real or fake")
    if not set(df["source"]).issubset({"stage1", "stage2", "real"}):
        raise ValueError("source must be stage1, stage2, or real")
    if any((src == "real") != (label == "real") for src, label in zip(df["source"], df["true_label"])):
        raise ValueError("Manifest source/true_label mismatch")
    return df.reset_index(drop=True)


def inspect_zip(zip_path: Path, manifest: pd.DataFrame) -> dict[str, str]:
    """Validate repository ZIP and map approved basenames to ZIP member names.

    ZIP may contain nested directories. Only manifest-approved files are accepted.
    No answer key is ever loaded from the repository ZIP.
    """
    expected = set(manifest["file"])
    matched = {}
    total = 0
    try:
        with zipfile.ZipFile(zip_path) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_MEMBERS:
                raise ValueError("Media ZIP contains too many entries")
            for info in infos:
                if info.is_dir():
                    continue
                path = PurePosixPath(info.filename.replace("\\", "/"))
                if path.is_absolute() or ".." in path.parts or not path.parts:
                    raise ValueError("Unsafe ZIP entry")
                if stat.S_IFMT(info.external_attr >> 16) == stat.S_IFLNK:
                    raise ValueError("ZIP symlinks are not allowed")
                name = path.name
                # ZIPs created on macOS sometimes contain harmless OS metadata.
                if name == ".DS_Store" or "__MACOSX" in path.parts:
                    continue
                if name not in expected:
                    raise ValueError(f"Unexpected file in ZIP: {name}")
                if name in matched:
                    raise ValueError(f"Duplicate filename inside ZIP: {name}")
                if info.file_size > MAX_SINGLE_MEDIA:
                    raise ValueError(f"Media file too large: {name}")
                total += info.file_size
                if total > MAX_TOTAL_UNCOMPRESSED:
                    raise ValueError("ZIP uncompressed content exceeds configured limit")
                matched[name] = info.filename
    except zipfile.BadZipFile as exc:
        raise ValueError("The repository media ZIP is invalid") from exc
    if set(matched) != expected:
        missing = sorted(expected - set(matched))
        raise ValueError(f"ZIP and private manifest differ: {len(missing)} missing media; examples: {missing[:5]}")
    return matched


def load_media_bytes(name: str) -> bytes:
    entry = st.session_state.zip_entries[name]
    with zipfile.ZipFile(st.session_state.zip_path) as archive:
        with archive.open(entry) as item:
            data = item.read(MAX_SINGLE_MEDIA + 1)
    if len(data) > MAX_SINGLE_MEDIA:
        raise ValueError("Media exceeds maximum permitted size")
    return data


def playable_video(name: str) -> Path:
    """Extract one video only, convert to H.264 for browser playback; cache per session."""
    outdir = Path(st.session_state.temp_handle.name)
    converted = outdir / (Path(name).stem + "_h264.mp4")
    if converted.is_file() and converted.stat().st_size:
        return converted
    source = outdir / (Path(name).stem + "_source.mp4")
    partial = outdir / (Path(name).stem + "_partial.mp4")
    try:
        source.write_bytes(load_media_bytes(name))
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
            "-nostdin", "-y", "-i", str(source), "-an", "-c:v", "libx264",
            "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(partial),
        ]
        done = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
        if done.returncode != 0 or not partial.is_file() or not partial.stat().st_size:
            raise RuntimeError("Browser-compatible video conversion failed")
        partial.replace(converted)
    except Exception:
        logging.exception("Video preparation failed: %s", name)
        raise
    finally:
        source.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)
    return converted


def score_rows(responses: list[dict]) -> list[dict]:
    data = pd.DataFrame(responses)
    output = []
    groups = [("overall", "all", data)]
    for field in ("source", "view"):
        groups.extend((field, name, subset) for name, subset in data.groupby(field, sort=True))
    for group, category, subset in groups:
        total = len(subset)
        correct = int(subset["correct"].sum()) if total else 0
        output.append({"group": group, "category": category, "total": total,
                       "correct": correct, "accuracy": correct / total if total else 0.0})
    return output


def csv_bytes(columns, rows) -> bytes:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8-sig")


def reset_session():
    handle = st.session_state.get("temp_handle")
    if handle is not None:
        handle.cleanup()
    for name in ("started", "dataset", "zip_path", "zip_entries", "reader", "modality",
                 "order", "idx", "responses", "session_id", "notes", "temp_handle", "finalized"):
        st.session_state.pop(name, None)


st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Classify each echocardiogram as real or synthetic. Your score appears at the end.")

if not st.session_state.get("started", False):
    with st.form("setup"):
        modality = st.radio("Choose experiment", ["ED images", "Videos"], horizontal=True)
        reader = st.text_input("Reader ID (pseudonym)", placeholder="clinician_01")
        start = st.form_submit_button("Start classification", type="primary")
    if start:
        if not reader.strip():
            st.error("Please enter a reader ID.")
            st.stop()
        handle = None
        try:
            manifest = load_private_manifest(modality)
            zip_path = media_zip_path(modality)
            entries = inspect_zip(zip_path, manifest)
            handle = tempfile.TemporaryDirectory(prefix="echoclinician_")
        except Exception as exc:
            if handle is not None:
                handle.cleanup()
            logging.exception("Study initialization failed")
            st.error(f"Cannot start experiment: {exc}")
            st.stop()
        order = list(range(len(manifest)))
        secrets.SystemRandom().shuffle(order)
        st.session_state.update(dict(
            started=True, dataset=manifest, zip_path=str(zip_path), zip_entries=entries,
            temp_handle=handle, reader=reader.strip(), modality=modality, order=order,
            idx=0, responses=[], session_id=secrets.token_hex(12),
            notes="", finalized=False,
        ))
        st.rerun()
    st.stop()

count = len(st.session_state.order)
idx = st.session_state.idx
st.progress(idx / count)
st.caption(f"{st.session_state.modality} · {idx} of {count} answers submitted")

if idx < count and not st.session_state.finalized:
    sample = st.session_state.dataset.iloc[st.session_state.order[idx]]
    filename = str(sample["file"])
    media, answer = st.columns([4, 2], gap="large")
    with media:
        try:
            if st.session_state.modality == "ED images":
                st.image(load_media_bytes(filename), width=DISPLAY_SIZE)
            else:
                video = playable_video(filename)
                st.video(str(video), format="video/mp4", autoplay=False,
                         loop=True, width=DISPLAY_SIZE)
        except Exception:
            logging.exception("Cannot present media")
            st.error("Unable to display this media. Please contact the study administrator.")
            st.stop()
    with answer:
        st.subheader("Classification")
        st.caption("Submitted answers cannot be changed.")
        selected = st.radio("This sample appears to be:", ["Real", "Synthetic"],
                            index=None, key=f"answer_{st.session_state.session_id}_{idx}")
        if st.button("Submit answer", type="primary", disabled=selected is None):
            prediction = "real" if selected == "Real" else "fake"
            st.session_state.responses.append(dict(
                session_id=st.session_state.session_id,
                reader_id=st.session_state.reader,
                modality=st.session_state.modality,
                sample_number=idx + 1,
                sample_id=str(sample["sample_id"]),
                view=str(sample["view"]),
                prediction=prediction,
                true_label=str(sample["true_label"]),
                source=str(sample["source"]),
                correct=int(prediction == sample["true_label"]),
                original_acquisition=str(sample["original_acquisition"]),
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                notes="",
            ))
            st.session_state.idx += 1
            st.rerun()
    st.divider()
    if st.button("Finish early and show my score", disabled=not st.session_state.responses):
        st.session_state.finalized = True
        st.rerun()
    st.stop()

# Scores and true labels are shown only after a reader finishes or stops early.
completed = len(st.session_state.responses)
st.success("Experiment complete." if completed == count else
           f"Finished early: {completed} of {count} samples classified.")
st.session_state.notes = st.text_area("Optional comments", value=st.session_state.notes)
for record in st.session_state.responses:
    record["notes"] = st.session_state.notes

scores = score_rows(st.session_state.responses)
overall = scores[0]
st.header("Your results")
c1, c2, c3 = st.columns(3)
c1.metric("Accuracy", f"{overall['accuracy']:.1%}")
c2.metric("Correct answers", str(overall["correct"]))
c3.metric("Answered", f"{overall['total']} / {count}")

score_frame = pd.DataFrame(scores)
score_frame["accuracy"] = score_frame["accuracy"].map(lambda x: f"{x:.1%}")
st.subheader("Accuracy by source and echocardiographic view")
st.dataframe(score_frame, hide_index=True, use_container_width=True)
st.caption("Stage 1 and Stage 2 are both synthetic; real samples are the reference category. "
           "Early finishes are scored only on submitted answers.")

col1, col2 = st.columns(2)
with col1:
    st.download_button("Download my responses (CSV)",
                       csv_bytes(RESPONSE_COLUMNS, st.session_state.responses),
                       file_name=f"responses_{st.session_state.session_id}.csv",
                       mime="text/csv")
with col2:
    st.download_button("Download my scores (CSV)",
                       csv_bytes(SCORE_COLUMNS, scores),
                       file_name=f"scores_{st.session_state.session_id}.csv",
                       mime="text/csv")

st.info("Results are shown here and are not emailed. Download the CSV files if you need to keep a copy.")
if st.button("Start another experiment"):
    reset_session()
    st.rerun()
