"""Server-hosted blinded clinician study. Run: streamlit run app.py"""
from __future__ import annotations

import csv
import secrets
import smtplib
import ssl
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title='Echocardiography realism study', layout='wide')
st.title('Echocardiography realism study')
st.caption('For each sample, decide whether the echocardiogram is real or synthetic.')


def secret_path(key):
    return Path(str(st.secrets['study'][key])).expanduser().resolve()


def get_dataset(modality):
    media_dir = secret_path('images_dir' if modality == 'ED images' else 'videos_dir')
    manifest = secret_path('images_labels' if modality == 'ED images' else 'videos_labels')
    if not media_dir.is_dir() or not manifest.is_file():
        raise FileNotFoundError(f'Study directory or label manifest missing for {modality}')
    data = pd.read_csv(manifest, keep_default_na=False)
    expected = {'sample_id', 'file', 'view', 'source', 'true_label', 'original_acquisition'}
    if not expected.issubset(data.columns):
        raise ValueError(f'Manifest is missing: {sorted(expected - set(data.columns))}')
    for name in data['file']:
        # Ensure a manifest cannot traverse outside the media directory.
        if Path(name).name != name or not (media_dir / name).is_file():
            raise FileNotFoundError(f'Missing or unsafe media filename: {name}')
    return data, media_dir


def reset():
    for key in ('started', 'dataset', 'media_dir', 'reader', 'modality', 'order',
                'idx', 'responses', 'session_id', 'submitted', 'notes'):
        st.session_state.pop(key, None)


def result_path():
    folder = secret_path('results_dir')
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"responses_{st.session_state.session_id}.csv"


def save_responses():
    rows = st.session_state.responses
    if not rows:
        return None
    path = result_path()
    fieldnames = ['session_id', 'reader_id', 'modality', 'sample_number', 'sample_id',
                  'view', 'prediction', 'true_label', 'source', 'correct',
                  'original_acquisition', 'timestamp_utc', 'notes']
    # Append/update the same per-session CSV after every response.
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def send_email(path):
    smtp = st.secrets['smtp']
    msg = EmailMessage()
    msg['From'] = smtp['sender_email']
    msg['To'] = smtp['recipient_email']
    msg['Subject'] = f"Echocardiography clinician study: {st.session_state.modality} / {st.session_state.session_id}"
    msg.set_content(
        f"Session: {st.session_state.session_id}\n"
        f"Reader: {st.session_state.reader}\n"
        f"Study: {st.session_state.modality}\n"
        f"Responses: {len(st.session_state.responses)}\n"
        'The full response CSV is attached.\n'
    )
    msg.add_attachment(path.read_bytes(), maintype='text', subtype='csv', filename=path.name)
    host, port = smtp['host'], int(smtp['port'])
    username, password = smtp['username'], smtp['password']
    if port == 465:
        with smtplib.SMTP_SSL(host, port, context=ssl.create_default_context(), timeout=30) as server:
            server.login(username, password)
            server.send_message(msg)
    else:
        with smtplib.SMTP(host, port, timeout=30) as server:
            server.starttls(context=ssl.create_default_context())
            server.login(username, password)
            server.send_message(msg)


if 'started' not in st.session_state:
    st.session_state.started = False

if not st.session_state.started:
    with st.form('setup'):
        modality = st.radio('Choose experiment', ['ED images', 'Videos'], horizontal=True)
        reader = st.text_input('Reader ID (pseudonym)', placeholder='clinician_01')
        submitted = st.form_submit_button('Start classification', type='primary')
    if submitted:
        if not reader.strip():
            st.error('Enter a reader ID.')
            st.stop()
        try:
            dataset, media_dir = get_dataset(modality)
        except Exception as exc:
            st.error(f'Cannot load study: {exc}')
            st.stop()
        # Per-reader blinded permutation. Do not send truth or filenames in UI text.
        st.session_state.dataset = dataset
        st.session_state.media_dir = str(media_dir)
        st.session_state.order = list(pd.Series(range(len(dataset))).sample(frac=1,
                                              random_state=secrets.randbelow(2**32)).values)
        st.session_state.reader = reader.strip()
        st.session_state.modality = modality
        st.session_state.idx = 0
        st.session_state.responses = []
        st.session_state.session_id = secrets.token_hex(12)
        st.session_state.submitted = False
        st.session_state.notes = ''
        st.session_state.started = True
        st.rerun()
    st.stop()

n = len(st.session_state.order)
i = st.session_state.idx
st.progress(i / n if n else 0)
st.caption(f"{st.session_state.modality} · Sample {min(i + 1, n)} of {n}")

if i < n:
    row = st.session_state.dataset.iloc[st.session_state.order[i]]
    path = Path(st.session_state.media_dir) / row['file']
    media_col, answer_col = st.columns([4, 2], gap='large')
    with media_col:
        if st.session_state.modality == 'ED images':
            st.image(str(path), width=420)
        else:
            st.video(str(path), format='video/mp4', autoplay=False, loop=True)
    with answer_col:
        st.subheader('Your classification')
        st.caption('Select one answer. You cannot change a submitted response.')
        choice = st.radio('This sample appears to be:', ['Real', 'Synthetic'],
                          index=None, key=f"choice_{st.session_state.session_id}_{i}")
        if st.button('Submit answer', type='primary', disabled=choice is None):
            prediction = 'real' if choice == 'Real' else 'fake'
            st.session_state.responses.append({
                'session_id': st.session_state.session_id,
                'reader_id': st.session_state.reader,
                'modality': st.session_state.modality,
                'sample_number': i + 1,
                'sample_id': row['sample_id'],
                'view': row['view'],
                'prediction': prediction,
                'true_label': row['true_label'],
                'source': row['source'],
                'correct': int(prediction == row['true_label']),
                'original_acquisition': row['original_acquisition'],
                'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                'notes': '',
            })
            save_responses()
            st.session_state.idx += 1
            st.rerun()
    st.stop()

st.success('You have classified all samples.')
st.session_state.notes = st.text_area('Optional comments', value=st.session_state.notes)
if st.session_state.responses:
    for row in st.session_state.responses:
        row['notes'] = st.session_state.notes
    csv_path = save_responses()
    if not st.session_state.submitted:
        if st.button('Submit results by email', type='primary'):
            try:
                send_email(csv_path)
                st.session_state.submitted = True
                st.success('Results emailed successfully.')
            except Exception as exc:
                st.error(f'Email failed. Responses were saved on the server. Error: {exc}')
    else:
        st.success('Results have been emailed.')
if st.button('Start another experiment'):
    reset()
    st.rerun()
