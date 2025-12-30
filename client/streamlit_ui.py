import json
import os
import time
from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st
from kafka import KafkaProducer

from common.kafka_topics import TRAINING_REQUESTS_TOPIC
from client.preprocessing import preprocess_csv_to_flat_file

st.set_page_config(page_title="Cardio Federated Client", layout="wide")

CLIENT_ID = os.getenv("CLIENT_ID", "client_1")
RAW_DATA_PATH = os.getenv("CLIENT_RAW_DATA_PATH", "data/raw.csv")
PROCESSED_DATA_PATH = os.getenv("CLIENT_DATA_PATH", "data/processed/train.csv")
LOG_PATH = os.getenv("CLIENT_LOG_PATH", "logs/client_training.log")
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Fraunces:wght@600&display=swap');

:root {
  --bg: #f6f2ec;
  --bg-accent: #e6eef1;
  --ink: #1b1a17;
  --ink-soft: #4b4a46;
  --panel: #ffffffcc;
  --panel-strong: #ffffff;
  --line: #e0d8cf;
  --accent: #0f6d6a;
  --accent-2: #d9773b;
}

.stApp {
  background: radial-gradient(1200px 600px at 15% 10%, #fff7f1 0%, var(--bg) 50%, var(--bg-accent) 100%);
  color: var(--ink);
  font-family: "Space Grotesk", sans-serif;
}

h1, h2, h3 {
  font-family: "Fraunces", serif !important;
  letter-spacing: 0.3px;
}

.hero {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 18px 22px;
  margin-bottom: 18px;
  backdrop-filter: blur(8px);
}

.hero-title {
  font-size: 26px;
  margin: 0;
}

.hero-sub {
  color: var(--ink-soft);
  margin-top: 6px;
}

.panel {
  background: var(--panel-strong);
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 16px;
  margin-bottom: 16px;
}

.grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(140px, 1fr));
  gap: 12px;
}

.card {
  background: #fff;
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 12px;
}

.card-title {
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: var(--ink-soft);
  margin-bottom: 6px;
}

.card-value {
  font-size: 20px;
  font-weight: 700;
}

.tag {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  font-size: 12px;
  background: #f2efe9;
  border: 1px solid var(--line);
  color: var(--ink-soft);
}

.tag.accent {
  background: #e0f1ef;
  border-color: #c3e4e1;
  color: var(--accent);
}

.tag.warn {
  background: #fde6d7;
  border-color: #f7c8a8;
  color: var(--accent-2);
}

.step {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 6px 0;
}

.step-dot {
  width: 10px;
  height: 10px;
  border-radius: 999px;
  background: #c8c1b7;
}

.step-dot.done {
  background: var(--accent);
}

.step-dot.failed {
  background: var(--accent-2);
}

.step-dot.waiting {
  background: #f2c88b;
}

.step-label {
  font-weight: 600;
}

.step-meta {
  color: var(--ink-soft);
  font-size: 12px;
}

div.stButton > button {
  background: var(--accent);
  color: white;
  border-radius: 12px;
  border: none;
  padding: 10px 18px;
  font-weight: 600;
}

div.stButton > button:hover {
  background: #0c5c59;
  color: white;
}
</style>
""",
    unsafe_allow_html=True,
)


def tail_file(path: str, max_lines: int = 200) -> List[str]:
    try:
        with open(path, "r") as handle:
            lines = handle.readlines()
        return lines[-max_lines:]
    except FileNotFoundError:
        return []


def get_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: v,
    )


def log_ui_event(message: str) -> None:
    if not LOG_PATH:
        return
    log_dir = os.path.dirname(LOG_PATH)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_PATH, "a") as handle:
        handle.write(f"{timestamp} INFO:streamlit_ui:{message}\n")


def extract_last_server_decision(lines: List[str]) -> str:
    for line in reversed(lines):
        if "Server decision:" in line:
            return line.strip()
    return "No server decision yet."


def _find_last_line(lines: List[str], token: str) -> Optional[str]:
    for line in reversed(lines):
        if token in line:
            return line.strip()
    return None


def _parse_decision_status(line: Optional[str]) -> Tuple[str, str]:
    if not line:
        return "pending", "Waiting for server decision."
    if "accepted': True" in line or "accepted\": true" in line:
        return "done", "Accepted by server."
    if "accepted': False" in line or "accepted\": false" in line:
        return "failed", "Rejected by server."
    return "waiting", "Decision received."


def build_progress_steps(
    logs: List[str],
    upload_state: str,
    pre_state: str,
) -> List[Tuple[str, str, str]]:
    steps = []

    steps.append(
        (
            "Upload data",
            "done" if upload_state == "ready" else "pending",
            "CSV uploaded." if upload_state == "ready" else "Waiting for upload.",
        )
    )
    steps.append(
        (
            "Preprocess data",
            "done" if pre_state == "done" else "pending",
            "Processed file ready." if pre_state == "done" else "Waiting for preprocess.",
        )
    )

    request_line = _find_last_line(logs, "Training request sent.")
    steps.append(
        (
            "Training request",
            "done" if request_line else "pending",
            "Sent to server." if request_line else "Not sent yet.",
        )
    )

    received_line = _find_last_line(logs, "Received global model from Kafka")
    steps.append(
        (
            "Global model received",
            "done" if received_line else "pending",
            "Model dispatched by server." if received_line else "Waiting for dispatch.",
        )
    )

    validation_line = _find_last_line(logs, "Data validation failed")
    if validation_line:
        steps.append(
            (
                "Data validation",
                "failed",
                validation_line.split("Data validation failed:", 1)[-1].strip(),
            )
        )
    else:
        steps.append(
            (
                "Data validation",
                "done" if received_line else "pending",
                "Validation passed." if received_line else "Not started.",
            )
        )

    training_line = _find_last_line(logs, "Starting training for round")
    steps.append(
        (
            "Local training",
            "done" if training_line else "pending",
            training_line if training_line else "Waiting for training.",
        )
    )

    sent_line = _find_last_line(logs, "sent to Kafka")
    steps.append(
        (
            "Update sent",
            "done" if sent_line else "pending",
            sent_line if sent_line else "No update sent yet.",
        )
    )

    decision_line = _find_last_line(logs, "Server decision:")
    decision_status, decision_hint = _parse_decision_status(decision_line)
    steps.append(("Server decision", decision_status, decision_hint))

    return steps


def extract_training_state(lines: List[str]) -> Tuple[str, Optional[int]]:
    state = "idle"
    last_round = None
    for line in reversed(lines):
        if "Starting training for round" in line:
            state = "training"
            try:
                last_round = int(line.split("round")[-1].strip())
            except Exception:
                last_round = None
            break
        if "Round" in line and "sent to Kafka" in line:
            state = "waiting_server"
            try:
                last_round = int(line.split("Round")[1].split()[0])
            except Exception:
                last_round = None
            break
    return state, last_round


st.markdown(
    f"""
<div class="hero">
  <p class="hero-title">Cardio Client - {CLIENT_ID}</p>
  <div class="hero-sub">
    Real-time pipeline: upload, spark preprocessing, local training, server selection.
  </div>
</div>
""",
    unsafe_allow_html=True,
)

left, right = st.columns([2, 1], gap="large")

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Upload data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
        df.to_csv(RAW_DATA_PATH, index=False)
        st.success(f"{len(df)} rows uploaded.")
        log_ui_event(f"Upload complete ({len(df)} rows).")
        st.dataframe(df.head())
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Actions")

    if "last_request_ts" not in st.session_state:
        st.session_state.last_request_ts = None

    if st.button("Start preprocessing + training"):
        if os.path.exists(RAW_DATA_PATH):
            try:
                processed_path = preprocess_csv_to_flat_file(
                    RAW_DATA_PATH, PROCESSED_DATA_PATH
                )
                st.success(f"Preprocessing complete: {processed_path}")
                log_ui_event(f"Preprocess complete: {processed_path}.")

                producer = get_producer()
                payload = {
                    "client_id": CLIENT_ID,
                    "timestamp": time.time(),
                    "data_path": PROCESSED_DATA_PATH,
                }
                message = json.dumps(payload).encode("utf-8")
                producer.send(TRAINING_REQUESTS_TOPIC, value=message)
                producer.flush()
                st.session_state.last_request_ts = time.time()
                st.success("Training request sent.")
                log_ui_event("Training request sent.")
            except Exception as exc:
                st.error(f"Preprocessing/training failed: {exc}")
        else:
            st.warning("Upload a CSV file first.")

    if st.session_state.last_request_ts:
        st.caption("Training request sent. Waiting for server dispatch.")

    st.markdown("</div>", unsafe_allow_html=True)

logs = tail_file(LOG_PATH, max_lines=200)
state, last_round = extract_training_state(logs)

upload_state = "ready" if os.path.exists(RAW_DATA_PATH) else "missing"
pre_state = "done" if os.path.exists(PROCESSED_DATA_PATH) else "pending"

st.markdown('<div class="panel">', unsafe_allow_html=True)
st.subheader("Live status")
st.markdown(
    f"""
<div class="grid">
  <div class="card">
    <div class="card-title">Upload</div>
    <div class="card-value">{upload_state}</div>
    <span class="tag {'accent' if upload_state == 'ready' else 'warn'}">{upload_state}</span>
  </div>
  <div class="card">
    <div class="card-title">Preprocess</div>
    <div class="card-value">{pre_state}</div>
    <span class="tag {'accent' if pre_state == 'done' else 'warn'}">{pre_state}</span>
  </div>
  <div class="card">
    <div class="card-title">Training</div>
    <div class="card-value">{state}</div>
    <span class="tag {'accent' if state == 'training' else ''}">{state}</span>
  </div>
  <div class="card">
    <div class="card-title">Last Round</div>
    <div class="card-value">{last_round if last_round is not None else '-'}</div>
    <span class="tag">round</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="panel">', unsafe_allow_html=True)
st.subheader("Pipeline progress")
steps = build_progress_steps(logs, upload_state, pre_state)
for title, status, detail in steps:
    dot_class = "step-dot"
    if status in ("done", "failed", "waiting"):
        dot_class = f"{dot_class} {status}"
    st.markdown(
        f"""
<div class="step">
  <span class="{dot_class}"></span>
  <div>
    <div class="step-label">{title}</div>
    <div class="step-meta">{detail}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="panel">', unsafe_allow_html=True)
st.subheader("Server decision")
st.text(extract_last_server_decision(logs))
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="panel">', unsafe_allow_html=True)
st.subheader("Client logs")
if logs:
    st.text_area("Logs", value="".join(logs), height=280)
else:
    st.text_area("Logs", value="No logs yet.", height=200)
st.markdown("</div>", unsafe_allow_html=True)
