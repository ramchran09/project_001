import os
import time
import pickle
import threading

import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_PATH  = os.path.join("artifacts", "final_model.keras")
SCALER_PATH = os.path.join("artifacts", "scaler.pkl")
HINT_URL    = "http://127.0.0.1:6000/generate-hint"
HINT_COOLDOWN_SECONDS    = 10     # cooldown after each hint response
INITIAL_COOLDOWN_SECONDS = 10     # block hints for 10s after session starts

FEATURE_COLS = [
    "avg_dwell", "avg_flight", "typing_speed",
    "pause_count", "backspace_count", "std_dwell", "std_flight",
]
CLASS_MAP = {0: "low", 1: "medium", 2: "high"}

# ── Load artifacts ────────────────────────────────────────────────────────────

model = tf.keras.models.load_model(MODEL_PATH)
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# ── Hint-fetch gate ───────────────────────────────────────────────────────────

_hint_lock          = threading.Lock()
_hint_fetching      = False
_hint_ready_at      = 0.0
_session_start_time = 0.0     # set when /start is called

# ── Session state ─────────────────────────────────────────────────────────────

session: dict = {}

def reset_session() -> None:
    session.update({
        "row_buffer":      [],
        "current_hint":    "",
        "last_prediction": None,
        "problem":         "",
        "answer":          "",
        "prev_hint":       "",
    })

reset_session()

# ── Helpers ───────────────────────────────────────────────────────────────────

def predict_cognitive_load(rows: list[dict]) -> str:
    df     = pd.DataFrame(rows, columns=FEATURE_COLS)
    scaled = scaler.transform(df)
    X      = scaled.reshape(1, 5, 7)
    probs  = model.predict(X, verbose=0)
    idx    = int(np.argmax(probs, axis=1)[0])
    return CLASS_MAP[idx]


def fetch_hint(problem: str, answer: str, label: str, prev_hint: str) -> str:
    global _hint_fetching, _hint_ready_at

    with _hint_lock:
        if _hint_fetching:
            print("[hint gate] blocked – fetch already in-flight")
            return ""
        if time.time() < _hint_ready_at:
            remaining = _hint_ready_at - time.time()
            print(f"[hint gate] blocked – cooldown active ({remaining:.1f}s left)")
            return ""
        _hint_fetching = True

    hint = ""
    try:
        print("[hint gate] fetching hint …")
        res  = requests.post(
            HINT_URL,
            json={"problem": problem, "answer": answer,
                  "label": label, "prev_hint": prev_hint},
            timeout=15,
        )
        hint = res.json().get("hint", "")
        print("[hint gate] hint received")
    except Exception as e:
        print(f"[hint server error] {e}")
        hint = "Try breaking the problem into smaller steps."
    finally:
        with _hint_lock:
            _hint_ready_at = time.time() + HINT_COOLDOWN_SECONDS
            _hint_fetching = False
            print(f"[hint gate] cooldown started – next request allowed at "
                  f"{time.strftime('%H:%M:%S', time.localtime(_hint_ready_at))}")

    return hint

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/start", methods=["POST"])
def start():
    global _hint_fetching, _hint_ready_at, _session_start_time
    reset_session()
    with _hint_lock:
        _hint_fetching      = False
        _hint_ready_at      = 0.0
    _session_start_time = time.time()     # record session start
    print(f"[session] started – hints blocked until "
          f"{time.strftime('%H:%M:%S', time.localtime(_session_start_time + INITIAL_COOLDOWN_SECONDS))}")
    return jsonify({"status": "session started"})


@app.route("/stream-data", methods=["POST"])
def stream_data():
    data = request.get_json(force=True)

    session["problem"] = data.get("problem", "")
    session["answer"]  = data.get("answer", "")

    row = {col: float(data.get("row", {}).get(col, 0)) for col in FEATURE_COLS}
    session["row_buffer"].append(row)

    n = len(session["row_buffer"])

    if n < 5:
        return jsonify({"prediction": None, "window_range": None,
                        "hint": "", "windows_done": n})

    label        = predict_cognitive_load(session["row_buffer"][-5:])
    window_range = f"w{n-4}–w{n}"
    print(f"[prediction] window={window_range}  label={label}")

    prediction_changed = (label != session["last_prediction"])
    needs_hint = label in ("medium", "high")

    # Check initial cooldown
    initial_passed = (time.time() - _session_start_time) >= INITIAL_COOLDOWN_SECONDS

    hint = session["current_hint"]

    if needs_hint and prediction_changed and initial_passed:
        new_hint = fetch_hint(
            session["problem"], session["answer"],
            label, session["prev_hint"],
        )
        if new_hint:
            session["prev_hint"]    = new_hint
            session["current_hint"] = new_hint
            hint = new_hint
    elif needs_hint and not initial_passed:
        remaining = INITIAL_COOLDOWN_SECONDS - (time.time() - _session_start_time)
        print(f"[initial cooldown] hints blocked – {remaining:.1f}s remaining")

    session["last_prediction"] = label
    return jsonify({
        "prediction":   label,
        "window_range": window_range,
        "hint":         hint,
        "windows_done": n,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
