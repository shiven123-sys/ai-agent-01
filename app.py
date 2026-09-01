"""
Flask web app for the agent.

Run: python api/app.py
Then open http://localhost:5000

Note: uses a single global agent/session for simplicity (fine for local demo
use). For multi-user production use, keep a dict of Agent() per session id.
"""

import os
import sys

from flask import Flask, request, jsonify, render_template, session

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from agent.agent import Agent  # noqa: E402
from agent import config  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = os.path.join(BASE_DIR, "static")

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=TEMPLATE_DIR)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

_agents = {}  # session_id -> Agent  (simple in-memory store, demo-scale only)


def get_agent():
    sid = session.get("sid")
    if sid is None:
        import uuid
        sid = str(uuid.uuid4())
        session["sid"] = sid
    if sid not in _agents:
        _agents[sid] = Agent()
    return _agents[sid]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "configured": config.is_configured(), "provider": config.LLM_PROVIDER})


@app.route("/chat", methods=["POST"])
def chat():
    if not config.is_configured():
        return jsonify({"error": config.setup_instructions()}), 503

    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "Field 'message' is required."}), 400

    try:
        agent = get_agent()
        result = agent.chat(message)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/reset", methods=["POST"])
def reset():
    try:
        agent = get_agent()
        agent.reset()
        return jsonify({"status": "reset"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
