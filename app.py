"""
AI-Enhanced Phishing Email Detection — API
---------------------------------------------
Flask backend implementing the four-stage pipeline described in the
project report: ingest, extract features, score in parallel
(content classifier + behavior profiling engine), then fuse into one
decision.

Local development:
    pip install -r requirements.txt
    python app.py

Production:
    See README.md "Deploying this" — short version: run behind
    gunicorn, not `python app.py`, and set the environment variables
    described below.

Environment variables (all optional, safe defaults for local dev):
    FLASK_DEBUG        "1" to enable Flask's debugger. Defaults to "0"
                        (off). Never set this to "1" anywhere but your
                        own machine — see README "Production hardening".
    ALLOWED_ORIGINS     Comma-separated list of origins allowed to call
                        this API, e.g. "https://your-frontend.com".
                        Defaults to "*" (any origin) for local demo use.
    API_KEY             If set, every request to /api/analyze must
                        include a matching "X-API-Key" header. If unset,
                        the API is open — fine for a local demo, not for
                        a public deployment.
    RATE_LIMIT          Requests allowed per minute per client IP.
                        Defaults to "30/minute".
"""

import json
import os
from datetime import datetime
from functools import wraps

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from behavior_engine import load_profiles, score_behavior
from classifier import AVAILABLE_MODELS, DEFAULT_MODEL, classify_content
from feature_extraction import extract_features
from fusion import fuse_decision
from url_model import score_url

# ---- configuration (all overridable via environment variables) ----
DEBUG_MODE = os.environ.get("FLASK_DEBUG", "0") == "1"
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*")
API_KEY = os.environ.get("API_KEY")  # None = auth disabled (local demo mode)
RATE_LIMIT = os.environ.get("RATE_LIMIT", "30/minute")
MAX_CONTENT_LENGTH = 200 * 1024  # 200 KB — an email body has no business being bigger

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

origins = "*" if ALLOWED_ORIGINS == "*" else [o.strip() for o in ALLOWED_ORIGINS.split(",")]
CORS(app, origins=origins)

limiter = Limiter(get_remote_address, app=app, default_limits=[RATE_LIMIT])

if DEBUG_MODE:
    print("WARNING: FLASK_DEBUG=1 — Flask's interactive debugger is ENABLED. "
          "Do not run this anywhere but your own machine.")
if origins == "*":
    print("WARNING: ALLOWED_ORIGINS not set — accepting requests from any origin. "
          "Set ALLOWED_ORIGINS before deploying publicly.")
if not API_KEY:
    print("NOTE: API_KEY not set — /api/analyze has no authentication. "
          "Set API_KEY before deploying publicly.")


def require_api_key(view):
    """No-op if API_KEY isn't configured (local demo mode). Otherwise
    requires a matching X-API-Key header on every request."""
    @wraps(view)
    def wrapped(*args, **kwargs):
        if API_KEY and request.headers.get("X-API-Key") != API_KEY:
            return jsonify({"error": "missing or invalid API key"}), 401
        return view(*args, **kwargs)
    return wrapped


@app.errorhandler(413)
def request_too_large(_e):
    return jsonify({"error": f"request body too large (max {MAX_CONTENT_LENGTH // 1024} KB)"}), 413


@app.route("/api/health", methods=["GET"])
def health():
    """Simple liveness check for hosting platforms / uptime monitors."""
    return jsonify({"status": "ok"})


@app.route("/api/senders", methods=["GET"])
def get_senders():
    """List the known sender profiles, for populating a sender dropdown."""
    return jsonify(load_profiles())


@app.route("/api/models", methods=["GET"])
def get_models():
    """List the available content classifiers and their real evaluation metrics."""
    metrics_path = os.path.join(os.path.dirname(__file__), "models", "metrics.json")
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)

    return jsonify({"available": list(AVAILABLE_MODELS), "default": DEFAULT_MODEL, "metrics": metrics})


@app.route("/api/analyze", methods=["POST"])
@limiter.limit(RATE_LIMIT)
@require_api_key
def analyze_email():
    """
    Run the full detection pipeline on one email.

    Expected JSON body:
        {
          "email_text": "...",              required
          "sender_id": "known_hr",          optional, defaults to "unknown"
          "hour": 14,                       optional, defaults to the current hour
          "model": "logistic_regression"    optional, one of AVAILABLE_MODELS
        }

    If API_KEY is set on the server, requests must also include a
    matching "X-API-Key" header.
    """
    data = request.get_json(force=True, silent=True) or {}

    email_text = (data.get("email_text") or "").strip()
    if not email_text:
        return jsonify({"error": "email_text is required"}), 400
    if len(email_text) > 50_000:
        return jsonify({"error": "email_text too long (50,000 character limit)"}), 400

    sender_id = data.get("sender_id", "unknown")
    hour = data.get("hour", datetime.now().hour)
    model_name = data.get("model", DEFAULT_MODEL)

    # Stage 1 (parsing) is implicit here — the request already arrives
    # as separated text, sender, and timestamp fields.

    # Stage 2: feature extraction.
    features = extract_features(email_text)

    # Stage 3: score in parallel — content, behavior, and (if any URLs
    # were found) the secondary URL risk model.
    content_result = classify_content(email_text, model_name=model_name)
    behavior_result = score_behavior(sender_id, hour, features["urgency_ratio"])

    url_results = [score_url(u) for u in features.get("urls", [])]
    url_score = max((r["score"] for r in url_results), default=0.0)

    # Stage 4: fuse into one decision. The URL model nudges the content
    # score upward when it disagrees more strongly, rather than being
    # averaged in and diluting a clear text-based verdict either way.
    effective_content_score = max(content_result["score"], url_score)
    final_score, verdict = fuse_decision(effective_content_score, behavior_result["score"])

    return jsonify({
        "verdict": verdict,
        "final_score": final_score,
        "content": content_result,
        "behavior": behavior_result,
        "url_risk": url_results,
        "features": features,
    })


if __name__ == "__main__":
    # For local development only. In production, run behind a real WSGI
    # server instead of Flask's dev server, e.g.:
    #   gunicorn -w 2 -b 0.0.0.0:5000 app:app
    app.run(debug=DEBUG_MODE, host="127.0.0.1", port=int(os.environ.get("PORT", 5000)))
