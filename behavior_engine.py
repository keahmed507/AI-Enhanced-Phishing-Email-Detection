"""
Sender Behavior Profiling Engine
-----------------------------------
Combines two anomaly-detection approaches over a sender's historical
behavior, matching Section 4.9.2 of the report: Z-score analysis and
Isolation Forest.

Honest disclosure: no public dataset reachable from this environment
pairs real per-sender, timestamped email history with account-compromise
labels (Enron's raw maildir has timestamps but no phishing/compromise
labels; no such dataset exists publicly). Each sender's baseline is
therefore a small hand-set distribution (data/sender_profiles.json),
and a synthetic sample is drawn from it to fit an Isolation Forest,
standing in for the months of real traffic a production deployment
would use. The Isolation Forest algorithm itself is real and untouched
— only the historical data it trains on is simulated. This is stated
here and in the README rather than presented as scored on real
behavioral logs.
"""

import json
import os

import numpy as np
from sklearn.ensemble import IsolationForest

PROFILES_PATH = os.path.join(os.path.dirname(__file__), "data", "sender_profiles.json")

_forest_cache = {}


def load_profiles() -> dict:
    """Load all known sender baselines from disk."""
    with open(PROFILES_PATH) as f:
        return json.load(f)


def _synthetic_history(profile: dict, n_samples: int = 150) -> np.ndarray:
    """
    Draw a synthetic behavioral history for one sender from their
    baseline mean/std — standing in for months of real traffic.
    Columns: [hour_of_day, urgency_ratio].
    """
    seed = abs(hash(profile["label"])) % (2**32)
    rng = np.random.default_rng(seed)
    hours = rng.normal(profile["typical_hour_mean"], profile["typical_hour_std"] or 1, n_samples)
    urgency = np.clip(
        rng.normal(profile["urgency_ratio_mean"], profile["urgency_ratio_std"] or 0.05, n_samples), 0, 1
    )
    return np.column_stack([hours, urgency])


def _get_forest(sender_id: str, profile: dict) -> IsolationForest:
    """Fit (or reuse a cached) Isolation Forest for one sender."""
    if sender_id not in _forest_cache:
        history = _synthetic_history(profile)
        forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        forest.fit(history)
        _forest_cache[sender_id] = forest
    return _forest_cache[sender_id]


def _z_score(value: float, mean: float, std: float) -> float:
    """Standard Z-score, guarding against a zero standard deviation."""
    std = std or 0.01
    return abs(value - mean) / std


def score_behavior(sender_id: str, current_hour: int, urgency_ratio: float) -> dict:
    """
    Compare the current email against the sender's baseline using both
    Z-score analysis and Isolation Forest. Returns a combined 0-100
    anomaly score.
    """
    profiles = load_profiles()
    profile = profiles.get(sender_id)

    # No baseline exists yet: treat as moderate-high risk by default.
    if profile is None or profile["history_count"] == 0:
        return {
            "score": 55.0,
            "method": "default (no baseline)",
            "reason": "no historical profile for this sender",
        }

    # ---- Z-score component ----
    hour_z = _z_score(current_hour, profile["typical_hour_mean"], profile["typical_hour_std"])
    urgency_z = _z_score(urgency_ratio, profile["urgency_ratio_mean"], profile["urgency_ratio_std"])
    combined_z = (hour_z * 0.4) + (urgency_z * 0.6)
    z_score_result = min(100.0, combined_z * 25)

    # ---- Isolation Forest component ----
    forest = _get_forest(sender_id, profile)
    current_vector = np.array([[current_hour, urgency_ratio]])
    # decision_function: higher = more "normal", negative = more anomalous.
    raw_score = forest.decision_function(current_vector)[0]
    isolation_score = min(100.0, max(0.0, (0.15 - raw_score) * 250))

    # Combine both signals evenly.
    final_score = round((z_score_result * 0.5) + (isolation_score * 0.5), 1)

    return {
        "score": final_score,
        "method": "z-score + isolation forest",
        "hour_z_score": round(hour_z, 2),
        "urgency_z_score": round(urgency_z, 2),
        "isolation_forest_score": round(isolation_score, 1),
    }
