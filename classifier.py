"""
Machine Learning Classifier
-----------------------------
Loads the four trained content classifiers described in the report
(Section 4.9.1): Logistic Regression, Random Forest, SVM, and LSTM.
All four were trained on the real Enron-Spam dataset (33,345 emails)
by train_models.py — run that script first to produce the files this
module loads from /models.

Logistic Regression is used as the default, since it had the best
accuracy/speed trade-off in evaluation (see models/metrics.json), but
any of the four can be selected per-request through the API.
"""

import os

import joblib

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

_TFIDF_MODELS = ("logistic_regression", "random_forest", "svm")
DEFAULT_MODEL = "logistic_regression"
AVAILABLE_MODELS = ("logistic_regression", "random_forest", "svm", "lstm")

# Lazily populated on first use, so the module can be imported (e.g. for
# tests) without needing all five model files to already exist on disk.
_cache = {}


def _load(name, loader):
    if name not in _cache:
        _cache[name] = loader()
    return _cache[name]


def _tfidf_vectorizer():
    return _load("vectorizer", lambda: joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")))


def _sklearn_model(name):
    return _load(name, lambda: joblib.load(os.path.join(MODELS_DIR, f"{name}.joblib")))


def _lstm_model():
    def loader():
        # Imported lazily: TensorFlow is a heavy dependency and only
        # needed if the caller actually asks for the LSTM model.
        from tensorflow.keras.models import load_model
        return load_model(os.path.join(MODELS_DIR, "lstm_model.keras"))
    return _load("lstm_model", loader)


def _lstm_tokenizer():
    return _load("lstm_tokenizer", lambda: joblib.load(os.path.join(MODELS_DIR, "lstm_tokenizer.joblib")))


def classify_content(email_text: str, model_name: str = DEFAULT_MODEL) -> dict:
    """
    Score a single email's text content with the requested model.
    Returns a phishing probability on a 0-100 scale plus a label.
    """
    if model_name not in AVAILABLE_MODELS:
        model_name = DEFAULT_MODEL

    if model_name in _TFIDF_MODELS:
        vectorizer = _tfidf_vectorizer()
        model = _sklearn_model(model_name)
        features = vectorizer.transform([email_text])
        phishing_probability = model.predict_proba(features)[0][1]
    else:
        # LSTM path: tokenize + pad to the same length used in training.
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        tokenizer = _lstm_tokenizer()
        model = _lstm_model()
        sequence = tokenizer.texts_to_sequences([email_text])
        padded = pad_sequences(sequence, maxlen=120, padding="post", truncating="post")
        phishing_probability = float(model.predict(padded, verbose=0)[0][0])

    return {
        "score": round(float(phishing_probability) * 100, 1),
        "label": "phishing" if phishing_probability >= 0.5 else "legitimate",
        "model_used": model_name,
    }
