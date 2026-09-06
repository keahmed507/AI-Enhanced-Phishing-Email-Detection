# AI-Enhanced Phishing Email Detection — Backend

A Flask API implementing the detection pipeline described in the
project report: ingest → extract features → classify content +
profile sender behavior → fuse into one decision. Trained on real,
public datasets — see below.

## Setup

```bash
cd backend
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Windows + Python 3.13:** if `pip install` fails trying to compile
`scikit-learn` from source (a `meson`/`vswhere.exe` error), it's because
older scikit-learn releases don't ship a Windows wheel for Python 3.13.
`requirements.txt` already pins versions with 3.13 wheels
(`scikit-learn>=1.6.0`, `numpy>=2.1`, `pandas>=2.2.3,<3.0`) — this was
tested against the actual trained model files in `models/` to confirm
they still load and predict correctly under the newer versions (you'll
see a harmless `InconsistentVersionWarning` the first time each model
loads, since it was trained with an older scikit-learn point release —
predictions are unaffected). If you still hit a build error, you're
likely on an even older `requirements.txt` than this one — re-download
it, or fall back to installing Python 3.12 instead.

### 1. Get the data

```bash
cd data
curl -L -o enron_spam_data.zip https://raw.githubusercontent.com/MWiechmann/enron_spam_data/master/enron_spam_data.zip
unzip enron_spam_data.zip
curl -L -o uci_phishing_raw.csv https://raw.githubusercontent.com/Danish-lakhwani/Phishing-Detection-Using-Machine-Learning/main/Phishing%20dataset.csv
cd ..
```

(If `data/enron_spam_data.csv` and `data/uci_phishing_raw.csv` are
already present — they're included in this delivery — skip this step.)

### 2. Train the models

```bash
python train_models.py
```

This trains all five models and writes them to `models/` along with
`models/metrics.json`. It takes a few minutes — the LSTM step is the
slow part (~45s on a single CPU core; faster with a GPU). If `models/`
is already populated (also included in this delivery), you can skip
this and go straight to running the API.

### 3. Run the API

```bash
python app.py
```

The API starts at `http://127.0.0.1:5000`. Leave it running, then open
`index.html` in a browser — the page detects the backend automatically
(look for "backend connected" in the console header) and routes the
"Run classification" button through it, with a model dropdown to pick
which of the four classifiers scores the email.

## Datasets used

| Dataset | Size | Used for |
|---|---|---|
| [Enron-Spam](https://github.com/MWiechmann/enron_spam_data) (cleaned CSV, from Metsis, Androutsopoulos & Paliouras) | 33,345 real emails, 16,852 spam / 16,493 ham | Content classifier (LR, RF, SVM, LSTM) |
| [UCI Phishing Websites Dataset](https://archive.ics.uci.edu/ml/datasets/phishing+websites) (via GitHub mirror) | 10,000 labeled URLs, 16 lexical/host features | Secondary URL risk model |

Spam is used as the phishing-class proxy for the content classifier,
since Enron doesn't carry a separate phishing label — this is the
standard substitution used in the spam/phishing detection literature
this project cites.

## Real evaluation results

Trained on an 80/20 stratified split (26,676 train / 6,669 test
emails), evaluated on the held-out test set:

| Model | Accuracy | Precision | Recall | F1 | AUC | Train time |
|---|---|---|---|---|---|---|
| **SVM** (linear, calibrated) | **98.89%** | 98.64% | 99.17% | **98.91%** | **99.91%** | 0.5s |
| Logistic Regression | 98.65% | 97.95% | 99.41% | 98.67% | 99.85% | 0.1s |
| Random Forest | 97.56% | 95.70% | 99.64% | 97.63% | 99.79% | 16s |
| LSTM | 91.74% | 96.38% | 86.91% | 91.40% | 96.11% | 44s |
| URL model (UCI data) | 86.75% | 92.49% | 80.00% | 85.79% | 92.56% | 0.4s |

SVM edges out Logistic Regression, but LR gets within half a point at
roughly 1/5th the training time — a genuine speed/accuracy trade-off,
not a fluke of this dataset. The LSTM trained from scratch for only 2
epochs on a single CPU core in this environment; more epochs, a larger
embedding, or pretrained word vectors would likely close the gap. This
matches the general pattern in the literature — classical ML on TF-IDF
is a strong, cheap baseline for spam/phishing text, and deep learning
tends to pull ahead only with more data, more compute, or when the
task depends on longer-range context than TF-IDF captures.

The URL model scores lower because several of the UCI dataset's
original 16 features (`Domain_Age`, `Web_Traffic`, `DNS_Record`,
`iFrame`, `Mouse_Over`, `Right_Click`, `Web_Forwards`) require a live
WHOIS/DNS lookup or a fetched page to compute. This offline API can't
fetch arbitrary user-submitted URLs, so at inference time those
features are defaulted to 0 and only the URL-string-derivable features
(IP address, @ symbol, length, path depth, HTTPS placement, shortener,
hyphenated domain) are used — see `url_model.py` for the exact
breakdown. The 86.75% above is the model's *training-time* accuracy
with all 16 real features; expect lower real-world accuracy from the
API's reduced feature set.

## Endpoints

### `POST /api/analyze`

Request:

```json
{
  "email_text": "Click here immediately to verify your account.",
  "sender_id": "known_hr",
  "hour": 14,
  "model": "logistic_regression"
}
```

`sender_id`, `hour`, and `model` are optional. `sender_id` defaults to
`"unknown"`, `hour` to the server's current hour, `model` to
`"logistic_regression"`. `model` accepts `logistic_regression`,
`random_forest`, `svm`, or `lstm`.

Response:

```json
{
  "verdict": "phishing",
  "final_score": 100.0,
  "content": { "score": 96.8, "label": "phishing", "model_used": "logistic_regression" },
  "behavior": { "score": 55.0, "method": "default (no baseline)", "reason": "no historical profile for this sender" },
  "url_risk": [{ "score": 100.0, "url": "http://192.168.44.10/secure-verify-login", "note": "..." }],
  "features": { "urgency_hits": ["urgent", "immediately", "verify"], "urgency_ratio": 0.5, "url_count": 1, "urls": [...], "has_ip_url": true, "exclamation_count": 1 }
}
```

### `GET /api/senders`

Returns the known sender baseline profiles (used to populate the
sender dropdown on the frontend).

### `GET /api/models`

Returns the available classifier names plus the real metrics from
`models/metrics.json`.

## File layout

| File | Role |
|---|---|
| `app.py` | Flask app, request handling, wires the pipeline together |
| `feature_extraction.py` | **Feature Extraction Module** — pulls urgency/URL/exclamation signals from raw text |
| `classifier.py` | **Machine Learning Classifier** — loads the 4 trained models, scores with the requested one |
| `behavior_engine.py` | **Sender Behavior Profiling Engine** — Z-score + Isolation Forest anomaly scoring |
| `url_model.py` | Secondary URL risk model, trained on the real UCI dataset |
| `fusion.py` | Decision fusion — combines content, URL, and behavior scores into one verdict |
| `train_models.py` | Trains all 5 models from the real datasets and writes `models/metrics.json` |
| `data/enron_spam_data.csv` | Real Enron-Spam dataset (33,345 emails) |
| `data/uci_phishing_raw.csv` | Real UCI Phishing Websites dataset (10,000 URLs) |
| `data/sender_profiles.json` | Hand-written sender baselines (frequency, typical hour, urgency ratio) |
| `models/` | Trained model artifacts + `metrics.json` (produced by `train_models.py`) |

## What's still simulated (read this before presenting results as final)

Being upfront about what's real vs. approximated in this delivery:

- **Sender behavior baselines are synthetic.** No public dataset pairs
  real per-sender, timestamped email history with account-compromise
  labels — Enron's raw maildir has timestamps but no such labels, and
  nothing else reachable from this environment fills that gap. Each
  sender's baseline (`data/sender_profiles.json`) is a small
  hand-set distribution; `behavior_engine.py` draws a synthetic sample
  from it to fit a real Isolation Forest, standing in for months of
  real traffic. The *algorithm* is real and untouched — the *data it
  trains on* is simulated. Swapping in real per-sender logs (from an
  actual mail server, with consent) means replacing
  `_synthetic_history()` with a real feature extraction step over that
  history — everything downstream stays the same.
- **The URL model's live-lookup features are stubbed to 0** at
  inference time, per the "Real evaluation results" section above.
- **The LSTM is undertrained relative to what a real deployment would
  use** — 2 epochs, a 32-dim embedding, single CPU core. It's a real,
  independently-trained model on real data, just a smaller one than
  the report's target architecture would justify given more compute.

## Production hardening — done

Everything below used to be a gap; it's now handled in `app.py`,
configured via environment variables (see `.env.example`):

| Concern | How it's handled |
|---|---|
| Flask debugger exposed | `FLASK_DEBUG` defaults to `0`. Only set to `1` on your own machine. |
| Dev server under real traffic | Deploy behind gunicorn (`Dockerfile` does this), not `python app.py`. |
| CORS wide open | `ALLOWED_ORIGINS` restricts which frontend origins can call the API. Defaults to `*` for local demo use — set it before deploying. |
| No authentication | `API_KEY`, if set, requires a matching `X-API-Key` header on `/api/analyze`. Unset by default (local demo mode). |
| No rate limiting | `RATE_LIMIT` (default `30/minute` per IP) via Flask-Limiter. |
| Unbounded request size | `MAX_CONTENT_LENGTH` caps the request body at 200 KB; `email_text` is separately capped at 50,000 characters. |
| No liveness endpoint | `GET /api/health` for uptime monitors / hosting platform health checks. |

Two things still worth knowing:

- **Model file integrity isn't checked.** Not a live risk for a
  single-tenant deployment you control, but if this ever runs
  somewhere multi-tenant, verify the model files haven't been swapped
  before loading them.
- **Flask-Limiter's default storage is in-memory**, which means rate
  limits reset if the process restarts and don't share state across
  multiple worker processes/replicas. Fine for one gunicorn worker on
  one instance; if you scale to multiple workers or instances, point
  `Limiter` at Redis instead (see the
  [Flask-Limiter docs](https://flask-limiter.readthedocs.io) on
  configuring a storage backend).

## Deploying this

The `Dockerfile` in this folder is host-agnostic — it builds a
container that runs the API behind gunicorn. How you get that
container running publicly depends on where you deploy:

### Any Docker host (Render, Railway, Fly.io, a VPS, etc.)

```bash
docker build -t phishing-api .
docker run -p 5000:5000 \
  -e ALLOWED_ORIGINS="https://your-frontend-domain.com" \
  -e API_KEY="pick-a-long-random-string" \
  -e RATE_LIMIT="30/minute" \
  phishing-api
```

Most platforms (Render, Railway, Fly.io) let you point them at this
repo, detect the `Dockerfile` automatically, and set the same
environment variables through their dashboard — no code changes needed.

### Bare VPS (no Docker)

```bash
git clone <this repo> && cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# models/ already included — skip train_models.py unless you want to retrain
export ALLOWED_ORIGINS="https://your-frontend-domain.com"
export API_KEY="pick-a-long-random-string"
gunicorn -w 2 -b 0.0.0.0:5000 --timeout 60 app:app
```

Put this behind a reverse proxy (nginx/Caddy) for TLS — gunicorn itself
doesn't terminate HTTPS.

### After deploying

Update `BACKEND_URL` near the top of the `<script>` tag in `index.html`
from `http://127.0.0.1:5000` to your deployed URL, and add
`X-API-Key: <your key>` to the `fetch` headers in `analyzeWithBackend()`
if you set `API_KEY`. Both are marked with a comment in the file.
