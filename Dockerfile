# Host-agnostic deployment: build this image and run it anywhere that
# runs containers (Render, Railway, Fly.io, a VPS, etc.). See
# README.md "Deploying this" for platform-specific quick starts.

FROM python:3.12-slim

WORKDIR /app

# System deps for scientific Python wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Models must already be trained (models/*.joblib, models/*.keras present).
# If they aren't, uncomment the next line to train them at build time —
# adds several minutes to the build.
# RUN python train_models.py

EXPOSE 5000

# gunicorn, not `python app.py` — a real WSGI server for real traffic.
# Single worker by default since the models are loaded per-process and
# each one holds ~30MB in memory; raise -w if your host has the RAM.
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "--timeout", "60", "app:app"]
