---
title: FocusFlow AI
emoji: 🎯
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Real-time focus and posture monitor (YOLOv11 + MediaPipe)
---

# FocusFlow AI 

> Real-time productivity and posture monitoring using Computer Vision and Deep Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-green)
![YOLO](https://img.shields.io/badge/YOLOv11-Ultralytics-red)
![Flask](https://img.shields.io/badge/Flask-SocketIO-orange)
![License](https://img.shields.io/badge/License-MIT-purple)

FocusFlow AI is a web-based Computer Vision Attention Monitor that uses edge AI to track real-time productivity. It runs in any browser with a webcam: the browser streams camera frames to the server, which runs the AI models and sends back the annotated video and live stats. Frames are processed in memory and never stored; only the session summary (score, alert counts, duration) is saved.

---

## Features

### Adaptive Posture Calibration
Instead of a hardcoded threshold, FocusFlow spends 30 seconds at the start of each session learning *your* specific eye–shoulder gap ratio. It sets a personalised slouch threshold at 82% of your natural baseline — works for any body type, desk height, or camera position.

### Smart Phone Detection
YOLOv11 detects cell phones in real time (confidence ≥ 55%, set via YOLO's own `conf` parameter). On top of that, three custom filters remove false positives from the raw detections:
- **Minimum bounding box area** — rejects boxes under 0.8% of frame area (background noise)
- **Aspect ratio filter** — rejects boxes wider than 2:1 (flat, wide rectangles are usually clothing patterns or objects, not a phone held up)
- **Vertical position filter** — rejects detections in the bottom 10% of the frame (a common source of false positives)

### Distraction-Aware Smart Pomodoro
The Pomodoro timer only counts **verified clean focus time**. Any detected distraction (phone or posture) or absence from frame pauses the timer. One Pomodoro = 25 genuinely productive minutes.

### Focus Score & Session History
A real-time focus score (0–100, starts at 100 and decays/recovers frame by frame) with academic grading (A+ to F). Every session is saved to a local SQLite database. Full session history visible on the dashboard.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| AI — Object Detection | YOLOv11 Nano (COCO dataset, class 67 — "cell phone") |
| AI — Pose Estimation | MediaPipe BlazePose (33 landmarks) |
| Backend | Python, Flask, Flask-SocketIO (`async_mode='threading'`) |
| Frontend | HTML5, CSS3, Vanilla JS, Socket.io |
| Database | SQLite |
| Video | Browser webcam (`getUserMedia`) → JPEG frames over Socket.IO → OpenCV |
| Audio | Web Audio API (chimes play in the browser) |
| Deployment | Docker (Hugging Face Spaces) · gunicorn |

---

## Project Structure

```
FocusFlow_Core/
├── app.py              ← Flask + Flask-SocketIO web server (entry point)
├── engine.py           ← AI engine: all CV models + detection logic
├── database.py         ← SQLite session history (per browser)
├── requirements.txt
├── Dockerfile          ← production image (used by Hugging Face Spaces)
├── focusflow.db        ← auto-created on first run
├── yolo11n.pt          ← YOLO weights (auto-downloaded locally; baked into the Docker image)
├── pose_landmarker_lite.task   ← auto-downloaded on first run (MediaPipe Tasks API only)
└── templates/
    └── index.html      ← dashboard UI
```

> `engine.py` supports both the legacy MediaPipe API (`mp.solutions`, MediaPipe ≤ 0.9.x) and the newer Tasks API (MediaPipe ≥ 0.10) automatically. The `.task` model files are only downloaded if you're on the Tasks API branch.

---

## Deploy (free) — Hugging Face Spaces

The repo is ready to run as a Docker Space. The free CPU tier (2 vCPU, 16 GB RAM) handles a few simultaneous users at roughly 5–10 processed frames per second each.

1. Create a free account at [huggingface.co](https://huggingface.co/join).
2. Go to **New Space** → pick a name (e.g. `focusflow-ai`) → **SDK: Docker** → **Blank** → hardware **CPU basic (free)** → **Create Space**.
3. Push this repo to the Space (use a Hugging Face [access token](https://huggingface.co/settings/tokens) with *write* permission as the password):
   ```bash
   git remote add space https://huggingface.co/spaces/YOUR_HF_USERNAME/focusflow-ai
   git push space main
   ```
4. Wait for the build (~5–10 min the first time). The app is then live at
   `https://YOUR_HF_USERNAME-focusflow-ai.hf.space`.

**Keep the Space in sync with GitHub automatically (optional):** in this GitHub repo go to *Settings → Secrets and variables → Actions* and add a secret `HF_TOKEN` (your write token) and a variable `HF_SPACE` (e.g. `YOUR_HF_USERNAME/focusflow-ai`). Every push to `main` then redeploys via `.github/workflows/sync-to-hf.yml`.

**Notes**
- Webcam access requires HTTPS, which Spaces provides. If the camera prompt doesn't appear on the `huggingface.co/spaces/...` page, open the direct `*.hf.space` URL.
- Free Spaces sleep after 48 h without visitors (the first visit wakes them in about a minute) and their disk is not persistent, so session history resets when the Space restarts. Attach persistent storage and set `FOCUSFLOW_DB=/data/focusflow.db` to keep it.
- `MAX_SESSIONS` (default `4`) limits how many people can run a session at once so the CPU isn't overloaded.

---

## Run Locally

### Prerequisites
- Python 3.10–3.12
- Webcam

### 1. Clone the repository
```bash
git clone https://github.com/AditiJHA-ai/FocusFlow-AI.git
cd FocusFlow-AI
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run
```bash
python app.py
```

### 4. Open in browser
```
http://localhost:5000
```

> **First run note:** YOLO weights (`yolo11n.pt`, ~5 MB) are downloaded automatically on first run if they aren't in the project folder. Internet connection required for this one-time download only.

Or with Docker: `docker build -t focusflow . && docker run -p 7860:7860 focusflow`, then open `http://localhost:7860`.

---

## How to Use

1. Open the app in your browser
2. Click **Begin Session** and allow camera access
3. **Sit naturally for 30 seconds** during calibration — the system learns your posture
4. Monitor yourself! Alerts fire for phone use and slouching
5. Click **End Session** — your session is saved automatically

---

## How the Algorithms Work

### Posture Detection
```python
eye_y      = (lm[1].y + lm[2].y) / 2    # average inner eye Y
shoulder_y = (lm[11].y + lm[12].y) / 2  # average shoulder Y
gap        = shoulder_y - eye_y           # normalised vertical distance

# After 30s calibration (needs ≥10 valid samples, else falls back to a default 0.26):
posture_threshold = mean(calibration_samples) * 0.82
if gap < posture_threshold:
    # SLOUCH DETECTED
```

### Focus Score (per frame, at a 30 fps reference)
| Event | Score Change |
|-------|-------------|
| Phone detected | −0.4 |
| Posture alert | −0.2 |
| User away (no pose detected) | −0.1 |
| Clean focus | +0.05 (capped at 100) |

Score is clamped to the 0–100 range every frame. Because frames arrive at whatever rate the server can process, each change is scaled by the real time elapsed since the previous frame (so the score moves at the same speed at 8 fps as at 30 fps). An alert is counted, and chimes, once when it starts, not on every frame it stays active.

### Smart Pomodoro
A frame only counts as "clean" and advances the Pomodoro clock when status is `PRODUCTIVE` with no active phone or posture alert. Once 25 minutes of clean time accumulate, a Pomodoro completes, the counter increments, and the session enters a `break` phase.

---

## Inventive Steps (Patent Strategy)

| Feature | Novelty |
|---------|---------|
| Adaptive Calibration | Per-user biometric threshold normalisation (30s baseline × 0.82) — no existing tool does this |
| Distraction-Aware Pomodoro | Timer gated by the live CV pipeline (phone/posture/absence), not wall-clock time |
| Dual MediaPipe API support | Runtime fallback between MediaPipe's legacy `solutions` API and the newer Tasks API, so the same codebase works across MediaPipe versions without a hard pin |

> A gaze-estimation helper (`estimate_gaze_point`) using iris landmarks is present in `engine.py` but is **not currently called** anywhere in the active detection loop — it doesn't yet affect status, scoring, or alerts. Treat it as scaffolding for a future "gaze-correlated detection" feature, not a shipped one.

---

## Known Limitations

- Processing runs server-side on CPU, so frame rate per user drops as more people use the app at once.
- Skeleton drawing only runs on the legacy MediaPipe API branch; the Tasks API branch doesn't draw the pose skeleton (would need a custom drawing implementation).
- Face Mesh is not run (it fed no decision); re-enable it when the gaze-estimation path is connected.
- No drowsiness/eye-closure detection exists yet — it would build on that same Face Mesh pipeline.

---

## References

- Lin, T.Y. et al. (2014). Microsoft COCO: Common Objects in Context. *ECCV 2014.*
- Ultralytics. (2025). YOLOv11 Documentation. https://docs.ultralytics.com
- Google MediaPipe. (2025). Tasks Vision API. https://developers.google.com/mediapipe

---

## Author

**Aditi Jha** — B.Tech CSE (AIML), Manipal University Jaipur  
**Akshita Sai Pery** — B.Tech CSE, Manipal University Jaipur
