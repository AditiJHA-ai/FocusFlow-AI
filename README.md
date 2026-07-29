# FocusFlow AI 

> Real-time productivity and posture monitoring using Computer Vision and Deep Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-green)
![YOLO](https://img.shields.io/badge/YOLOv11-Ultralytics-red)
![Flask](https://img.shields.io/badge/Flask-SocketIO-orange)
![License](https://img.shields.io/badge/License-MIT-purple)

FocusFlow AI is a web-based Computer Vision Attention Monitor that uses edge AI to track real-time productivity. It runs as a local web app on any laptop with a webcam — no cloud, no data leaves your machine.

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
| AI — Face Mesh | MediaPipe Face Mesh (468 + iris landmarks) — currently loaded per frame but only feeds a dormant gaze-estimation helper, see *Known Limitations* |
| Backend | Python, Flask, Flask-SocketIO (`async_mode='threading'`) |
| Frontend | HTML5, CSS3, Vanilla JS, Socket.io |
| Database | SQLite |
| Video | OpenCV (MJPEG stream) |
| Audio | winsound (Windows) |

---

## Project Structure

```
FocusFlow_Core/
├── app.py              ← Flask + Flask-SocketIO web server (entry point)
├── engine.py           ← AI engine: all CV models + detection logic
├── database.py         ← SQLite session history
├── focusflow.db        ← auto-created on first run
├── yolo11n.pt          ← YOLO weights (download separately)
├── pose_landmarker_lite.task   ← auto-downloaded on first run (MediaPipe Tasks API only)
├── face_landmarker.task        ← auto-downloaded on first run (MediaPipe Tasks API only)
└── templates/
    └── index.html      ← dashboard UI
```

> `engine.py` supports both the legacy MediaPipe API (`mp.solutions`, MediaPipe ≤ 0.9.x) and the newer Tasks API (MediaPipe ≥ 0.10) automatically. The `.task` model files are only downloaded if you're on the Tasks API branch.

---

## Setup & Installation

### Prerequisites
- Python 3.10 or higher
- Windows (for audio alerts — see note below)
- Webcam

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/FocusFlow-AI.git
cd FocusFlow-AI
```

### 2. Install dependencies
```bash
pip install flask flask-socketio opencv-python mediapipe ultralytics numpy
```

### 3. Download YOLO weights
Download `yolo11n.pt` from [Ultralytics releases](https://github.com/ultralytics/assets/releases) and place it in the project folder.

### 4. Run
```bash
python app.py
```

### 5. Open in browser
```
http://localhost:5000
```

> **First run note:** if you're on MediaPipe's Tasks API (≥ 0.10), the app will automatically download two model files (~35MB total) on first run. Internet connection required for this one-time download only.

---

## How to Use

1. Open `http://localhost:5000` in your browser
2. Click **Begin Session**
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

### Focus Score (per frame)
| Event | Score Change |
|-------|-------------|
| Phone detected | −0.4 |
| Posture alert | −0.2 |
| User away (no pose detected) | −0.1 |
| Clean focus | +0.05 (capped at 100) |

Score is clamped to the 0–100 range every frame.

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

- Audio alerts use `winsound` (Windows only). Replace with `pygame.mixer` for cross-platform.
- Single user per instance — one webcam session (`cv2.VideoCapture(0)`) and one global state at a time.
- Skeleton drawing only runs on the legacy MediaPipe API branch; the Tasks API branch doesn't draw the pose skeleton (would need a custom drawing implementation).
- Face Mesh is computed every frame but isn't wired into any decision — it's currently unused compute overhead until the gaze-estimation path is connected.
- No drowsiness/eye-closure detection exists yet, despite it being a natural extension of the existing Face Mesh pipeline.

---

## References

- Lin, T.Y. et al. (2014). Microsoft COCO: Common Objects in Context. *ECCV 2014.*
- Ultralytics. (2025). YOLOv11 Documentation. https://docs.ultralytics.com
- Google MediaPipe. (2025). Tasks Vision API. https://developers.google.com/mediapipe

---

## Author

**Aditi Jha** — B.Tech CSE (AIML), Manipal University Jaipur  
**Akshita Sai Pery** — B.Tech CSE, Manipal University Jaipur
