# FocusFlow AI 

> Real-time productivity and posture monitoring using Computer Vision and Deep Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-green)
![YOLO](https://img.shields.io/badge/YOLOv11-Ultralytics-red)
![Flask](https://img.shields.io/badge/Flask-SocketIO-orange)
![License](https://img.shields.io/badge/License-MIT-purple)

FocusFlow AI is a web-based Computer Vision Attention Monitor that uses edge AI to track real-time productivity. It runs as a local web app on any laptop with a webcam — no cloud, no data leaves your machine.

---

##  Features

###  Adaptive Posture Calibration
Instead of a hardcoded threshold, FocusFlow spends 30 seconds at the start of each session learning *your* specific eye–shoulder gap ratio. It sets a personalised slouch threshold at 82% of your natural baseline — works for any body type, desk height, or camera position.

###  Smart Phone Detection
YOLOv11 detects cell phones in real time with three custom filters to eliminate false positives:
- Confidence threshold ≥ 55%
- Minimum bounding box area (eliminates background noise)
- Aspect ratio filter (rejects clothing patterns that look like phones)

###  Drowsiness Detection (Eye Aspect Ratio)
Uses MediaPipe Face Mesh iris landmarks to compute EAR (Eye Aspect Ratio) in real time. Distinguishes normal blinks from drowsiness using a consecutive-frame debounce. Fires alert after 4 seconds of sustained eye closure.

```
EAR = (||p2−p6|| + ||p3−p5||) / (2 × ||p1−p4||)
```

###  Distraction-Aware Smart Pomodoro
The Pomodoro timer only counts **verified clean focus time**. Any detected distraction (phone, posture, drowsiness, absence) pauses the timer. One Pomodoro = 25 genuinely productive minutes.

###  Focus Score & Session History
A real-time focus score (0–100) with academic grading (A+ to F). Every session is saved to a local SQLite database. Full session history visible on the dashboard.

---

##  Tech Stack

| Layer | Technology |
|-------|-----------|
| AI — Object Detection | YOLOv11 Nano (COCO dataset, class 67) |
| AI — Pose Estimation | MediaPipe BlazePose (33 landmarks) |
| AI — Face Mesh | MediaPipe Face Mesh (468 + iris landmarks) |
| Backend | Python, Flask, Flask-SocketIO |
| Frontend | HTML5, CSS3, Vanilla JS, Socket.io |
| Database | SQLite |
| Video | OpenCV (MJPEG stream) |
| Audio | winsound (Windows) |

---

##  Project Structure

```
FocusFlow_Core/
├── app.py              ← Flask web server (entry point)
├── engine.py           ← AI engine: all CV models + detection logic
├── database.py         ← SQLite session history
├── focusflow.db        ← auto-created on first run
├── yolo11n.pt          ← YOLO weights (download separately)
├── pose_landmarker_lite.task   ← auto-downloaded on first run
├── face_landmarker.task        ← auto-downloaded on first run
└── templates/
    └── index.html      ← dashboard UI
```

---

##  Setup & Installation

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
pip install flask flask-socketio eventlet opencv-python mediapipe ultralytics numpy
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

> **First run note:** MediaPipe will automatically download two model files (~35MB total) on the first run. Internet connection required for this one-time download only.

---

##  How to Use

1. Open `http://localhost:5000` in your browser
2. Click **Begin Session**
3. **Sit naturally for 30 seconds** during calibration — the system learns your posture
4. Monitor yourself! Alerts fire for phone use, slouching, and drowsiness
5. Click **End Session** — your session is saved automatically

---

##  How the Algorithms Work

### Posture Detection
```python
eye_y      = (lm[1].y + lm[2].y) / 2    # average inner eye Y
shoulder_y = (lm[11].y + lm[12].y) / 2  # average shoulder Y
gap        = shoulder_y - eye_y           # normalised vertical distance

# After calibration:
posture_threshold = mean(calibration_samples) * 0.82
if gap < posture_threshold:
    # SLOUCH DETECTED
```

### EAR Drowsiness Detection
```python
# Left eye landmarks: 33, 160, 158, 133, 153, 144
# Right eye landmarks: 362, 385, 387, 263, 373, 380

EAR = (dist(p2,p6) + dist(p3,p5)) / (2 * dist(p1,p4))
# Open eye: EAR ≈ 0.25-0.35
# Closed eye: EAR ≈ 0.0
# Alert fires after 4 seconds below EAR threshold (0.20)
```

### Focus Score
| Event | Score Change |
|-------|-------------|
| Phone detected | −0.4 per frame |
| Drowsy | −0.3 per frame |
| Posture alert | −0.2 per frame |
| User away | −0.1 per frame |
| Clean focus | +0.05 per frame |

---

##  Inventive Steps (Patent Strategy)

| Feature | Novelty |
|---------|---------|
| Adaptive Calibration | Per-user biometric threshold normalisation — no existing tool does this |
| Gaze-Correlated Detection | Correlates object presence with iris gaze vector (implemented, toggleable) |
| Distraction-Aware Pomodoro | Timer gated by CV pipeline — not wall-clock time |
| EAR in Productivity Context | Combines driver-monitoring technique with academic grading system |

---

##  Known Limitations

- Audio alerts use `winsound` (Windows only). Replace with `pygame.mixer` for cross-platform.
- Single user per instance — one webcam session at a time.
- Skeleton drawing disabled in MediaPipe 0.10+ (Tasks API). Custom drawing implementation needed.
- Glasses can cause minor EAR drift — consider raising `EAR_THRESHOLD` to 0.22.

---

## 📚 References

- Soukupova, T. & Cech, J. (2016). Real-Time Eye Blink Detection using Facial Landmarks. *21st Computer Vision Winter Workshop.*
- Lin, T.Y. et al. (2014). Microsoft COCO: Common Objects in Context. *ECCV 2014.*
- Ultralytics. (2025). YOLOv11 Documentation. https://docs.ultralytics.com
- Google MediaPipe. (2025). Tasks Vision API. https://developers.google.com/mediapipe

---

##  Author

**Aditi Jha** — B.Tech CSE (AIML), Manipal University Jaipur  
**Akshita Sai Pery** — B.Tech CSE, Manipal University Jaipur  

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.
