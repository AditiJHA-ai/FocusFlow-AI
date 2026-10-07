import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import os
import time
import threading

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Audio alerts are played in the browser (Web Audio) — the server has no speakers.
# The browser chimes whenever phone_alerts / posture_alerts / pomodoros_done increase.

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
PHONE_CONFIDENCE    = 0.55     # raised from 0.4 — reduces clothing false positives
PHONE_MIN_AREA      = 0.008    # phone box must be at least 0.8% of frame area
CALIBRATION_SECS    = 30
POMODORO_FOCUS_SECS = 25 * 60

# Frames arrive from the browser at whatever rate the server can process (often
# well under 30 fps on a CPU host), so per-frame score changes and alert hold
# times are scaled by real elapsed time, expressed relative to a 30 fps reference.
REF_FPS             = 30
ALERT_HOLD_SECS     = 15 / REF_FPS   # was a 15-frame buffer at ~30 fps
MAX_FRAME_DT        = 1.0            # cap on time credited to a single frame

MODEL_DIR = os.environ.get("FOCUSFLOW_MODEL_DIR", os.path.dirname(os.path.abspath(__file__)))
YOLO_WEIGHTS = os.path.join(MODEL_DIR, "yolo11n.pt")


# ── SHARED MODELS (loaded once, used by every session) ────────────────────────
_models_lock  = threading.Lock()
_models_ready = threading.Event()
_yolo_model   = None
_yolo_lock    = threading.Lock()
USE_LEGACY_MP = None


def _load_models():
    global _yolo_model, USE_LEGACY_MP
    with _models_lock:
        if _models_ready.is_set():
            return
        print("[engine] loading models...")

        # ── MediaPipe compatibility: 0.9.x uses mp.solutions, 0.10+ uses Tasks API ──
        try:
            mp.solutions.pose
            USE_LEGACY_MP = True
            print("[engine] using MediaPipe legacy API (solutions)")
        except AttributeError:
            import urllib.request
            pose_model = os.path.join(MODEL_DIR, "pose_landmarker_lite.task")
            if not os.path.exists(pose_model):
                print("[engine] downloading pose_landmarker_lite.task ...")
                urllib.request.urlretrieve(
                    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
                    pose_model
                )
            USE_LEGACY_MP = False
            print("[engine] using MediaPipe Tasks API (>= 0.10)")

        _yolo_model = YOLO(YOLO_WEIGHTS if os.path.exists(YOLO_WEIGHTS) else "yolo11n.pt")
        # Warm-up so the first real frame isn't slow
        _yolo_model.predict(np.zeros((480, 640, 3), dtype=np.uint8), classes=[67], verbose=False)
        _models_ready.set()
        print("[engine] models loaded")


def preload_models_async():
    threading.Thread(target=_load_models, daemon=True).start()


def _create_pose():
    if USE_LEGACY_MP:
        return mp.solutions.pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=os.path.join(MODEL_DIR, "pose_landmarker_lite.task")),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.7,
        min_pose_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    return mp_vision.PoseLandmarker.create_from_options(opts)


# ── GAZE ESTIMATION (Inventive Step 2 helper) ─────────────────────────────────
def estimate_gaze_point(face_lm):
    """
    Works with both:
    - Legacy API: NormalizedLandmarkList (has .landmark[n])
    - Tasks API:  plain list of NormalizedLandmark
    """
    # Normalise to plain list access
    lms = face_lm.landmark if hasattr(face_lm, 'landmark') else face_lm
    try:
        l_iris  = lms[468]
        r_iris  = lms[473]
        nose    = lms[1]
        iris_mx = (l_iris.x + r_iris.x) / 2
        iris_my = (l_iris.y + r_iris.y) / 2
        dx = nose.x - iris_mx
        dy = nose.y - iris_my
        return iris_mx - dx * 1.5, iris_my - dy * 1.5
    except Exception:
        try:
            nose = lms[1]
            return nose.x, nose.y
        except Exception:
            return 0.5, 0.5


# ── CALIBRATION OVERLAY ───────────────────────────────────────────────────────
def _draw_calibration(frame, progress, sample_count):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (10, 10, 15), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    bar_x, bar_y, bar_w_max, bar_h = 60, h // 2 + 30, w - 120, 14
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w_max, bar_y + bar_h), (40, 40, 50), -1)
    filled = int(progress * bar_w_max)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h), (0, 220, 140), -1)

    cv2.putText(frame, "CALIBRATING YOUR POSTURE BASELINE",
                (bar_x, h // 2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 140), 2)
    cv2.putText(frame, "Sit naturally and look at your screen",
                (bar_x, h // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
    cv2.putText(frame, f"{int(progress * 100)}%  ({sample_count} samples)",
                (bar_x, h // 2 + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


# ── PER-VISITOR SESSION ───────────────────────────────────────────────────────
class Session:
    """One monitoring session for one browser. Frames are captured by the
    visitor's webcam, sent in by the web layer, processed, and returned annotated."""

    def __init__(self):
        _load_models()
        self._lock = threading.Lock()
        self.pose  = _create_pose()

        # Session counters
        self.start_time           = time.time()
        self.last_frame_time      = self.start_time
        self.phone_timer          = 0.0
        self.attention_timer      = 0.0
        self.phone_alerts_count   = 0
        self.posture_alerts_count = 0
        self.focus_score          = 100.0
        self.prev_status          = None

        # Inventive Step 1 — Adaptive Calibration
        self.calib_samples     = []
        self.calibrated        = False
        self.posture_threshold = 0.26        # default until calibration completes

        # Inventive Step 3 — Smart Pomodoro
        self.clean_focus_secs = 0.0
        self.pomodoros_done   = 0
        self.phase            = "calibrating"

        self.state = {
            "focus_score":      100.0,
            "status":           "CALIBRATING",
            "phase":            "calibrating",
            "phone_alerts":     0,
            "posture_alerts":   0,
            "session_mins":     0.0,
            "pomodoro_secs":    0,
            "pomodoro_target":  POMODORO_FOCUS_SECS,
            "pomodoros_done":   0,
            "calibration_pct":  0,
            "posture_baseline": None,
            "running":          True,
        }

    def process(self, jpeg_bytes):
        """Process one JPEG frame. Returns (annotated_jpeg_bytes or None, state)."""
        frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
        with self._lock:
            if frame is None or not self.state["running"]:
                return None, dict(self.state)
            frame = self._process(frame)
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            return buf.tobytes(), dict(self.state)

    def _process(self, frame):
        now                  = time.time()
        frame_dt             = min(now - self.last_frame_time, MAX_FRAME_DT)
        self.last_frame_time = now
        elapsed              = now - self.start_time
        rate                 = frame_dt * REF_FPS   # this frame's weight vs. one 30 fps frame

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── Run models (API-compatible) ──────────────────────────────────────
        if USE_LEGACY_MP:
            pose_lms = self.pose.process(rgb).pose_landmarks      # NormalizedLandmarkList or None
        else:
            _mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            _pose_res = self.pose.detect(_mp_img)
            pose_lms  = _pose_res.pose_landmarks[0] if _pose_res.pose_landmarks else None

        with _yolo_lock:
            yolo_res = _yolo_model.predict(frame, classes=[67], conf=PHONE_CONFIDENCE, verbose=False)

        # ── CALIBRATION PHASE (Inventive Step 1) ─────────────────────────────
        if not self.calibrated:
            calib_progress = min(elapsed / CALIBRATION_SECS, 1.0)

            if pose_lms:
                lm    = pose_lms.landmark if hasattr(pose_lms, 'landmark') else pose_lms
                eye_y = (lm[1].y + lm[2].y) / 2
                sh_y  = (lm[11].y + lm[12].y) / 2
                gap   = sh_y - eye_y
                if gap > 0.05:
                    self.calib_samples.append(gap)

            if elapsed >= CALIBRATION_SECS:
                if len(self.calib_samples) >= 10:
                    baseline               = float(np.mean(self.calib_samples))
                    self.posture_threshold = baseline * 0.82   # 18% below natural = clear slouch
                    print(f"[engine] calibrated: baseline={baseline:.3f}, threshold={self.posture_threshold:.3f}")
                else:
                    print("[engine] not enough samples — using default threshold 0.26")
                self.calibrated = True
                self.phase      = "focusing"
                self.state["posture_baseline"] = round(self.posture_threshold, 3)

            _draw_calibration(frame, calib_progress, len(self.calib_samples))
            self.state.update({
                "status":          "CALIBRATING",
                "phase":           self.phase,
                "calibration_pct": int(calib_progress * 100),
                "session_mins":    round(elapsed / 60, 1),
            })
            return frame

        # ── ACTIVE MONITORING ─────────────────────────────────────────────────
        current_status = "PRODUCTIVE"

        if not pose_lms:
            current_status = "USER AWAY"
        else:
            lm = pose_lms.landmark if hasattr(pose_lms, 'landmark') else pose_lms

            # ── Phone Detection (direct YOLO — phone visible = distraction) ──
            phone_detected = False
            if len(yolo_res[0].boxes) > 0:
                for box in yolo_res[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    fw, fh = frame.shape[1], frame.shape[0]
                    conf = float(box.conf[0])

                    box_w = (x2 - x1) / fw
                    box_h = (y2 - y1) / fh
                    area  = box_w * box_h

                    # Filter 1: must be a reasonable size
                    if area < PHONE_MIN_AREA:
                        continue

                    # Filter 2: aspect ratio — a phone is taller or roughly square,
                    # never a very wide flat rectangle (that's usually clothing detail)
                    aspect = box_w / max(box_h, 0.001)
                    if aspect > 2.0:   # too wide = not a phone
                        continue

                    # Filter 3: phone must be in upper 90% of frame
                    if y1 >= (fh * 0.9):
                        continue

                    phone_detected = True
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                  (60, 60, 255), 2)
                    cv2.putText(frame, f"PHONE {conf:.0%}",
                                (int(x1), int(y1) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 255), 1)

            if phone_detected:
                self.phone_timer = ALERT_HOLD_SECS

            # ── Posture (personalised threshold) ─────────────────────────────
            eye_y      = (lm[1].y + lm[2].y) / 2
            shoulder_y = (lm[11].y + lm[12].y) / 2
            gap        = shoulder_y - eye_y
            if gap < self.posture_threshold:
                self.attention_timer = ALERT_HOLD_SECS

            # Draw skeleton (legacy API only — new API needs different drawing)
            if USE_LEGACY_MP:
                try:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, pose_lms, mp.solutions.pose.POSE_CONNECTIONS)
                except Exception:
                    pass

        # ── ALERTS & SCORE ────────────────────────────────────────────────────
        # An alert is counted (and chimed in the browser) once when it starts,
        # not on every frame it stays active.
        frame_is_clean = False

        if current_status == "PRODUCTIVE":
            if self.phone_timer > 0:
                current_status = "PHONE DETECTED"
                if self.prev_status != current_status:
                    self.phone_alerts_count += 1
                self.phone_timer -= frame_dt
                self.focus_score -= 0.4 * rate
            elif self.attention_timer > 0:
                current_status = "POSTURE ALERT"
                if self.prev_status != current_status:
                    self.posture_alerts_count += 1
                self.attention_timer -= frame_dt
                self.focus_score     -= 0.2 * rate
            else:
                self.focus_score = min(100.0, self.focus_score + 0.05 * rate)
                frame_is_clean   = True
        elif current_status == "USER AWAY":
            self.focus_score -= 0.1 * rate

        self.focus_score = max(0.0, self.focus_score)
        self.prev_status = current_status

        # ── SMART POMODORO (Inventive Step 3) ────────────────────────────────
        # Timer only advances on genuinely clean productive frames
        if frame_is_clean:
            self.clean_focus_secs = min(self.clean_focus_secs + frame_dt, POMODORO_FOCUS_SECS)

        if self.clean_focus_secs >= POMODORO_FOCUS_SECS:
            self.pomodoros_done  += 1
            self.clean_focus_secs = 0.0
            self.phase            = "break"
            print(f"[engine] Pomodoro #{self.pomodoros_done} complete! Take a 5-minute break.")

        # ── HUD ───────────────────────────────────────────────────────────────
        focus_score = self.focus_score
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 65), (w, h), (12, 12, 15), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Focus score bar
        bar_w   = int((focus_score / 100) * (w - 40))
        bar_col = (0, 200, 80) if focus_score > 70 else (30, 140, 255) if focus_score > 40 else (60, 60, 255)
        cv2.rectangle(frame, (20, h - 40), (20 + bar_w, h - 22), bar_col, -1)

        # Pomodoro progress bar (purple, thinner)
        pomo_w = int((self.clean_focus_secs / POMODORO_FOCUS_SECS) * (w - 40))
        cv2.rectangle(frame, (20, h - 16), (20 + pomo_w, h - 8), (120, 80, 220), -1)

        pomo_pct = int((self.clean_focus_secs / POMODORO_FOCUS_SECS) * 100)
        cv2.putText(frame,
                    f"FOCUS {int(focus_score)}%  |  {current_status}  |  POMO {pomo_pct}%  |  x{self.pomodoros_done}",
                    (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1)

        self.state.update({
            "focus_score":      round(focus_score, 1),
            "status":           current_status,
            "phase":            self.phase,
            "phone_alerts":     self.phone_alerts_count,
            "posture_alerts":   self.posture_alerts_count,
            "session_mins":     round(elapsed / 60, 1),
            "pomodoro_secs":    int(self.clean_focus_secs),
            "pomodoro_target":  POMODORO_FOCUS_SECS,
            "pomodoros_done":   self.pomodoros_done,
            "calibration_pct":  100,
        })
        return frame

    def finish(self):
        """End the session. Returns (focus_score, phone_count, posture_count, duration_mins, state)."""
        with self._lock:
            try: self.pose.close()
            except Exception: pass
            elapsed = time.time() - self.start_time
            self.state.update({"running": False, "status": "STOPPED", "phase": "stopped"})
            print("[engine] stopped")
            return (self.focus_score, self.phone_alerts_count, self.posture_alerts_count,
                    elapsed / 60, dict(self.state))
