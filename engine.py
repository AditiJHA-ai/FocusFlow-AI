import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import os
import time
import threading
import winsound

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ── NON-BLOCKING AUDIO ────────────────────────────────────────────────────────
def _beep(freq, duration_ms):
    threading.Thread(target=winsound.Beep, args=(freq, duration_ms), daemon=True).start()

def _chime_phone():
    """Soft descending two-tone — attention-grabbing but not jarring"""
    def _play():
        winsound.Beep(880, 120)
        time.sleep(0.05)
        winsound.Beep(660, 200)
    threading.Thread(target=_play, daemon=True).start()

def _chime_posture():
    """Single soft mid-tone nudge"""
    def _play():
        winsound.Beep(523, 100)   # C5 — gentle
        time.sleep(0.04)
        winsound.Beep(440, 150)   # A4
    threading.Thread(target=_play, daemon=True).start()

def _chime_pomodoro():
    """Ascending triumphant three notes"""
    def _play():
        for freq in [523, 659, 784]:
            winsound.Beep(freq, 180)
            time.sleep(0.04)
    threading.Thread(target=_play, daemon=True).start()

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
PHONE_CONFIDENCE    = 0.55     # raised from 0.4 — reduces clothing false positives
PHONE_MIN_AREA      = 0.008    # phone box must be at least 0.8% of frame area
ALERT_BUFFER_LIMIT  = 15
CALIBRATION_SECS    = 30
POMODORO_FOCUS_SECS = 25 * 60

# ── SHARED STATE ──────────────────────────────────────────────────────────────
_state = {
    "focus_score":      100.0,
    "status":           "STARTING",
    "phase":            "calibrating",
    "phone_alerts":     0,
    "posture_alerts":   0,
    "session_mins":     0.0,
    "pomodoro_secs":    0,
    "pomodoro_target":  POMODORO_FOCUS_SECS,
    "pomodoros_done":   0,
    "calibration_pct":  0,
    "posture_baseline": None,
    "running":          False,
}
_frame_bytes = None
_lock        = threading.Lock()

def get_state():
    with _lock:
        return dict(_state)

def get_frame():
    with _lock:
        return _frame_bytes

def stop_engine():
    with _lock:
        _state["running"] = False


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


# ── MAIN ENGINE LOOP ──────────────────────────────────────────────────────────
def run_engine():
    global _frame_bytes

    # Session counters
    start_time           = time.time()
    phone_counter        = 0
    attention_counter    = 0
    phone_alerts_count   = 0
    posture_alerts_count = 0
    focus_score          = 100.0

    # Inventive Step 1 — Adaptive Calibration
    calib_samples     = []
    calibrated        = False
    posture_threshold = 0.26        # default until calibration completes

    # Inventive Step 3 — Smart Pomodoro
    clean_focus_secs  = 0.0
    pomodoros_done    = 0
    last_frame_time   = time.time()

    with _lock:
        _state["running"] = True
        _state["phase"]   = "loading"
        _state["status"]  = "LOADING"

    print("[engine] loading models...")

    # ── MediaPipe compatibility: 0.9.x uses mp.solutions, 0.10+ uses Tasks API ──
    try:
        # Try legacy API first (mediapipe <= 0.9.x)
        _pose_mod    = mp.solutions.pose
        _face_mod    = mp.solutions.face_mesh
        _drawing_mod = mp.solutions.drawing_utils
        _POSE_CONN   = mp.solutions.pose.POSE_CONNECTIONS

        pose = _pose_mod.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        face_mesh = _face_mod.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        USE_LEGACY_MP = True
        print("[engine] using MediaPipe legacy API (solutions)")

    except AttributeError:
        # New Tasks API (mediapipe >= 0.10)
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision
        import urllib.request, os

        # Download pose landmarker model if not present
        POSE_MODEL = "pose_landmarker_lite.task"
        if not os.path.exists(POSE_MODEL):
            print("[engine] downloading pose_landmarker_lite.task ...")
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
                POSE_MODEL
            )

        # Download face landmarker model if not present
        FACE_MODEL = "face_landmarker.task"
        if not os.path.exists(FACE_MODEL):
            print("[engine] downloading face_landmarker.task ...")
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
                FACE_MODEL
            )

        _pose_opts = mp_vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=POSE_MODEL),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.7,
            min_pose_presence_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        _face_opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=FACE_MODEL),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
        )
        pose      = mp_vision.PoseLandmarker.create_from_options(_pose_opts)
        face_mesh = mp_vision.FaceLandmarker.create_from_options(_face_opts)
        USE_LEGACY_MP = False
        _POSE_CONN    = None   # drawing not supported same way in new API
        print("[engine] using MediaPipe Tasks API (>= 0.10)")

    yolo_model = YOLO('yolo11n.pt')
    cap        = cv2.VideoCapture(0)

    with _lock:
        _state["phase"]  = "calibrating"
        _state["status"] = "CALIBRATING"

    print("[engine] models loaded — calibrating posture for 30s, sit naturally...")

    while True:
        with _lock:
            if not _state["running"]:
                break

        now           = time.time()
        frame_dt      = now - last_frame_time
        last_frame_time = now
        elapsed       = now - start_time

        success, frame = cap.read()
        if not success:
            break

        rgb          = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── Run models (API-compatible) ──────────────────────────────────────
        if USE_LEGACY_MP:
            pose_results = pose.process(rgb)
            face_results = face_mesh.process(rgb)
            pose_lms     = pose_results.pose_landmarks          # NormalizedLandmarkList or None
            face_lm      = (face_results.multi_face_landmarks[0]
                            if face_results.multi_face_landmarks else None)
        else:
            import mediapipe as _mp2
            _mp_img       = _mp2.Image(image_format=_mp2.ImageFormat.SRGB, data=rgb)
            _pose_res     = pose.detect(_mp_img)
            _face_res     = face_mesh.detect(_mp_img)
            # Normalise to legacy-style access
            pose_lms      = _pose_res.pose_landmarks[0]  if _pose_res.pose_landmarks  else None
            face_lm       = _face_res.face_landmarks[0]  if _face_res.face_landmarks  else None

        yolo_res = yolo_model.predict(frame, classes=[67], conf=PHONE_CONFIDENCE, verbose=False)

        # ── CALIBRATION PHASE (Inventive Step 1) ─────────────────────────────
        if not calibrated:
            calib_progress = min(elapsed / CALIBRATION_SECS, 1.0)

            if pose_lms:
                lm    = pose_lms.landmark if hasattr(pose_lms, 'landmark') else pose_lms
                eye_y = (lm[1].y + lm[2].y) / 2
                sh_y  = (lm[11].y + lm[12].y) / 2
                gap   = sh_y - eye_y
                if gap > 0.05:
                    calib_samples.append(gap)

            if elapsed >= CALIBRATION_SECS:
                if len(calib_samples) >= 10:
                    baseline          = float(np.mean(calib_samples))
                    posture_threshold = baseline * 0.82   # 18% below natural = clear slouch
                    print(f"[engine] calibrated: baseline={baseline:.3f}, threshold={posture_threshold:.3f}")
                else:
                    print("[engine] not enough samples — using default threshold 0.26")
                calibrated = True
                with _lock:
                    _state["phase"]            = "focusing"
                    _state["posture_baseline"] = round(posture_threshold, 3)

            _draw_calibration(frame, calib_progress, len(calib_samples))
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            with _lock:
                _frame_bytes = buf.tobytes()
                _state.update({
                    "status":          "CALIBRATING",
                    "calibration_pct": int(calib_progress * 100),
                    "session_mins":    round(elapsed / 60, 1),
                })
            continue

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
                phone_counter = ALERT_BUFFER_LIMIT

            # ── Posture (personalised threshold) ─────────────────────────────
            eye_y      = (lm[1].y + lm[2].y) / 2
            shoulder_y = (lm[11].y + lm[12].y) / 2
            gap        = shoulder_y - eye_y
            if gap < posture_threshold:
                attention_counter = ALERT_BUFFER_LIMIT

            # Draw skeleton (legacy API only — new API needs different drawing)
            if USE_LEGACY_MP and pose_lms:
                try:
                    _drawing_mod.draw_landmarks(frame, pose_lms, _POSE_CONN)
                except Exception:
                    pass

        # ── ALERTS & SCORE ────────────────────────────────────────────────────
        frame_is_clean = False

        if current_status == "PRODUCTIVE":
            if phone_counter > 0:
                current_status = "PHONE DETECTED"
                if phone_counter == ALERT_BUFFER_LIMIT:
                    phone_alerts_count += 1
                    _chime_phone()
                phone_counter -= 1
                focus_score   -= 0.4
            elif attention_counter > 0:
                current_status = "POSTURE ALERT"
                if attention_counter == ALERT_BUFFER_LIMIT:
                    posture_alerts_count += 1
                    _chime_posture()
                attention_counter -= 1
                focus_score       -= 0.2
            else:
                focus_score    = min(100.0, focus_score + 0.05)
                frame_is_clean = True
        elif current_status == "USER AWAY":
            focus_score -= 0.1

        focus_score = max(0.0, focus_score)

        # ── SMART POMODORO (Inventive Step 3) ────────────────────────────────
        # Timer only advances on genuinely clean productive frames
        if frame_is_clean:
            clean_focus_secs = min(clean_focus_secs + frame_dt, POMODORO_FOCUS_SECS)

        if clean_focus_secs >= POMODORO_FOCUS_SECS:
            pomodoros_done  += 1
            clean_focus_secs = 0.0
            _chime_pomodoro()
            print(f"[engine] Pomodoro #{pomodoros_done} complete! Take a 5-minute break.")
            with _lock:
                _state["phase"]          = "break"
                _state["pomodoros_done"] = pomodoros_done

        # ── HUD ───────────────────────────────────────────────────────────────
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 65), (w, h), (12, 12, 15), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Focus score bar
        bar_w   = int((focus_score / 100) * (w - 40))
        bar_col = (0, 200, 80) if focus_score > 70 else (30, 140, 255) if focus_score > 40 else (60, 60, 255)
        cv2.rectangle(frame, (20, h - 40), (20 + bar_w, h - 22), bar_col, -1)

        # Pomodoro progress bar (purple, thinner)
        pomo_w = int((clean_focus_secs / POMODORO_FOCUS_SECS) * (w - 40))
        cv2.rectangle(frame, (20, h - 16), (20 + pomo_w, h - 8), (120, 80, 220), -1)

        pomo_pct = int((clean_focus_secs / POMODORO_FOCUS_SECS) * 100)
        cv2.putText(frame,
                    f"FOCUS {int(focus_score)}%  |  {current_status}  |  POMO {pomo_pct}%  |  x{pomodoros_done}",
                    (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1)

        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        with _lock:
            _frame_bytes = buf.tobytes()
            _state.update({
                "focus_score":      round(focus_score, 1),
                "status":           current_status,
                "phase":            _state["phase"],
                "phone_alerts":     phone_alerts_count,
                "posture_alerts":   posture_alerts_count,
                "session_mins":     round(elapsed / 60, 1),
                "pomodoro_secs":    int(clean_focus_secs),
                "pomodoro_target":  POMODORO_FOCUS_SECS,
                "pomodoros_done":   pomodoros_done,
                "calibration_pct":  100,
            })

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    try: pose.close()
    except Exception: pass
    try: face_mesh.close()
    except Exception: pass
    with _lock:
        _state["running"] = False
        _state["status"]  = "STOPPED"
        _state["phase"]   = "stopped"
    print("[engine] stopped")
    return focus_score, phone_alerts_count, posture_alerts_count, elapsed / 60