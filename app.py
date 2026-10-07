import os
import threading
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import database
import engine as eng

MAX_SESSIONS    = int(os.environ.get('MAX_SESSIONS', '4'))   # concurrent webcams the CPU can serve
MAX_FRAME_BYTES = 2 * 1024 * 1024

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'focusflow-secret')
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading',
                    max_http_buffer_size=MAX_FRAME_BYTES + 1024)

database.init_db()
eng.preload_models_async()

# ── SESSIONS: one per connected browser (socket id) ───────────────────────────
_sessions      = {}   # sid -> {"session": eng.Session or None, "client_id": str}
_sessions_lock = threading.Lock()


def _end_session(sid, notify):
    with _sessions_lock:
        entry = _sessions.pop(sid, None)
    if not entry or entry["session"] is None:
        return None
    focus_score, phone_count, posture_count, duration, state = entry["session"].finish()
    database.save_session(entry["client_id"], duration, focus_score, phone_count, posture_count)
    if notify:
        emit('session_saved', {'msg': 'Session saved!'})
    return state


# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/history')
def api_history():
    client_id = request.args.get('client_id', '')[:64]
    return jsonify(database.get_sessions(client_id, 30))

@app.route('/healthz')
def healthz():
    return jsonify({'ok': True})


# ── SOCKET EVENTS ─────────────────────────────────────────────────────────────
@socketio.on('start')
def on_start(data):
    sid       = request.sid
    client_id = str((data or {}).get('client_id', ''))[:64]
    with _sessions_lock:
        if sid in _sessions:
            return {'ok': False, 'msg': 'Already running'}
        if len(_sessions) >= MAX_SESSIONS:
            return {'ok': False, 'msg': 'Server is busy — please try again in a few minutes'}
        _sessions[sid] = {"session": None, "client_id": client_id}

    emit('state', {'phase': 'loading', 'status': 'LOADING'})
    try:
        session = eng.Session()   # waits for models on a cold start
    except Exception as e:
        print(f"[app] could not start session: {e}")
        with _sessions_lock:
            _sessions.pop(sid, None)
        return {'ok': False, 'msg': 'Could not load AI models'}

    with _sessions_lock:
        if sid not in _sessions:      # stopped/disconnected while loading
            return {'ok': False, 'msg': 'Session cancelled'}
        _sessions[sid]["session"] = session
    return {'ok': True}

@socketio.on('frame')
def on_frame(data):
    with _sessions_lock:
        entry = _sessions.get(request.sid)
    if not entry or entry["session"] is None:
        return None
    if not isinstance(data, (bytes, bytearray)) or len(data) > MAX_FRAME_BYTES:
        return None
    img, state = entry["session"].process(bytes(data))
    return {'img': img, 'state': state}

@socketio.on('stop')
def on_stop(data=None):
    state = _end_session(request.sid, notify=True)
    if state:
        emit('state', state)
    return {'ok': True}

@socketio.on('disconnect')
def on_disconnect(reason=None):
    _end_session(request.sid, notify=False)


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n  FocusFlow is running → open http://localhost:{port}\n")
    socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True)
