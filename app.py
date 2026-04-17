import threading
import time
from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO
import database
import engine as eng

app = Flask(__name__)
app.config['SECRET_KEY'] = 'focusflow-secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

database.init_db()
_engine_thread = None

# ── BACKGROUND: push state to all clients every second ────────────────────────
def _push_state():
    while True:
        state = eng.get_state()
        socketio.emit('state', state)
        time.sleep(1)

threading.Thread(target=_push_state, daemon=True).start()

# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def gen():
        while True:
            frame = eng.get_frame()
            if frame:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.033)   # ~30 fps cap
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/history')
def api_history():
    return jsonify(database.get_sessions(30))

@app.route('/api/start', methods=['POST'])
def api_start():
    global _engine_thread
    state = eng.get_state()
    if state['running']:
        return jsonify({'ok': False, 'msg': 'Already running'})

    def _run():
        result = eng.run_engine()
        if result:
            focus_score, phone_count, posture_count, duration = result
            database.save_session(duration, focus_score, phone_count, posture_count)
            socketio.emit('session_saved', {'msg': 'Session saved!'})

    _engine_thread = threading.Thread(target=_run, daemon=True)
    _engine_thread.start()
    return jsonify({'ok': True})

@app.route('/api/stop', methods=['POST'])
def api_stop():
    eng.stop_engine()
    return jsonify({'ok': True})

# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n  FocusFlow is running → open http://localhost:5000\n")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)