# ============================================================================
# ADEEB AI VISION - Multi-Device Edition
# Real-time person and vehicle detection, tracking, counting, and line crossing
# Run: python adeeb_ai_vision_modified.py
# ============================================================================

import base64
import math
import os
import threading
import time
from collections import defaultdict, deque

import cv2
import numpy as np
from flask import Flask, request
from flask_socketio import SocketIO, emit
from ultralytics import YOLO


HOST = "0.0.0.0"
PORT = 5000
MAX_DEVICES = 4
MODEL_NAME = os.environ.get("ADEEB_MODEL", "yolov8n.pt")
TARGET_CLASSES = {0, 2, 3, 5, 7}  # COCO: person, car, motorcycle, bus, truck

# RGB-like colors expressed in OpenCV's BGR order.
CLASS_INFO = {
    0: {"key": "person", "ar": "شخص", "en": "Person", "color": (255, 0, 255)},
    2: {"key": "car", "ar": "سيارة", "en": "Car", "color": (0, 230, 0)},
    3: {"key": "motorcycle", "ar": "دراجة نارية", "en": "Motorcycle", "color": (255, 130, 0)},
    5: {"key": "bus", "ar": "باص", "en": "Bus", "color": (0, 220, 255)},
    7: {"key": "truck", "ar": "شاحنة", "en": "Truck", "color": (0, 0, 255)},
}
CLASS_KEYS = ["person", "car", "motorcycle", "bus", "truck"]

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("ADEEB_SECRET_KEY", "change-this-local-demo-key")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading", max_http_buffer_size=4_000_000)

STATE_LOCK = threading.RLock()
MODEL_LOCK = threading.Lock()
CLIENTS = {}  # Socket.IO SID -> per-device state


def empty_class_counts():
    return {key: 0 for key in CLASS_KEYS}


def safe_emit_to_laptops(event_name, payload):
    with STATE_LOCK:
        laptop_sids = [sid for sid, item in CLIENTS.items() if item["device_type"] == "laptop"]
    for sid in laptop_sids:
        socketio.emit(event_name, payload, to=sid)


def roster_snapshot():
    with STATE_LOCK:
        roster = []
        for sid, item in CLIENTS.items():
            roster.append({
                "sid": sid,
                "name": item["name"],
                "device_type": item["device_type"],
                "camera_active": item["camera_active"],
                "camera_facing": item["camera_facing"],
                "stats": item["stats"],
            })
    return roster


def broadcast_device_list():
    socketio.emit("devices_update", {"devices": roster_snapshot(), "limit": MAX_DEVICES})


class CentroidTracker:
    """Small per-device tracker, so IDs from different cameras do not overlap."""

    def __init__(self, max_missed=15):
        self.next_id = 1
        self.max_missed = max_missed
        self.tracks = {}
        self.totals = empty_class_counts()

    def reset_totals(self):
        self.totals = empty_class_counts()

    def reset(self):
        self.next_id = 1
        self.tracks.clear()
        self.reset_totals()

    def update(self, detections):
        # detections are dictionaries with box, center, class_id and confidence.
        for track in self.tracks.values():
            track["missed"] += 1

        candidate_pairs = []
        for det_index, det in enumerate(detections):
            dx, dy = det["center"]
            for track_id, track in self.tracks.items():
                if track["class_id"] != det["class_id"] or track["missed"] > self.max_missed:
                    continue
                tx, ty = track["center"]
                distance = math.hypot(dx - tx, dy - ty)
                x1, y1, x2, y2 = track["box"]
                diagonal = math.hypot(x2 - x1, y2 - y1)
                threshold = max(45.0, diagonal * 0.8)
                if distance <= threshold:
                    candidate_pairs.append((distance, det_index, track_id))

        matched_detections = set()
        matched_tracks = set()
        for _, det_index, track_id in sorted(candidate_pairs, key=lambda pair: pair[0]):
            if det_index in matched_detections or track_id in matched_tracks:
                continue
            det = detections[det_index]
            old_center = self.tracks[track_id]["center"]
            self.tracks[track_id].update({
                "center": det["center"],
                "box": det["box"],
                "class_id": det["class_id"],
                "missed": 0,
            })
            det["track_id"] = track_id
            det["previous_center"] = old_center
            matched_detections.add(det_index)
            matched_tracks.add(track_id)

        for det_index, det in enumerate(detections):
            if det_index in matched_detections:
                continue
            track_id = self.next_id
            self.next_id += 1
            self.tracks[track_id] = {
                "center": det["center"],
                "box": det["box"],
                "class_id": det["class_id"],
                "missed": 0,
            }
            det["track_id"] = track_id
            det["previous_center"] = None
            self.totals[CLASS_INFO[det["class_id"]]["key"]] += 1

        old_ids = [track_id for track_id, track in self.tracks.items() if track["missed"] > self.max_missed]
        for track_id in old_ids:
            del self.tracks[track_id]
        return detections


class LineCounter:
    def __init__(self):
        self.line = None
        self.previous_centers = {}
        self.counted_tracks = set()
        self.counts = {"entry": 0, "exit": 0}

    def set_line(self, p1, p2, name="خط العد"):
        self.line = {"p1": tuple(map(float, p1)), "p2": tuple(map(float, p2)), "name": name}
        self.previous_centers.clear()
        self.counted_tracks.clear()
        self.counts = {"entry": 0, "exit": 0}

    def reset(self):
        self.previous_centers.clear()
        self.counted_tracks.clear()
        self.counts = {"entry": 0, "exit": 0}

    def _side(self, point):
        p1 = np.array(self.line["p1"], dtype=float)
        p2 = np.array(self.line["p2"], dtype=float)
        pt = np.array(point, dtype=float)
        line_vec = p2 - p1
        normal_vec = np.array([-line_vec[1], line_vec[0]])
        return float(np.dot(pt - p1, normal_vec))

    def observe(self, track_id, center, class_id):
        if self.line is None:
            return None
        previous = self.previous_centers.get(track_id)
        self.previous_centers[track_id] = center
        if previous is None or track_id in self.counted_tracks:
            return None
        previous_side = self._side(previous)
        current_side = self._side(center)
        if previous_side < 0 <= current_side:
            direction = "entry"
        elif previous_side >= 0 > current_side:
            direction = "exit"
        else:
            return None
        self.counted_tracks.add(track_id)
        self.counts[direction] += 1
        return {
            "track_id": int(track_id),
            "class_name": CLASS_INFO[class_id]["en"],
            "class_ar": CLASS_INFO[class_id]["ar"],
            "direction": direction,
            "line_name": self.line["name"],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }


class ADEEBDetector:
    def __init__(self, model_name=MODEL_NAME):
        print(f"[ADEEB AI] Loading model: {model_name}")
        self.model = YOLO(model_name)

    def process_frame(self, frame, tracker, line_counter):
        started = time.time()
        result = self.model.predict(frame, conf=0.35, imgsz=640, verbose=False)[0]
        detections = []

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()
            for box, class_id, confidence in zip(boxes, class_ids, confidences):
                if class_id not in TARGET_CLASSES:
                    continue
                x1, y1, x2, y2 = [int(v) for v in box]
                detections.append({
                    "box": (x1, y1, x2, y2),
                    "center": (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                    "class_id": int(class_id),
                    "confidence": float(confidence),
                })

        detections = tracker.update(detections)
        annotated = frame.copy()
        current_counts = empty_class_counts()
        events = []

        for det in detections:
            class_id = det["class_id"]
            info = CLASS_INFO[class_id]
            current_counts[info["key"]] += 1
            x1, y1, x2, y2 = det["box"]
            color = info["color"]
            track_id = det["track_id"]
            label = f'{info["en"]} #{track_id} {det["confidence"]:.2f}'
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
            label_top = max(0, y1 - text_h - baseline - 8)
            cv2.rectangle(annotated, (x1, label_top), (x1 + text_w + 8, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (10, 10, 10), 2, cv2.LINE_AA)
            event = line_counter.observe(track_id, det["center"], class_id)
            if event:
                events.append(event)

        if line_counter.line:
            p1 = tuple(int(v) for v in line_counter.line["p1"])
            p2 = tuple(int(v) for v in line_counter.line["p2"])
            cv2.line(annotated, p1, p2, (0, 220, 255), 3)
            cv2.circle(annotated, p1, 5, (0, 220, 255), -1)
            cv2.circle(annotated, p2, 5, (0, 220, 255), -1)
            cv2.putText(annotated, line_counter.line["name"], (p1[0], max(20, p1[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)

        elapsed = max(time.time() - started, 0.0001)
        stats = {
            "fps": round(1.0 / elapsed, 1),
            "objects": len(detections),
            "counts": current_counts,
            "totals": dict(tracker.totals),
            "line_counts": dict(line_counter.counts),
            "events": events,
        }
        return annotated, stats, events


# Load the model once. Inference is serialized by MODEL_LOCK for stability.
detector = ADEEBDetector()


def encode_frame(frame):
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 68])
    if not ok:
        return None
    return base64.b64encode(buffer).decode("ascii")


def processing_loop():
    """Process the latest frame from each device without building a long queue."""
    while True:
        work = None
        with STATE_LOCK:
            for sid, item in list(CLIENTS.items()):
                if item["camera_active"] and item["raw_frame"] is not None and item["frame_seq"] > item["processed_seq"]:
                    frame = item["raw_frame"].copy()
                    item["processed_seq"] = item["frame_seq"]
                    work = (sid, item, frame)
                    break
        if work is None:
            time.sleep(0.01)
            continue

        sid, item, frame = work
        try:
            with MODEL_LOCK:
                annotated, stats, events = detector.process_frame(frame, item["tracker"], item["line_counter"])
            jpg_b64 = encode_frame(annotated)
            if not jpg_b64:
                continue
            payload = {
                "sid": sid,
                "name": item["name"],
                "device_type": item["device_type"],
                "camera_facing": item["camera_facing"],
                "frame": jpg_b64,
                "stats": stats,
            }
            with STATE_LOCK:
                still_connected = CLIENTS.get(sid) is item
                if still_connected:
                    item["stats"] = stats
                    item["last_frame_time"] = time.time()
            if not still_connected:
                continue

            # Laptop views act as dashboards. Phone users receive their own annotated feed.
            with STATE_LOCK:
                dashboard_sids = [client_sid for client_sid, client in CLIENTS.items() if client["device_type"] == "laptop"]
            for dashboard_sid in dashboard_sids:
                socketio.emit("device_frame", payload, to=dashboard_sid)
            if item["device_type"] == "phone":
                socketio.emit("device_frame", payload, to=sid)
            if events:
                event_payload = {"sid": sid, "name": item["name"], "events": events}
                for dashboard_sid in dashboard_sids:
                    socketio.emit("new_events", event_payload, to=dashboard_sid)
                if item["device_type"] == "phone":
                    socketio.emit("new_events", event_payload, to=sid)
        except Exception as exc:
            print(f"[ADEEB AI] Frame processing error for {sid}: {exc}")
            time.sleep(0.03)


@app.route("/")
def index():
    return INDEX_HTML


@socketio.on("register_device")
def handle_register(data):
    sid = request.sid
    name = str((data or {}).get("name", "")).strip()[:40]
    device_type = str((data or {}).get("device_type", "")).strip().lower()
    if not name:
        emit("registration_error", {"message": "اكتب اسمك أولاً."})
        return
    if device_type not in {"phone", "laptop"}:
        emit("registration_error", {"message": "اختر نوع الجهاز: جوال أو لابتوب."})
        return

    with STATE_LOCK:
        if sid not in CLIENTS and len(CLIENTS) >= MAX_DEVICES:
            emit("registration_error", {"message": f"وصل السيرفر إلى الحد الأقصى ({MAX_DEVICES} أجهزة). افصل جهازاً أولاً."})
            return
        CLIENTS[sid] = {
            "name": name,
            "device_type": device_type,
            "camera_active": False,
            "camera_facing": "laptop" if device_type == "laptop" else "environment",
            "raw_frame": None,
            "frame_seq": 0,
            "processed_seq": 0,
            "last_frame_time": 0.0,
            "tracker": CentroidTracker(),
            "line_counter": LineCounter(),
            "stats": {
                "fps": 0,
                "objects": 0,
                "counts": empty_class_counts(),
                "totals": empty_class_counts(),
                "line_counts": {"entry": 0, "exit": 0},
                "events": [],
            },
        }
    emit("registered", {"sid": sid, "name": name, "device_type": device_type, "limit": MAX_DEVICES})
    broadcast_device_list()


@socketio.on("camera_state")
def handle_camera_state(data):
    sid = request.sid
    data = data or {}
    active = bool(data.get("active", False))
    facing = str(data.get("facing", "environment"))[:20]
    with STATE_LOCK:
        item = CLIENTS.get(sid)
        if not item:
            return
        item["camera_active"] = active
        item["camera_facing"] = facing
        if not active:
            item["raw_frame"] = None
            item["stats"]["objects"] = 0
            item["stats"]["counts"] = empty_class_counts()
    broadcast_device_list()


@socketio.on("camera_frame")
def handle_camera_frame(data):
    sid = request.sid
    if not isinstance(data, dict):
        return
    encoded = data.get("frame", "")
    if not isinstance(encoded, str) or not encoded:
        return
    try:
        if "," in encoded:
            encoded = encoded.split(",", 1)[1]
        raw = base64.b64decode(encoded, validate=False)
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return
        height, width = frame.shape[:2]
        if width > 640:
            new_height = max(1, int(height * (640 / width)))
            frame = cv2.resize(frame, (640, new_height), interpolation=cv2.INTER_AREA)
        with STATE_LOCK:
            item = CLIENTS.get(sid)
            if not item or not item["camera_active"]:
                return
            item["raw_frame"] = frame
            item["frame_seq"] += 1
    except Exception as exc:
        print(f"[ADEEB AI] Invalid frame from {sid}: {exc}")


@socketio.on("set_counting_line")
def handle_set_counting_line(data):
    data = data or {}
    sid = request.sid
    target_sid = str(data.get("device_sid", sid))
    p1 = data.get("p1") or {}
    p2 = data.get("p2") or {}
    try:
        point1 = (max(0, min(639, float(p1["x"]))), max(0, min(479, float(p1["y"]))))
        point2 = (max(0, min(639, float(p2["x"]))), max(0, min(479, float(p2["y"]))))
    except (KeyError, TypeError, ValueError):
        emit("server_notice", {"message": "تعذر ضبط خط العد. جرّب مرة أخرى."})
        return
    with STATE_LOCK:
        owner = CLIENTS.get(sid)
        target = CLIENTS.get(target_sid)
        if not owner or not target:
            return
        # Only laptop dashboards can set a line on another device.
        if sid != target_sid and owner["device_type"] != "laptop":
            return
    with MODEL_LOCK:
        target["line_counter"].set_line(point1, point2, "Counting Line")
    emit("server_notice", {"message": "تم حفظ خط العد."})


@socketio.on("reset_counts")
def handle_reset_counts(data=None):
    sid = request.sid
    with STATE_LOCK:
        owner = CLIENTS.get(sid)
        if not owner:
            return
        targets = list(CLIENTS.items()) if owner["device_type"] == "laptop" else [(sid, owner)]
    with MODEL_LOCK:
        for _, item in targets:
            item["tracker"].reset()
            item["line_counter"].reset()
            item["stats"]["objects"] = 0
            item["stats"]["counts"] = empty_class_counts()
            item["stats"]["totals"] = empty_class_counts()
            item["stats"]["line_counts"] = {"entry": 0, "exit": 0}
    broadcast_device_list()


@socketio.on("disconnect")
def handle_disconnect(*_args):
    sid = request.sid
    with STATE_LOCK:
        CLIENTS.pop(sid, None)
    broadcast_device_list()


INDEX_HTML = r'''<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
  <title>ADEEB AI VISION | فريق الرواد</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.5/socket.io.min.js"></script>
  <style>
    :root{--bg:#070b09;--panel:#101814;--panel2:#151f19;--green:#00f078;--green2:#00b85c;--text:#edf7ef;--muted:#93a69a;--border:#26382d;--purple:#df55ff;--red:#ff4545;--yellow:#ffd642;--blue:#36a6ff;--radius:18px}
    *{box-sizing:border-box} html{min-height:100%;background:var(--bg)} body{margin:0;min-height:100vh;background:radial-gradient(ellipse at top,#10251a 0%,var(--bg) 54%);color:var(--text);font-family:Tahoma,Arial,sans-serif}
    button,input,select{font:inherit} button{cursor:pointer;transition:transform .15s ease,background .15s ease} button:active{transform:scale(.98)}
    .hidden{display:none!important}.muted{color:var(--muted)}.green{color:var(--green)}
    .topbar{display:flex;align-items:center;justify-content:space-between;gap:14px;padding:16px 22px;border-bottom:1px solid var(--border);background:rgba(6,12,8,.88);position:sticky;top:0;z-index:5;backdrop-filter:blur(12px)}
    .brand{display:flex;align-items:center;gap:12px}.brand-mark{width:42px;height:42px;border-radius:13px;background:linear-gradient(135deg,#00f078,#08743d);display:grid;place-items:center;color:#031108;font-weight:900;font-size:20px;box-shadow:0 0 24px #00f07822}.brand h1{font-size:17px;margin:0;letter-spacing:.7px}.brand p{font-size:11px;color:var(--muted);margin:5px 0 0}.status-pill{border:1px solid var(--border);border-radius:999px;padding:8px 12px;font-size:12px;color:var(--muted);white-space:nowrap}.status-pill.on{color:var(--green);border-color:#00f07855;background:#00f0780d}
    .setup-wrap{min-height:calc(100vh - 76px);display:grid;place-items:center;padding:22px}.setup-card{width:min(530px,100%);border:1px solid #254331;background:linear-gradient(155deg,#122219,#0b110d 72%);border-radius:26px;padding:clamp(22px,5vw,38px);box-shadow:0 24px 90px #0008}.eyebrow{font-size:11px;color:var(--green);font-weight:bold;letter-spacing:2px}.setup-card h2{font-size:clamp(25px,4vw,34px);line-height:1.3;margin:13px 0 8px}.setup-card .desc{color:var(--muted);font-size:14px;line-height:1.9;margin:0 0 25px}.field-label{display:block;color:#dcebe0;font-size:13px;margin:15px 0 9px}.text-input{width:100%;padding:14px 15px;border-radius:12px;border:1px solid var(--border);outline:none;color:var(--text);background:#080f0a}.text-input:focus{border-color:var(--green);box-shadow:0 0 0 3px #00f07816}.choice-row{display:grid;grid-template-columns:1fr 1fr;gap:12px}.device-choice{position:relative;display:block}.device-choice input{position:absolute;opacity:0}.choice-content{display:flex;align-items:center;gap:12px;padding:15px;border:1px solid var(--border);border-radius:14px;background:#0a110c;min-height:84px}.device-choice input:checked+.choice-content{border-color:var(--green);background:#00f07810;box-shadow:inset 0 0 0 1px #00f07835}.choice-icon{font-size:25px}.choice-content strong{display:block;font-size:14px}.choice-content small{display:block;color:var(--muted);font-size:11px;margin-top:5px}.primary-btn,.secondary-btn,.danger-btn{border:0;border-radius:12px;padding:12px 16px;font-weight:bold}.primary-btn{background:var(--green);color:#041009}.primary-btn:hover{background:#49ff9f}.secondary-btn{background:#1a2a20;border:1px solid #304637;color:var(--text)}.danger-btn{background:#4b1919;border:1px solid #713030;color:#ffb4b4}.wide{width:100%;margin-top:23px}.note{font-size:11px;line-height:1.8;color:var(--muted);margin-top:14px}.error-box{margin-top:12px;color:#ffc1c1;font-size:12px;line-height:1.6}
    .app-content{max-width:1600px;margin:auto;padding:22px}.welcome-row{display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap;margin-bottom:18px}.welcome-row h2{font-size:22px;margin:0 0 7px}.welcome-row p{font-size:12px;color:var(--muted);margin:0}.toolbar{display:flex;gap:8px;flex-wrap:wrap}.toolbar button{padding:10px 12px;font-size:12px}
    .stats-grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;margin:18px 0}.stat-card{position:relative;overflow:hidden;border:1px solid var(--border);border-radius:16px;background:linear-gradient(145deg,#131e17,#0b110d);padding:15px;min-width:0}.stat-card:after{content:"";position:absolute;width:65px;height:65px;right:-25px;top:-30px;border-radius:50%;background:var(--accent);opacity:.09;filter:blur(4px)}.stat-label{font-size:12px;color:var(--muted);display:flex;align-items:center;gap:7px}.color-dot{width:8px;height:8px;border-radius:50%;background:var(--accent);box-shadow:0 0 12px var(--accent)}.stat-number{font-size:29px;font-weight:800;color:var(--accent);margin:9px 0 3px;direction:ltr;text-align:right}.stat-sub{font-size:10px;color:var(--muted)}
    .section-heading{display:flex;align-items:center;justify-content:space-between;gap:12px;margin:23px 0 13px}.section-heading h3{font-size:16px;margin:0}.section-heading span{font-size:11px;color:var(--muted)}.device-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:15px}.device-card,.empty-slot{background:linear-gradient(145deg,#101a13,#0a100c);border:1px solid var(--border);border-radius:18px;padding:13px;min-width:0;overflow:hidden}.device-card{border-top:2px solid #00f07877}.device-card-head{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:11px}.device-title{min-width:0}.device-title strong{font-size:13px;display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.device-title small{display:block;color:var(--muted);font-size:10px;margin-top:5px}.mini-status{font-size:10px;padding:6px 8px;border:1px solid var(--border);border-radius:999px;color:var(--muted);white-space:nowrap}.mini-status.on{color:var(--green);border-color:#00f07855}.feed-wrap{position:relative;width:100%;aspect-ratio:4/3;border-radius:12px;overflow:hidden;background:#020403;border:1px solid #1b3423}.feed-wrap img{width:100%;height:100%;display:block;object-fit:fill}.feed-placeholder{position:absolute;inset:0;display:grid;place-items:center;text-align:center;padding:20px;color:#718478;font-size:12px;line-height:1.8}.overlay{position:absolute;inset:0;width:100%;height:100%;pointer-events:none;touch-action:none}.overlay.drawing{pointer-events:auto;cursor:crosshair}.device-card-bottom{display:flex;align-items:center;justify-content:space-between;gap:8px;flex-wrap:wrap;margin-top:10px}.tiny-counts{display:flex;flex-wrap:wrap;gap:5px}.tiny-chip{border:1px solid var(--border);border-radius:7px;padding:5px 7px;font-size:10px;color:#cedbd1}.tiny-chip b{color:var(--text)}.line-btn{font-size:10px!important;padding:7px 9px!important}.empty-slot{display:flex;align-items:center;justify-content:center;flex-direction:column;text-align:center;min-height:210px;border-style:dashed;color:#687d6e}.empty-slot .plus{font-size:28px;color:#355741;margin-bottom:8px}.empty-slot strong{font-size:12px;font-weight:normal}.empty-slot small{font-size:10px;margin-top:7px}
    .analytics-row{display:grid;grid-template-columns:1.2fr .8fr;gap:15px;margin-top:16px}.panel{border:1px solid var(--border);background:#0d1510;border-radius:17px;padding:16px;min-width:0}.panel h3{font-size:14px;margin:0 0 13px}.event-list{max-height:230px;overflow:auto}.event-item{display:grid;grid-template-columns:130px 1fr;gap:12px;border-bottom:1px solid #1d2b21;padding:10px 2px;font-size:11px}.event-item time{color:var(--muted);direction:ltr;text-align:right}.line-summary{display:flex;gap:12px}.line-box{flex:1;padding:13px;border-radius:12px;background:#111d15;border:1px solid var(--border)}.line-box span{display:block;font-size:11px;color:var(--muted)}.line-box strong{display:block;font-size:23px;color:var(--green);margin-top:8px}
    .phone-layout{max-width:720px;margin:auto;padding:16px}.phone-top{display:flex;align-items:center;justify-content:space-between;gap:12px;margin:7px 0 15px}.phone-top h2{font-size:20px;margin:0 0 6px}.phone-top p{font-size:11px;color:var(--muted);margin:0}.phone-video-card{border:1px solid var(--border);border-radius:20px;padding:12px;background:#0e1711}.phone-feed-wrap{position:relative;aspect-ratio:4/3;background:#020403;border-radius:14px;overflow:hidden}.phone-feed-wrap img{width:100%;height:100%;object-fit:fill;display:block}.phone-controls{display:grid;grid-template-columns:1fr 1fr;gap:9px;margin-top:12px}.phone-controls button{min-height:44px}.phone-controls .primary-btn{grid-column:1/-1}.phone-stats{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:9px;margin-top:12px}.phone-stats .stat-card{padding:12px}.phone-stats .stat-number{font-size:25px}.connection-note{margin-top:14px;padding:11px 12px;border-radius:12px;background:#111a13;border:1px solid var(--border);font-size:11px;line-height:1.8;color:var(--muted)}.phone-feed-wrap .feed-placeholder{font-size:12px}
    @media(max-width:900px){.app-content{padding:16px}.stats-grid{grid-template-columns:repeat(3,minmax(0,1fr))}.analytics-row{grid-template-columns:1fr}.device-grid{gap:10px}.topbar{padding:13px 15px}.brand h1{font-size:14px}}
    @media(max-width:620px){.topbar{padding:12px}.brand-mark{width:36px;height:36px}.brand p{max-width:190px;line-height:1.4}.status-pill{font-size:10px;padding:7px 9px}.setup-wrap{padding:14px}.choice-row{gap:8px}.choice-content{padding:12px 9px;gap:8px;min-height:78px}.choice-content strong{font-size:12px}.choice-content small{font-size:10px}.app-content{padding:12px}.welcome-row h2{font-size:19px}.stats-grid{grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.stat-card{padding:12px}.stat-number{font-size:25px}.device-grid{grid-template-columns:1fr}.empty-slot{min-height:95px}.device-card{padding:10px}.event-item{grid-template-columns:1fr;gap:5px}.event-item time{text-align:right}.phone-layout{padding:12px}.phone-top h2{font-size:18px}.toolbar button{padding:9px;font-size:11px}}
  </style>
</head>
<body>
  <header class="topbar">
    <div class="brand"><div class="brand-mark">A</div><div><h1>ADEEB AI VISION</h1><p>الرؤية الحاسوبية · كشف وتتبع وعدّ المركبات والأشخاص</p></div></div>
    <div class="status-pill" id="connectionStatus">جارٍ الاتصال بالسيرفر...</div>
  </header>

  <main id="setupScreen" class="setup-wrap">
    <form class="setup-card" id="setupForm">
      <div class="eyebrow">SMART VISION PLATFORM</div>
      <h2>أهلاً بك في <span class="green">Adeeb AI Vision</span></h2>
      <p class="desc">قبل الدخول، عرّفنا بنفسك وحدد الجهاز الذي تستخدمه حتى نجهز لك واجهة مناسبة.</p>
      <label class="field-label" for="userName">شو اسمك؟</label>
      <input class="text-input" id="userName" maxlength="40" autocomplete="nickname" placeholder="اكتب اسمك هنا" required>
      <span class="field-label">شو نوع جهازك؟</span>
      <div class="choice-row">
        <label class="device-choice"><input type="radio" name="deviceType" value="phone" checked><span class="choice-content"><span class="choice-icon">📱</span><span><strong>جوال</strong><small>واجهة صغيرة + أمامية وخلفية</small></span></span></label>
        <label class="device-choice"><input type="radio" name="deviceType" value="laptop"><span class="choice-content"><span class="choice-icon">💻</span><span><strong>لابتوب</strong><small>لوحة مراقبة حتى 4 أجهزة</small></span></span></label>
      </div>
      <button class="primary-btn wide" type="submit">دخول إلى النظام ←</button>
      <div class="note">السيرفر يسمح بحد أقصى 4 أجهزة متصلة في الوقت نفسه. كاميرا الجوال تحتاج إلى إذن المتصفح، وقد يتطلب تشغيلها عبر شبكة Wi-Fi استخدام HTTPS.</div>
      <div id="setupError" class="error-box"></div>
    </form>
  </main>

  <main id="desktopApp" class="app-content hidden">
    <div class="welcome-row"><div><h2>لوحة المراقبة <span class="green" id="desktopUser"></span></h2><p>عرض الأجهزة المتصلة وتحليل الفيديو لحظياً · الحد الأقصى 4 أجهزة</p></div><div class="toolbar"><button class="secondary-btn" id="startLaptopCamera">تشغيل كاميرا اللابتوب</button><button class="danger-btn" id="stopLaptopCamera" disabled>إيقاف الكاميرا</button><button class="secondary-btn" id="resetAllCounts">تصفير الإحصائيات</button></div></div>
    <section class="stats-grid" id="desktopStats"></section>
    <div class="section-heading"><h3>الأجهزة المتصلة</h3><span id="deviceCapacity">0 / 4 أجهزة</span></div>
    <section class="device-grid" id="deviceGrid"></section>
    <section class="analytics-row"><div class="panel"><h3>سجل الأحداث وخط العد</h3><div class="event-list" id="eventList"><div class="muted" style="font-size:11px">ستظهر هنا أحداث عبور خط العد.</div></div></div><div class="panel"><h3>عبور خط العد</h3><div class="line-summary"><div class="line-box"><span>دخول</span><strong id="lineEntryTotal">0</strong></div><div class="line-box"><span>خروج</span><strong id="lineExitTotal">0</strong></div></div><p class="note">من بطاقة الجهاز اضغط «رسم خط العد»، ثم حدد نقطتين على الفيديو. يعدّ النظام عبور المسار مرة لكل جسم متتبّع.</p></div></section>
  </main>

  <main id="phoneApp" class="phone-layout hidden">
    <div class="phone-top"><div><h2>أهلاً <span class="green" id="phoneUser"></span></h2><p>كاميرا ذكية وتحليل مباشر للمشهد</p></div><span class="status-pill" id="phoneStatus">الكاميرا متوقفة</span></div>
    <section class="phone-video-card">
      <div class="phone-feed-wrap"><img id="phoneFeed" alt="الفيديو بعد الكشف"><div class="feed-placeholder" id="phonePlaceholder">اضغط «بدء الكاميرا» للسماح بالوصول إلى كاميرا جهازك وعرض الكشف الذكي هنا.</div><canvas id="phoneOverlay" class="overlay" width="640" height="480"></canvas></div>
      <div class="phone-controls"><button class="primary-btn" id="startPhoneCamera">▶ بدء الكاميرا والكشف</button><button class="secondary-btn" id="switchPhoneCamera">تبديل أمامية / خلفية</button><button class="danger-btn" id="stopPhoneCamera">إيقاف الكاميرا</button><button class="secondary-btn" id="drawPhoneLine">رسم خط العد</button></div>
    </section>
    <section class="phone-stats" id="phoneStats"></section>
    <div class="connection-note" id="phoneConnectionNote">متصل بالسيرفر. بعد بدء الكاميرا ستظهر المربعات الملونة وأعداد الأجسام المكتشفة.</div>
  </main>

  <video id="captureVideo" autoplay playsinline muted class="hidden"></video>
  <canvas id="captureCanvas" width="640" height="480" class="hidden"></canvas>
  <script>
    const socket = io();
    const MAX_DEVICES = 4;
    const classMeta = {
      person: {ar:'الأشخاص', color:'#df55ff', short:'شخص'},
      car: {ar:'السيارات', color:'#00e650', short:'سيارة'},
      motorcycle: {ar:'الدراجات النارية', color:'#ff9d35', short:'دراجة'},
      bus: {ar:'الباصات', color:'#ffd642', short:'باص'},
      truck: {ar:'الشاحنات', color:'#ff4545', short:'شاحنة'}
    };
    let myProfile = null;
    let deviceRoster = [];
    let latestStats = {};
    let latestFrames = {};
    let cameraStream = null;
    let streamTimer = null;
    let facingMode = 'environment';
    let activeLineSid = null;
    let linePoints = [];
    let drawingCanvas = null;
    const captureVideo = document.getElementById('captureVideo');
    const captureCanvas = document.getElementById('captureCanvas');
    const captureCtx = captureCanvas.getContext('2d');

    function escapeHTML(value) {
      return String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
    }
    function countCard(key, current, total, mobile=false) {
      const meta = classMeta[key];
      return `<div class="stat-card" style="--accent:${meta.color}"><div class="stat-label"><span class="color-dot"></span>${meta.ar}</div><div class="stat-number">${Number(current||0)}</div><div class="stat-sub">إجمالي الأجسام المتتبعة: ${Number(total||0)}</div></div>`;
    }
    function allCurrentCounts() {
      const sum = Object.fromEntries(Object.keys(classMeta).map(k => [k,0]));
      const totals = Object.fromEntries(Object.keys(classMeta).map(k => [k,0]));
      for (const device of deviceRoster) {
        const stats = latestStats[device.sid] || device.stats || {};
        for (const key of Object.keys(sum)) {
          sum[key] += Number((stats.counts || {})[key] || 0);
          totals[key] += Number((stats.totals || {})[key] || 0);
        }
      }
      return {sum, totals};
    }
    function renderDesktopStats() {
      const target = document.getElementById('desktopStats');
      const values = allCurrentCounts();
      target.innerHTML = Object.keys(classMeta).map(k => countCard(k, values.sum[k], values.totals[k])).join('');
      let entry = 0, exit = 0;
      for (const d of deviceRoster) {
        const lc = (latestStats[d.sid] || d.stats || {}).line_counts || {};
        entry += Number(lc.entry || 0); exit += Number(lc.exit || 0);
      }
      document.getElementById('lineEntryTotal').textContent = entry;
      document.getElementById('lineExitTotal').textContent = exit;
    }
    function tinyChips(stats) {
      return Object.keys(classMeta).map(k => `<span class="tiny-chip"><b style="color:${classMeta[k].color}">${Number(((stats||{}).counts||{})[k]||0)}</b> ${classMeta[k].short}</span>`).join('');
    }
    function renderDeviceGrid() {
      const grid = document.getElementById('deviceGrid');
      document.getElementById('deviceCapacity').textContent = `${deviceRoster.length} / ${MAX_DEVICES} أجهزة`;
      const cards = deviceRoster.slice(0, MAX_DEVICES).map(device => {
        const stats = latestStats[device.sid] || device.stats || {};
        const frame = latestFrames[device.sid] || '';
        const deviceType = device.device_type === 'phone' ? 'جوال' : 'لابتوب';
        const activeText = device.camera_active ? 'الكاميرا تعمل' : 'بانتظار الكاميرا';
        return `<article class="device-card" data-device-card="${escapeHTML(device.sid)}">
          <div class="device-card-head"><div class="device-title"><strong>${escapeHTML(device.name)}</strong><small>${deviceType} · ${escapeHTML(device.camera_facing==='user'?'أمامية':device.camera_facing==='environment'?'خلفية':device.camera_facing==='laptop'?'كاميرا اللابتوب':device.camera_facing||'كاميرا')}</small></div><span class="mini-status ${device.camera_active?'on':''}">${activeText}</span></div>
          <div class="feed-wrap"><img data-feed="${escapeHTML(device.sid)}" alt="فيديو ${escapeHTML(device.name)}" src="${frame}"><div class="feed-placeholder" ${frame?'style="display:none"':''}>${device.camera_active?'جارٍ انتظار أول إطار...':'هذا المكان مخصص لبث الجهاز عند تشغيل الكاميرا'}</div><canvas class="overlay" data-canvas="${escapeHTML(device.sid)}" width="640" height="480"></canvas></div>
          <div class="device-card-bottom"><div class="tiny-counts">${tinyChips(stats)}</div><button class="secondary-btn line-btn" data-line-target="${escapeHTML(device.sid)}">رسم خط العد</button></div>
        </article>`;
      });
      while (cards.length < MAX_DEVICES) {
        const slot = cards.length + 1;
        cards.push(`<div class="empty-slot"><div class="plus">＋</div><strong>منفذ جهاز ${slot}</strong><small>افتح رابط السيرفر على جهاز آخر وسجّل الدخول</small></div>`);
      }
      grid.innerHTML = cards.join('');
      grid.querySelectorAll('[data-line-target]').forEach(button => button.addEventListener('click', () => beginLineDrawing(button.dataset.lineTarget)));
      grid.querySelectorAll('img[data-feed]').forEach(img => {
        img.addEventListener('load', () => { const placeholder = img.parentElement.querySelector('.feed-placeholder'); if (placeholder) placeholder.style.display = 'none'; });
      });
      if (activeLineSid) activateDrawingCanvas(activeLineSid);
      renderDesktopStats();
    }
    function updateDeviceStats(sid, stats) {
      latestStats[sid] = stats;
      const device = deviceRoster.find(d => d.sid === sid);
      if (device) device.stats = stats;
      const card = document.querySelector(`[data-device-card="${CSS.escape(sid)}"]`);
      if (card) {
        const chips = card.querySelector('.tiny-counts');
        if (chips) chips.innerHTML = tinyChips(stats);
      }
      renderDesktopStats();
      if (myProfile && myProfile.device_type === 'phone' && sid === socket.id) renderPhoneStats(stats);
    }
    function renderPhoneStats(stats={}) {
      const counts = stats.counts || {}, totals = stats.totals || {};
      document.getElementById('phoneStats').innerHTML = Object.keys(classMeta).map(k => countCard(k, counts[k], totals[k], true)).join('') +
        `<div class="stat-card" style="--accent:#00f078"><div class="stat-label">سرعة المعالجة</div><div class="stat-number">${Number(stats.fps||0)}</div><div class="stat-sub">إطار/ثانية تقريباً</div></div>`;
    }
    function setConnectionStatus(text, on=false) {
      const el = document.getElementById('connectionStatus');
      el.textContent = text;
      el.classList.toggle('on', on);
    }
    function showApp(profile) {
      document.getElementById('setupScreen').classList.add('hidden');
      if (profile.device_type === 'laptop') {
        document.getElementById('desktopApp').classList.remove('hidden');
        document.getElementById('desktopUser').textContent = `— ${profile.name}`;
      } else {
        document.getElementById('phoneApp').classList.remove('hidden');
        document.getElementById('phoneUser').textContent = profile.name;
        renderPhoneStats({});
      }
    }
    document.getElementById('setupForm').addEventListener('submit', event => {
      event.preventDefault();
      const name = document.getElementById('userName').value.trim();
      const device_type = document.querySelector('input[name="deviceType"]:checked')?.value;
      document.getElementById('setupError').textContent = '';
      if (!name) { document.getElementById('setupError').textContent = 'اكتب اسمك قبل المتابعة.'; return; }
      if (!socket.connected) { document.getElementById('setupError').textContent = 'السيرفر غير متصل بعد. تأكد من تشغيل ملف Python ثم حدّث الصفحة.'; return; }
      socket.emit('register_device', {name, device_type});
    });
    socket.on('connect', () => setConnectionStatus('متصل بالسيرفر', true));
    socket.on('disconnect', () => setConnectionStatus('انقطع الاتصال', false));
    socket.on('registered', profile => {
      myProfile = profile;
      facingMode = profile.device_type === 'phone' ? 'environment' : 'laptop';
      showApp(profile);
      document.getElementById('startLaptopCamera').disabled = profile.device_type !== 'laptop';
      renderDesktopStats();
    });
    socket.on('registration_error', data => { document.getElementById('setupError').textContent = data.message || 'تعذّر الدخول.'; });
    socket.on('server_notice', data => {
      const note = document.getElementById('phoneConnectionNote');
      if (myProfile?.device_type === 'phone' && note) note.textContent = data.message;
    });
    socket.on('devices_update', data => {
      deviceRoster = (data.devices || []).slice(0, MAX_DEVICES);
      for (const device of deviceRoster) latestStats[device.sid] = device.stats || latestStats[device.sid] || {};
      if (myProfile?.device_type === 'laptop') renderDeviceGrid();
      if (myProfile?.device_type === 'phone') {
        document.getElementById('phoneConnectionNote').textContent = `متصل بالسيرفر · ${deviceRoster.length} من ${MAX_DEVICES} أجهزة متصلة.`;
        const own = deviceRoster.find(d => d.sid === socket.id);
        if (own) renderPhoneStats(own.stats || {});
      }
    });
    socket.on('device_frame', data => {
      if (!data || !data.sid || !data.frame) return;
      const url = 'data:image/jpeg;base64,' + data.frame;
      latestFrames[data.sid] = url;
      latestStats[data.sid] = data.stats || {};
      if (myProfile?.device_type === 'phone' && data.sid === socket.id) {
        const img = document.getElementById('phoneFeed');
        img.src = url;
        document.getElementById('phonePlaceholder').style.display = 'none';
        document.getElementById('phoneStatus').textContent = 'الكاميرا تعمل · كشف مباشر';
        renderPhoneStats(data.stats || {});
      }
      if (myProfile?.device_type === 'laptop') {
        const img = document.querySelector(`img[data-feed="${CSS.escape(data.sid)}"]`);
        if (img) {
          img.src = url;
          const placeholder = img.parentElement.querySelector('.feed-placeholder');
          if (placeholder) placeholder.style.display = 'none';
        }
        updateDeviceStats(data.sid, data.stats || {});
      }
    });
    socket.on('new_events', payload => {
      if (myProfile?.device_type !== 'laptop') return;
      (payload.events || []).forEach(eventData => {
        const log = document.getElementById('eventList');
        if (log.querySelector('.muted')) log.innerHTML = '';
        const row = document.createElement('div');
        row.className = 'event-item';
        const time = document.createElement('time'); time.textContent = eventData.timestamp || '';
        const detail = document.createElement('div');
        const direction = eventData.direction === 'entry' ? 'دخول' : 'خروج';
        detail.textContent = `${payload.name || 'جهاز'} · ${eventData.class_ar || eventData.class_name} · ${direction} · ID ${eventData.track_id}`;
        row.append(time, detail); log.prepend(row);
        while (log.children.length > 40) log.lastElementChild.remove();
      });
    });

    async function startCamera() {
      if (!myProfile) return;
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert('المتصفح لا يسمح بفتح الكاميرا في هذا العنوان. جرّب HTTPS أو افتح localhost على الجهاز نفسه.');
        return;
      }
      await stopCamera(false);
      const constraints = myProfile.device_type === 'phone'
        ? {audio:false, video:{facingMode:{ideal:facingMode}, width:{ideal:640}, height:{ideal:480}}}
        : {audio:false, video:{width:{ideal:640}, height:{ideal:480}}};
      try {
        cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
        captureVideo.srcObject = cameraStream;
        await captureVideo.play();
        socket.emit('camera_state', {active:true, facing:myProfile.device_type === 'phone' ? facingMode : 'laptop'});
        if (myProfile.device_type === 'phone') {
          document.getElementById('phoneStatus').textContent = 'جارٍ تحليل الفيديو...';
          document.getElementById('phonePlaceholder').style.display = 'grid';
          document.getElementById('phonePlaceholder').textContent = 'تم تشغيل الكاميرا، جارٍ تحليل الإطارات...';
        } else {
          document.getElementById('startLaptopCamera').disabled = true;
          document.getElementById('stopLaptopCamera').disabled = false;
        }
        streamTimer = setInterval(() => {
          if (!cameraStream || captureVideo.readyState < 2) return;
          captureCtx.drawImage(captureVideo, 0, 0, captureCanvas.width, captureCanvas.height);
          socket.emit('camera_frame', {frame:captureCanvas.toDataURL('image/jpeg', 0.65)});
        }, 220);
      } catch (err) {
        console.error(err);
        alert('لم نستطع فتح الكاميرا. تأكد من السماح للمتصفح باستخدامها، وأغلق أي برنامج آخر يستخدم الكاميرا. على الجوال عبر شبكة Wi-Fi قد يلزم HTTPS.');
      }
    }
    async function stopCamera(sendState=true) {
      if (streamTimer) { clearInterval(streamTimer); streamTimer = null; }
      if (cameraStream) { cameraStream.getTracks().forEach(track => track.stop()); cameraStream = null; }
      captureVideo.srcObject = null;
      if (sendState && myProfile) socket.emit('camera_state', {active:false, facing:myProfile.device_type === 'phone' ? facingMode : 'laptop'});
      if (myProfile?.device_type === 'phone') {
        document.getElementById('phoneStatus').textContent = 'الكاميرا متوقفة';
        document.getElementById('phonePlaceholder').style.display = 'grid';
        document.getElementById('phonePlaceholder').textContent = 'اضغط «بدء الكاميرا» للسماح بالوصول إلى كاميرا جهازك وعرض الكشف الذكي هنا.';
      } else if (myProfile?.device_type === 'laptop') {
        document.getElementById('startLaptopCamera').disabled = false;
        document.getElementById('stopLaptopCamera').disabled = true;
      }
    }
    document.getElementById('startLaptopCamera').addEventListener('click', startCamera);
    document.getElementById('stopLaptopCamera').addEventListener('click', () => stopCamera(true));
    document.getElementById('startPhoneCamera').addEventListener('click', startCamera);
    document.getElementById('stopPhoneCamera').addEventListener('click', () => stopCamera(true));
    document.getElementById('switchPhoneCamera').addEventListener('click', async () => {
      if (myProfile?.device_type !== 'phone') return;
      facingMode = facingMode === 'environment' ? 'user' : 'environment';
      if (cameraStream) await startCamera();
      else document.getElementById('phoneConnectionNote').textContent = `الكاميرا المحددة: ${facingMode === 'user' ? 'الأمامية' : 'الخلفية'} — اضغط بدء الكاميرا للتشغيل.`;
    });
    document.getElementById('resetAllCounts').addEventListener('click', () => {
      if (confirm('هل تريد تصفير إحصائيات جميع الأجهزة المتصلة؟')) socket.emit('reset_counts');
    });

    function beginLineDrawing(sid) {
      activeLineSid = sid;
      linePoints = [];
      drawingCanvas = activateDrawingCanvas(sid);
      if (!drawingCanvas) { alert('شغّل واجهة الفيديو أولاً ثم جرّب رسم الخط.'); return; }
      document.getElementById('eventList').insertAdjacentHTML('afterbegin', '<div class="event-item"><time>تعليمات</time><div>انقر نقطتين على الفيديو لتحديد خط العد.</div></div>');
    }
    function activateDrawingCanvas(sid) {
      document.querySelectorAll('.overlay').forEach(canvas => canvas.classList.remove('drawing'));
      const canvas = document.querySelector(`canvas[data-canvas="${CSS.escape(sid)}"]`) || (sid === socket.id ? document.getElementById('phoneOverlay') : null);
      if (canvas) canvas.classList.add('drawing');
      drawingCanvas = canvas;
      return canvas;
    }
    document.getElementById('drawPhoneLine').addEventListener('click', () => {
      if (!myProfile || myProfile.device_type !== 'phone') return;
      beginLineDrawing(socket.id);
    });
    document.addEventListener('pointerdown', event => {
      const canvas = event.target.closest('canvas.overlay.drawing');
      if (!canvas || !drawingCanvas) return;
      const rect = canvas.getBoundingClientRect();
      const point = {x:Math.round((event.clientX-rect.left)*640/rect.width), y:Math.round((event.clientY-rect.top)*480/rect.height)};
      linePoints.push(point);
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0,0,canvas.width,canvas.height);
      ctx.fillStyle = '#00f078'; ctx.strokeStyle = '#00f078'; ctx.lineWidth = 4;
      linePoints.forEach(p => {ctx.beginPath();ctx.arc(p.x,p.y,7,0,Math.PI*2);ctx.fill();});
      if (linePoints.length === 2) {
        ctx.beginPath();ctx.moveTo(linePoints[0].x,linePoints[0].y);ctx.lineTo(linePoints[1].x,linePoints[1].y);ctx.stroke();
        socket.emit('set_counting_line', {device_sid:activeLineSid,p1:linePoints[0],p2:linePoints[1]});
        canvas.classList.remove('drawing');
        activeLineSid = null; drawingCanvas = null; linePoints = [];
      }
    });
    window.addEventListener('beforeunload', () => { if (cameraStream) cameraStream.getTracks().forEach(track => track.stop()); });
  </script>
</body>
</html>'''


if __name__ == "__main__":
    print("=" * 66)
    print(" ADEEB AI VISION - Multi-Device Edition")
    print(f" Dashboard: http://localhost:{PORT}")
    print(f" LAN:       http://<LAPTOP_PRIVATE_IP>:{PORT}")
    print(f" Max devices: {MAX_DEVICES}")
    print(" For camera access from phones over Wi-Fi, browsers may require HTTPS.")
    print("=" * 66)
    threading.Thread(target=processing_loop, daemon=True, name="adeeb-ai-processing").start()
    socketio.run(app, host=HOST, port=PORT, debug=False, allow_unsafe_werkzeug=True)
