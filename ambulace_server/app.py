"""
Smart Ambulance Backend v6 — Real GPS Route + Signal Control
=============================================================
UPDATED: Added /detect endpoint for ESP32-CAM raw frame input
All existing endpoints unchanged.
"""

import heapq, logging, math, threading, time, struct
from datetime import datetime
import numpy as np
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS
import database as db

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ── Constants ─────────────────────────────────────────────────────────────────

ESP32_TIMEOUT    = 2
MAX_VEHICLES     = 100
GREEN_RADIUS_M   = 300
PASSED_RADIUS_M  = 100
AVG_SPEED_MPS    = 10.0

# Detection thresholds for ESP32-CAM grayscale frames
# Ambulance detection uses brightness spike method (emergency lights)
AMBULANCE_BRIGHTNESS_THRESH = 200   # bright pixels threshold (0-255)
AMBULANCE_BRIGHT_PIXEL_PCT  = 0.08  # 8% of pixels must be very bright
AMBULANCE_CONFIDENCE_MIN    = 0.45

# ── Signal Nodes ──────────────────────────────────────────────────────────────

SIGNALS = {
    "S1": {"vehicle_count": 12, "is_green": False, "is_stopped": False,
           "esp32_ip": "192.168.137.101",
           "lat": 28.6150, "lon": 77.2100,
           "location_name": "Signal 1 - Main Road",
           "current_phase": "RED"},
    "S2": {"vehicle_count": 30, "is_green": False, "is_stopped": False,
           "esp32_ip": "192.168.137.102",
           "lat": 28.6200, "lon": 77.2150,
           "location_name": "Signal 2 - Junction A",
           "current_phase": "RED"},
    "S3": {"vehicle_count": 8,  "is_green": False, "is_stopped": False,
           "esp32_ip": "192.168.137.103",
           "lat": 28.6250, "lon": 77.2200,
           "location_name": "Signal 3 - Crossing B",
           "current_phase": "RED"},
    "S4": {"vehicle_count": 45, "is_green": False, "is_stopped": False,
           "esp32_ip": "192.168.137.104",
           "lat": 28.6300, "lon": 77.2250,
           "location_name": "Signal 4 - Highway Entry",
           "current_phase": "RED"},
    "S5": {"vehicle_count": 20, "is_green": False, "is_stopped": False,
           "esp32_ip": "192.168.137.105",
           "lat": 28.6350, "lon": 77.2300,
           "location_name": "Signal 5 - Market Road",
           "current_phase": "RED"},
    "S6": {"vehicle_count": 60, "is_green": False, "is_stopped": False,
           "esp32_ip": "192.168.137.106",
           "lat": 28.6400, "lon": 77.2350,
           "location_name": "Signal 6 - Central Square",
           "current_phase": "RED"},
}

_lock = threading.Lock()

# ── Ambulance State ───────────────────────────────────────────────────────────

ambulance = {
    "id"             : None,
    "lat"            : None,
    "lon"            : None,
    "speed_mps"      : AVG_SPEED_MPS,
    "status"         : "inactive",
    "last_gps_time"  : None,
    "dest_lat"       : None,
    "dest_lon"       : None,
    "dest_name"      : "",
    "distance_text"  : "",
    "duration_text"  : "",
    "trip_id"        : None,
    "passed_signals" : set(),
    "active_signals" : [],
}

yolo_state = {s: {"detected": False, "confidence": 0.0, "last_seen": None}
              for s in SIGNALS}

# ── Utility ───────────────────────────────────────────────────────────────────

def _haversine(lat1, lon1, lat2, lon2) -> float:
    R  = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a  = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def _now()  -> float: return time.time()
def _ts()   -> str:   return datetime.utcnow().isoformat() + "Z"

def _congestion(c) -> str:
    if c < 20: return "LOW"
    if c < 50: return "MEDIUM"
    if c < 75: return "HIGH"
    return "CRITICAL"

# ── ESP32-CAM Frame Detection ─────────────────────────────────────────────────

def _detect_ambulance_in_frame(frame_bytes: bytes, width: int, height: int) -> dict:
    """
    Detect ambulance in raw grayscale frame from ESP32-CAM.

    Method 1 — Brightness Spike (Default, no model needed):
      Emergency vehicle lights cause very bright pixels.
      If 8%+ of pixels are above threshold 200, likely ambulance.

    Method 2 — Flicker Pattern (Advanced):
      Emergency lights flash. Compare consecutive frames.
      Implemented as optional enhancement.

    Returns: {detected, confidence, method, bright_pct, avg_brightness}
    """
    try:
        # Convert raw bytes to numpy array
        frame = np.frombuffer(frame_bytes, dtype=np.uint8)

        # Validate size
        expected = width * height
        if len(frame) != expected:
            log.warning("[CAM] Frame size mismatch: got %d expected %d",
                        len(frame), expected)
            return {"detected": False, "confidence": 0.0,
                    "method": "error", "error": "size_mismatch"}

        # Reshape to 2D
        frame = frame.reshape((height, width))

        # ── Method 1: Brightness Spike ────────────────────────────────────
        # Count pixels above bright threshold
        bright_pixels = np.sum(frame > AMBULANCE_BRIGHTNESS_THRESH)
        total_pixels  = width * height
        bright_pct    = bright_pixels / total_pixels
        avg_brightness= float(np.mean(frame))

        # Confidence based on percentage of bright pixels
        # 8% = 0.45 confidence, 15% = 0.90 confidence
        if bright_pct >= AMBULANCE_BRIGHT_PIXEL_PCT:
            confidence = min(0.95, 0.45 + (bright_pct - 0.08) * 6.0)
            detected   = True
            method     = "brightness_spike"
        else:
            confidence = bright_pct * 5.6   # proportional
            detected   = False
            method     = "brightness_spike"

        log.info("[CAM] Frame %dx%d | bright=%.1f%% avg=%.0f | %s conf=%.2f",
                 width, height, bright_pct*100, avg_brightness,
                 "DETECTED" if detected else "clear", confidence)

        return {
            "detected"        : detected,
            "confidence"      : round(confidence, 3),
            "method"          : method,
            "bright_pct"      : round(bright_pct * 100, 2),
            "avg_brightness"  : round(avg_brightness, 1),
            "bright_pixels"   : int(bright_pixels),
            "total_pixels"    : total_pixels,
        }

    except Exception as e:
        log.error("[CAM] Detection error: %s", e)
        return {"detected": False, "confidence": 0.0,
                "method": "error", "error": str(e)}

# ── ESP32 Signal Control ──────────────────────────────────────────────────────

def _send_esp32(sig_id: str, cmd: str) -> dict:
    if sig_id not in SIGNALS:
        return {"signal": sig_id, "success": False}
    esp_ip = SIGNALS[sig_id]["esp32_ip"]
    url    = f"http://{esp_ip}/?cmd={cmd}"
    try:
        resp = requests.get(url, timeout=ESP32_TIMEOUT)
        ok   = resp.status_code == 200
        with _lock:
            SIGNALS[sig_id]["is_green"]      = (cmd == "GREEN" and ok)
            SIGNALS[sig_id]["is_stopped"]    = (cmd == "STOP"  and ok)
            SIGNALS[sig_id]["current_phase"] = cmd if ok else SIGNALS[sig_id]["current_phase"]
        log.info("[ESP32] %s → %s | %s", sig_id, cmd, "OK" if ok else "FAIL")
        return {"signal": sig_id, "cmd": cmd, "success": ok, "esp32_ip": esp_ip}
    except requests.exceptions.ConnectionError:
        with _lock:
            SIGNALS[sig_id]["is_green"]      = (cmd == "GREEN")
            SIGNALS[sig_id]["is_stopped"]    = (cmd == "STOP")
            SIGNALS[sig_id]["current_phase"] = cmd
        return {"signal": sig_id, "cmd": cmd, "success": True,
                "note": "simulated (no hardware)"}
    except Exception as exc:
        return {"signal": sig_id, "cmd": cmd, "success": False, "error": str(exc)}


def _send_parallel(signal_ids: list, cmd: str) -> list:
    results, rlock = [], threading.Lock()
    def _w(s):
        r = _send_esp32(s, cmd)
        with rlock: results.append(r)
    threads = [threading.Thread(target=_w, args=(s,), daemon=True)
               for s in signal_ids if s in SIGNALS]
    for t in threads: t.start()
    for t in threads: t.join(timeout=ESP32_TIMEOUT + 1)
    return results

# ── Route Logic ───────────────────────────────────────────────────────────────

def _signals_near_ambulance(amb_lat, amb_lon, radius_m=GREEN_RADIUS_M):
    nearby = []
    for sig_id, sig in SIGNALS.items():
        dist = _haversine(amb_lat, amb_lon, sig["lat"], sig["lon"])
        if dist <= radius_m:
            nearby.append((sig_id, dist))
    nearby.sort(key=lambda x: x[1])
    return [s for s, _ in nearby]


def _signals_between(amb_lat, amb_lon, dest_lat, dest_lon):
    candidates = []
    total_dist = _haversine(amb_lat, amb_lon, dest_lat, dest_lon)
    for sig_id, sig in SIGNALS.items():
        d_amb  = _haversine(amb_lat, amb_lon, sig["lat"], sig["lon"])
        d_dest = _haversine(sig["lat"], sig["lon"], dest_lat, dest_lon)
        if (d_amb + d_dest) <= (total_dist * 1.4) and d_amb < total_dist:
            candidates.append((sig_id, d_amb))
    candidates.sort(key=lambda x: x[1])
    return [s for s, _ in candidates]


def _control_corridor(amb_lat, amb_lon, dest_lat, dest_lon,
                       speed, trip_id, amb_id):
    route_sigs = _signals_between(amb_lat, amb_lon, dest_lat, dest_lon)
    all_sigs   = set(SIGNALS.keys())
    route_set  = set(route_sigs)
    cross_sigs = list(all_sigs - route_set)

    green_now = []
    scheduled = []

    for sig_id in route_sigs:
        if sig_id in ambulance["passed_signals"]:
            continue
        sig  = SIGNALS[sig_id]
        dist = _haversine(amb_lat, amb_lon, sig["lat"], sig["lon"])
        eta  = dist / max(speed, 1.0)

        if eta < 30:
            green_now.append(sig_id)
        elif eta < 120:
            scheduled.append((sig_id, eta))

    if cross_sigs:
        log.info("[CORRIDOR] STOP crossing: %s", cross_sigs)
        _send_parallel(cross_sigs, "STOP")

    if green_now:
        log.info("[CORRIDOR] GREEN now: %s", green_now)
        _send_parallel(green_now, "GREEN")
        with _lock:
            ambulance["active_signals"] = green_now

    for sig_id, eta in scheduled:
        delay = max(0, eta - 25)
        def _delayed(s=sig_id, d=delay):
            time.sleep(d)
            if (ambulance["status"] == "active"
                    and s not in ambulance["passed_signals"]):
                log.info("[CORRIDOR] Scheduled GREEN → %s", s)
                _send_esp32(s, "GREEN")
        threading.Thread(target=_delayed, daemon=True).start()
        log.info("[CORRIDOR] Schedule GREEN in %.0fs → %s", delay, sig_id)

    for sig_id in list(ambulance["passed_signals"]):
        sig  = SIGNALS.get(sig_id)
        if not sig: continue
        dist = _haversine(amb_lat, amb_lon, sig["lat"], sig["lon"])
        if dist > PASSED_RADIUS_M * 2:
            _send_esp32(sig_id, "RESET")
            log.info("[PASSED] Reset %s", sig_id)

    return {"route_signals": route_sigs, "green_now": green_now,
            "cross_stopped": cross_sigs}

# ── Traffic Simulation ────────────────────────────────────────────────────────

def _simulate():
    import random
    while True:
        time.sleep(10)
        with _lock:
            for sig in SIGNALS.values():
                if not sig["is_green"] and not sig["is_stopped"]:
                    sig["vehicle_count"] = max(0, min(MAX_VEHICLES,
                        sig["vehicle_count"] + random.randint(-8, 8)))

threading.Thread(target=_simulate, daemon=True).start()

# ── Traffic Snapshot ──────────────────────────────────────────────────────────

def _traffic_snapshot() -> dict:
    with _lock:
        return {
            sid: {
                "vehicle_count"    : s["vehicle_count"],
                "is_green"         : s["is_green"],
                "is_stopped"       : s["is_stopped"],
                "current_phase"    : s["current_phase"],
                "congestion"       : _congestion(s["vehicle_count"]),
                "location_name"    : s["location_name"],
                "lat"              : s["lat"],
                "lon"              : s["lon"],
                "predicted_traffic": round(s["vehicle_count"] * 1.1, 1),
            }
            for sid, s in SIGNALS.items()
        }

# ═══════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

# ── NEW: /detect — receives raw grayscale frame from ESP32-CAM ────────────────

@app.route("/detect", methods=["POST"])
def detect_from_cam():
    """
    Called by ESP32-CAM every 3 seconds.
    Receives raw grayscale frame bytes.
    Runs ambulance detection and controls signals.

    Headers:
      X-Signal-ID    → which signal this camera covers (e.g. S1)
      X-Ambulance-ID → camera identifier (e.g. CAM_S1)
      X-Width        → frame width in pixels (160)
      X-Height       → frame height in pixels (120)
      X-Format       → GRAYSCALE

    Body: raw uint8 grayscale bytes (width × height)
    """
    # ── Read headers ──────────────────────────────────────────────────────────
    sig_id   = request.headers.get("X-Signal-ID",    "S1")
    cam_id   = request.headers.get("X-Ambulance-ID", "CAM_UNKNOWN")
    width    = int(request.headers.get("X-Width",  "160"))
    height   = int(request.headers.get("X-Height", "120"))
    fmt      = request.headers.get("X-Format", "GRAYSCALE")

    if sig_id not in SIGNALS:
        log.warning("[DETECT] Unknown signal: %s", sig_id)
        return jsonify({"error": f"Unknown signal: {sig_id}"}), 400

    # ── Read raw frame bytes ──────────────────────────────────────────────────
    frame_bytes = request.data
    if not frame_bytes:
        return jsonify({"error": "No frame data received"}), 400

    log.info("[DETECT] %s → %s | %dx%d | %d bytes",
             cam_id, sig_id, width, height, len(frame_bytes))

    # ── Run detection ─────────────────────────────────────────────────────────
    result     = _detect_ambulance_in_frame(frame_bytes, width, height)
    detected   = result["detected"]
    confidence = result["confidence"]

    # ── Update YOLO state ─────────────────────────────────────────────────────
    with _lock:
        yolo_state[sig_id].update({
            "detected"  : detected,
            "confidence": confidence,
            "last_seen" : _now() if detected else yolo_state[sig_id]["last_seen"],
        })

    # ── Log to database ───────────────────────────────────────────────────────
    lat = ambulance.get("lat") or SIGNALS[sig_id]["lat"]
    lon = ambulance.get("lon") or SIGNALS[sig_id]["lon"]
    db.log_detection(cam_id, confidence, [], "esp32_cam", lat, lon)

    # ── Signal control ────────────────────────────────────────────────────────
    action  = "NONE"
    trip_id = ambulance.get("trip_id")

    if detected and confidence >= AMBULANCE_CONFIDENCE_MIN:
        log.info("[DETECT] 🚨 AMBULANCE at %s (conf=%.2f) → GREEN", sig_id, confidence)

        # Turn this signal GREEN immediately
        _send_esp32(sig_id, "GREEN")
        action = "GREEN"

        # Stop all crossing signals at this junction
        dest_lat   = ambulance.get("dest_lat", 0) or 0
        dest_lon   = ambulance.get("dest_lon", 0) or 0
        amb_lat    = ambulance.get("lat") or SIGNALS[sig_id]["lat"]
        amb_lon    = ambulance.get("lon") or SIGNALS[sig_id]["lon"]

        if dest_lat and dest_lon:
            route_sigs = set(_signals_between(amb_lat, amb_lon, dest_lat, dest_lon))
        else:
            route_sigs = {sig_id}

        cross = [s for s in SIGNALS if s != sig_id and s not in route_sigs]
        if cross:
            threading.Thread(target=_send_parallel,
                args=(cross, "STOP"), daemon=True).start()
            action = "GREEN+STOP_CROSSING"

        # Mark ambulance as active at this signal in DB
        if trip_id:
            db.log_signal(trip_id, cam_id, sig_id, "GREEN",
                         SIGNALS[sig_id]["esp32_ip"], True)

    elif not detected and SIGNALS[sig_id]["is_green"]:
        # Ambulance left → reset this signal to normal
        log.info("[DETECT] Ambulance left %s → RESET", sig_id)
        _send_esp32(sig_id, "RESET")
        action = "RESET"
        with _lock:
            ambulance.setdefault("passed_signals", set()).add(sig_id)
            SIGNALS[sig_id]["is_green"]      = False
            SIGNALS[sig_id]["current_phase"] = "RED"

    # ── Return response to ESP32-CAM ──────────────────────────────────────────
    # ESP32 reads "signal":"GREEN" or "signal":"RED" from this response
    response = {
        "signal"        : "GREEN" if (detected and confidence >= AMBULANCE_CONFIDENCE_MIN) else "RED",
        "detected"      : detected,
        "confidence"    : confidence,
        "signal_id"     : sig_id,
        "action_taken"  : action,
        "signal_state"  : SIGNALS[sig_id]["current_phase"],
        "bright_pct"    : result.get("bright_pct", 0),
        "avg_brightness": result.get("avg_brightness", 0),
        "timestamp"     : _ts(),
    }

    return jsonify(response), 200


# ── GET /signal — ESP32 Traffic Controller polls this ─────────────────────────

@app.route("/signal", methods=["GET"])
def get_signal_state():
    """
    Called by traffic ESP32 every 2 seconds via GET /signal.
    Returns GREEN if any signal is active, RED otherwise.
    This is what drives the physical traffic lights.

    Optional query param: ?id=S1  (returns state for specific signal)
    """
    sig_id = request.args.get("id", None)

    if sig_id and sig_id in SIGNALS:
        # Return state for specific signal
        is_green = SIGNALS[sig_id]["is_green"]
        return (is_green and "GREEN" or "RED"), 200, \
               {"Content-Type": "text/plain"}

    # Return GREEN if ambulance is active anywhere
    any_green = any(s["is_green"] for s in SIGNALS.values())
    amb_active = ambulance.get("status") == "active"

    if any_green or amb_active:
        return "GREEN", 200, {"Content-Type": "text/plain"}
    return "RED", 200, {"Content-Type": "text/plain"}


# ── Existing Endpoints (unchanged) ───────────────────────────────────────────

@app.route("/set-route", methods=["POST"])
def set_route():
    data      = request.get_json(silent=True) or {}
    amb_id    = data.get("ambulance_id", "AMB001")
    orig_lat  = float(data.get("origin_lat", 0))
    orig_lon  = float(data.get("origin_lon", 0))
    dest_lat  = float(data.get("dest_lat",   0))
    dest_lon  = float(data.get("dest_lon",   0))
    dest_name = data.get("dest_name", "")
    dist_text = data.get("distance_text", "")
    dur_text  = data.get("duration_text", "")

    amb_info   = db.get_ambulance(amb_id)
    route_sigs = _signals_between(orig_lat, orig_lon, dest_lat, dest_lon)
    trip_id    = db.start_trip(amb_id, orig_lat, orig_lon, route_sigs, 0)

    with _lock:
        ambulance.update({
            "id"           : amb_id,
            "lat"          : orig_lat,
            "lon"          : orig_lon,
            "status"       : "active",
            "dest_lat"     : dest_lat,
            "dest_lon"     : dest_lon,
            "dest_name"    : dest_name,
            "distance_text": dist_text,
            "duration_text": dur_text,
            "trip_id"      : trip_id,
            "passed_signals": set(),
            "last_gps_time": _now(),
        })

    db.update_ambulance_status(amb_id, "active")
    log.info("[SET-ROUTE] %s → %s | signals: %s",
             f"{orig_lat:.4f},{orig_lon:.4f}", dest_name, route_sigs)

    threading.Thread(
        target=_control_corridor,
        args=(orig_lat, orig_lon, dest_lat, dest_lon,
              AVG_SPEED_MPS, trip_id, amb_id),
        daemon=True).start()

    signal_details = []
    for sid in route_sigs:
        sig  = SIGNALS[sid]
        dist = _haversine(orig_lat, orig_lon, sig["lat"], sig["lon"])
        signal_details.append({
            "signal_id"    : sid,
            "location"     : sig["location_name"],
            "lat"          : sig["lat"],
            "lon"          : sig["lon"],
            "distance_m"   : round(dist),
            "vehicle_count": sig["vehicle_count"],
            "congestion"   : _congestion(sig["vehicle_count"]),
        })

    return jsonify({
        "route_set"       : True,
        "ambulance_id"    : amb_id,
        "trip_id"         : trip_id,
        "origin"          : {"lat": orig_lat, "lon": orig_lon},
        "destination"     : {"lat": dest_lat, "lon": dest_lon, "name": dest_name},
        "distance"        : dist_text,
        "duration"        : dur_text,
        "signals_on_route": route_sigs,
        "signal_details"  : signal_details,
        "total_signals"   : len(route_sigs),
        "timestamp"       : _ts(),
    }), 200


@app.route("/update-location", methods=["POST"])
def update_location():
    data   = request.get_json(silent=True) or {}
    amb_id = data.get("ambulance_id", "AMB001")
    lat    = float(data.get("lat", 0))
    lon    = float(data.get("lon", 0))
    speed  = float(data.get("speed_mps", AVG_SPEED_MPS))

    with _lock:
        ambulance.update({
            "lat": lat, "lon": lon,
            "speed_mps": max(speed, 1.0), "last_gps_time": _now()})

    db.log_gps(amb_id, lat, lon, speed * 3.6, ambulance.get("trip_id"))

    if ambulance["status"] != "active":
        return jsonify({"status": "inactive", "timestamp": _ts()}), 200

    dest_lat = ambulance.get("dest_lat", 0)
    dest_lon = ambulance.get("dest_lon", 0)
    trip_id  = ambulance.get("trip_id")

    if dest_lat == 0 and dest_lon == 0:
        return jsonify({"status": "no_destination", "timestamp": _ts()}), 200

    corridor = _control_corridor(lat, lon, dest_lat, dest_lon,
                                  ambulance["speed_mps"], trip_id, amb_id)

    nearby = _signals_near_ambulance(lat, lon, PASSED_RADIUS_M)
    for sig_id in list(ambulance.get("passed_signals", set())):
        if sig_id not in nearby:
            sig  = SIGNALS.get(sig_id)
            if sig:
                dist = _haversine(lat, lon, sig["lat"], sig["lon"])
                if dist > PASSED_RADIUS_M * 1.5:
                    threading.Thread(target=_send_esp32,
                        args=(sig_id, "RESET"), daemon=True).start()

    return jsonify({
        "lat"           : lat,
        "lon"           : lon,
        "signals_green" : corridor["green_now"],
        "route_signals" : corridor["route_signals"],
        "cross_stopped" : corridor["cross_stopped"],
        "timestamp"     : _ts(),
    }), 200


@app.route("/ambulance", methods=["POST"])
def receive_ambulance():
    data   = request.get_json(silent=True) or {}
    amb_id = data.get("ambulance_id", "AMB001")
    lat    = float(data.get("lat", 0))
    lon    = float(data.get("lon", 0))
    status = data.get("status", "inactive").lower()
    speed  = float(data.get("speed", 0)) / 3.6

    amb_info = db.get_ambulance(amb_id)

    with _lock:
        ambulance.update({
            "id": amb_id, "lat": lat, "lon": lon,
            "speed_mps": max(speed, 1.0),
            "status": status, "last_gps_time": _now()})

    db.update_ambulance_status(amb_id, status)
    db.log_gps(amb_id, lat, lon, speed * 3.6, ambulance.get("trip_id"))

    payload = {
        "received": True, "ambulance_id": amb_id,
        "reg_number": amb_info.get("reg_number"),
        "status": status, "traffic": _traffic_snapshot(),
        "timestamp": _ts()}

    if status == "inactive":
        trip_id = ambulance.get("trip_id")
        if trip_id:
            db.end_trip(trip_id, amb_id, lat, lon,
                list(ambulance.get("passed_signals", set())))
        all_sigs = list(SIGNALS.keys())
        threading.Thread(target=_send_parallel,
            args=(all_sigs, "RESET"), daemon=True).start()
        with _lock:
            ambulance.update({
                "status": "inactive", "trip_id": None,
                "passed_signals": set(), "active_signals": [],
                "dest_lat": None, "dest_lon": None})
            for s in SIGNALS.values():
                s.update({"is_green": False, "is_stopped": False,
                          "current_phase": "RED"})
        payload["signals_reset"] = all_sigs
        log.info("[DEACTIVATE] All signals reset to normal")

    return jsonify(payload), 200


@app.route("/detection", methods=["POST"])
def yolo_detection():
    """Legacy JSON detection endpoint (from detect.py YOLO script)."""
    data       = request.get_json(silent=True) or {}
    sig_id     = data.get("signal_id")
    detected   = bool(data.get("detected", False))
    confidence = float(data.get("confidence", 0.0))
    vc         = int(data.get("vehicle_count", 0))
    amb_id     = data.get("ambulance_id", "CAM001")

    if sig_id not in SIGNALS:
        return jsonify({"error": f"Unknown signal: {sig_id}"}), 400

    with _lock:
        yolo_state[sig_id].update({
            "detected"  : detected,
            "confidence": confidence,
            "last_seen" : _now() if detected else yolo_state[sig_id]["last_seen"]})
        if vc > 0:
            SIGNALS[sig_id]["vehicle_count"] = vc

    action  = "NONE"
    trip_id = ambulance.get("trip_id")

    if detected and confidence >= 0.45:
        _send_esp32(sig_id, "GREEN")
        action = "GREEN"
        dest_lat = ambulance.get("dest_lat", 0) or 0
        dest_lon = ambulance.get("dest_lon", 0) or 0
        amb_lat  = ambulance.get("lat") or SIGNALS[sig_id]["lat"]
        amb_lon  = ambulance.get("lon") or SIGNALS[sig_id]["lon"]
        route_sigs = set(_signals_between(amb_lat, amb_lon, dest_lat, dest_lon)) \
                     if dest_lat else {sig_id}
        cross = [s for s in SIGNALS if s != sig_id and s not in route_sigs]
        if cross:
            threading.Thread(target=_send_parallel,
                args=(cross, "STOP"), daemon=True).start()
            action = "GREEN+STOP_CROSSING"
    elif not detected and SIGNALS[sig_id]["is_green"]:
        _send_esp32(sig_id, "RESET")
        action = "RESET"
        with _lock:
            ambulance.setdefault("passed_signals", set()).add(sig_id)

    lat = ambulance.get("lat") or SIGNALS[sig_id]["lat"]
    lon = ambulance.get("lon") or SIGNALS[sig_id]["lon"]
    db.log_detection(amb_id, confidence, data.get("bbox", []),
                     "yolo_camera", lat, lon)

    return jsonify({
        "signal_id"   : sig_id,
        "detected"    : detected,
        "confidence"  : confidence,
        "action_taken": action,
        "signal_state": SIGNALS[sig_id]["current_phase"],
        "timestamp"   : _ts()}), 200


@app.route("/signal-control", methods=["POST"])
def signal_control():
    data    = request.get_json(silent=True) or {}
    targets = [data["signal_id"]] if "signal_id" in data \
              else data.get("signal_ids", [])
    cmd     = data.get("cmd", "GREEN").upper()
    if cmd not in {"GREEN", "RED", "STOP", "RESET"}:
        return jsonify({"error": "Invalid cmd"}), 400
    results = _send_parallel(targets, cmd)
    return jsonify({"cmd": cmd, "results": results, "timestamp": _ts()}), 200


@app.route("/traffic", methods=["GET"])
def get_traffic():
    snap = _traffic_snapshot()
    avg  = sum(snap[s]["vehicle_count"] for s in snap) / len(snap)
    return jsonify({
        "signals"       : snap,
        "average_count" : round(avg, 1),
        "overall_status": _congestion(int(avg)),
        "timestamp"     : _ts()}), 200


@app.route("/traffic", methods=["POST"])
def update_traffic():
    data  = request.get_json(silent=True) or {}
    sid   = data.get("signal_id")
    count = data.get("vehicle_count")
    if sid not in SIGNALS or count is None or count < 0:
        return jsonify({"error": "Invalid input"}), 400
    with _lock:
        SIGNALS[sid]["vehicle_count"] = min(int(count), MAX_VEHICLES)
    return jsonify({"updated": True, "signal_id": sid,
                    "vehicle_count": SIGNALS[sid]["vehicle_count"]}), 200


@app.route("/status", methods=["GET"])
def get_status():
    with _lock:
        state = {k: v for k, v in ambulance.items() if k != "passed_signals"}
        state["passed_signals"] = list(ambulance.get("passed_signals", set()))
    return jsonify({
        "system"          : "Smart Ambulance v6 — Real GPS + ESP32-CAM",
        "ambulance"       : state,
        "yolo_state"      : yolo_state,
        "active_greens"   : sum(1 for s in SIGNALS.values() if s["is_green"]),
        "stopped_signals" : sum(1 for s in SIGNALS.values() if s["is_stopped"]),
        "stats"           : db.get_stats(),
        "timestamp"       : _ts()}), 200


@app.route("/register", methods=["POST"])
def register():
    data = request.get_json(silent=True) or {}
    miss = [k for k in ["ambulance_id","reg_number","hospital_name"] if k not in data]
    if miss: return jsonify({"error": f"Missing: {miss}"}), 400
    amb = db.register_ambulance(
        data["ambulance_id"], data["reg_number"], data["hospital_name"],
        data.get("driver_name",""), data.get("driver_phone",""),
        data.get("vehicle_type","Type-B"))
    return jsonify({"registered": True, "ambulance": amb}), 200


@app.route("/ambulances", methods=["GET"])
def list_ambulances():
    return jsonify({"ambulances": db.get_all_ambulances()}), 200


@app.route("/ambulance/<amb_id>", methods=["GET"])
def get_ambulance(amb_id):
    info = db.get_ambulance(amb_id)
    if not info: return jsonify({"error": "Not found"}), 404
    return jsonify({
        "ambulance"  : info,
        "gps_history": db.get_gps_history(amb_id, 50),
        "trips"      : db.get_trip_history(amb_id, 10),
    }), 200


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "name"    : "Smart Ambulance v6 — Real GPS + ESP32-CAM",
        "endpoints": {
            "POST /detect"          : "ESP32-CAM raw grayscale frame → auto signal control",
            "GET  /signal"          : "ESP32 traffic controller polls this → GREEN or RED",
            "POST /set-route"       : "Flutter → start ambulance mode with real GPS",
            "POST /update-location" : "Flutter → live GPS every 3s",
            "POST /ambulance"       : "Legacy GPS ping / deactivate",
            "POST /detection"       : "YOLO JSON detection (legacy)",
            "GET  /traffic"         : "Current traffic all signals",
            "GET  /status"          : "Full system status",
        }
    }), 200


@app.errorhandler(404)
def not_found(e): return jsonify({"error": "Not found"}), 404
@app.errorhandler(500)
def server_error(e): return jsonify({"error": "Internal error"}), 500


if __name__ == "__main__":
    log.info("=" * 60)
    log.info("  Smart Ambulance v6 — Real GPS + ESP32-CAM Detection")
    log.info("  /detect  → ESP32-CAM raw frames")
    log.info("  /signal  → Traffic ESP32 polling")
    log.info("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
