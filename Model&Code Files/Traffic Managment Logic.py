import cv2
import requests
from ultralytics import YOLO
import numpy
import threading

# ======================================================
# MODEL
# ======================================================
MODEL_PATH = r"C:\Users\Lenovo\Downloads\best 92.pt"

model_a = YOLO(MODEL_PATH)
model_b = YOLO(MODEL_PATH)

model_a.overrides["imgsz"] = 320
model_b.overrides["imgsz"] = 320

TRACKER_CFG = "bytetrack.yaml"

names   = ['bus', 'car', 'motorcycle', 'truck', 'ambulance', 'fire truck']
weights = {
    "motorcycle": 0.5,
    "car":        0.5,
    "bus":        3,
    "truck":      3,
    "ambulance":  3,
    "fire truck": 3,
}

EMERGENCY_CLASSES = {"ambulance", "fire truck"}

CLASS_COLORS = {
    "car":        (0,   255, 100),
    "motorcycle": (0,   200, 255),
    "bus":        (255, 150,   0),
    "truck":      (200,   0, 255),
    "ambulance":  (0,     0, 255),
    "fire truck": (0,     0, 255),
}

# ======================================================
# API
# ======================================================
TIMING_API    = "https://judgingly-cicatrisant-milly.ngrok-free.dev/signal-timings/"
COGNITION_API = "https://judgingly-cicatrisant-milly.ngrok-free.dev/cognition-records/"

HEADERS = {
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsImV4cCI6MTc4MDM5NzM1Nn0.k0i425a__P3FigttP70TwiypWFXKEtimbluOPuLVwiQ",
    "Content-Type":  "application/json"
}


def send_signal_data(direction, green, red, flow, traffic_mode, mode="ai-controlled"):
    def _send():
        data = {
            "direction":    direction,
            "green_time":   int(green),
            "yellow_time":  2,
            "red_time":     int(red),
            "mode":         mode,
            "applied_by":   145,
            "traffic_flow": int(flow),
            "traffic_mode": traffic_mode
        }
        try:
            r = requests.post(TIMING_API, json=data, headers=HEADERS, timeout=5)
            print(f"✅ Signal [{traffic_mode.upper()}] {direction}: "
                  f"green={int(green)}s red={int(red)}s | {r.status_code}")
        except Exception as e:
            print(f"❌ Signal error {direction}: {e}")
    threading.Thread(target=_send, daemon=True).start()

def send_cognition(camera_id, vehicle_count, level, class_counts=None):
    def _send():
        data = {
            "camera_id":        camera_id,
            "vehicle_count":    int(vehicle_count),
            "vehicle_types":    class_counts or {},
            "congestion_level": int(level)
        }
        try:
            r = requests.post(COGNITION_API, json=data, headers=HEADERS, timeout=5)
            print(f"📊 Cognition cam {camera_id}: count={int(vehicle_count)} "
                  f"types={class_counts} level={int(level)} | {r.status_code}")
        except Exception as e:
            print(f"❌ Cognition error cam {camera_id}: {e}")
    threading.Thread(target=_send, daemon=True).start()


def flush_cognition(cam_key):
    """Send and reset cognition data for one camera."""
    s     = state[cam_key]
    level = compute_congestion_level(s["cognition_counter"])
    send_cognition(s["camera_id"], s["cognition_counter"], level, s["class_counts"])
    s["cognition_counter"] = 0
    s["class_counts"]      = {}


# ======================================================
# VIDEO PATHS
# ======================================================
VIDEO_A_PATH = r"C:\Users\Lenovo\Downloads\WhatsApp Video 2026-05-22 at 00.50.07.mp4"
VIDEO_B_PATH = r"C:\Users\Lenovo\Downloads\IMG_8737 (1).mov"

# For live RPi streams swap the above two lines with:
# VIDEO_A_PATH = "http://192.168.1.51:8080/cam_a"
# VIDEO_B_PATH = "http://192.168.1.51:8080/cam_b"

cap_a = cv2.VideoCapture(VIDEO_A_PATH)
cap_b = cv2.VideoCapture(VIDEO_B_PATH)

cap_a.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap_b.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap_a.isOpened():
    raise RuntimeError("Could not open video / stream A")
if not cap_b.isOpened():
    raise RuntimeError("Could not open video / stream B")

fps_a = cap_a.get(cv2.CAP_PROP_FPS) or 30
fps_b = cap_b.get(cv2.CAP_PROP_FPS) or 30

frame_width_a  = int(cap_a.get(cv2.CAP_PROP_FRAME_WIDTH))  or 640
frame_height_a = int(cap_a.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
frame_width_b  = int(cap_b.get(cv2.CAP_PROP_FRAME_WIDTH))  or 640
frame_height_b = int(cap_b.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

print(f"Video A: {frame_width_a} x {frame_height_a}")
print(f"Video B: {frame_width_b} x {frame_height_b}")

cv2.namedWindow("Traffic System", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Traffic System", 1280, 640)

# ======================================================
# LINE CONFIG
# ======================================================
OFFSET = 5

line_y_a = int(frame_height_a * 0.60)
line_y_b = int(frame_height_b * 0.80)

queue_top_margin    = 120
queue_bottom_margin = 15

queue_y1_a = max(0, line_y_a - queue_top_margin)
queue_y2_a = max(0, line_y_a - queue_bottom_margin)
queue_y1_b = max(0, line_y_b - queue_top_margin)
queue_y2_b = max(0, line_y_b - queue_bottom_margin)

# ======================================================
# PER-CAMERA STATE
# ======================================================
def make_state(name, direction, camera_id, fps, fw, fh, line_y, qy1, qy2):
    return {
        "name":               name,
        "direction":          direction,
        "camera_id":          camera_id,
        "fps":                fps,
        "frame_width":        fw,
        "frame_height":       fh,
        "line_y":             line_y,
        "queue_y1":           qy1,
        "queue_y2":           qy2,
        "last_y":             {},
        "last_side":          {},
        "counted_ids_window": set(),
        "counted_ids_total":  set(),
        "cross_times":        [],
        "queue_ids":          set(),
        "cognition_counter":  0,
        "class_counts":       {},
        "total_count":        0,
        "recent_flow":        0.0,
        "queue_count":        0,
        "starvation":         0.0,
        "demand":             0.0,
        "emergency_active":   False,
        "emergency_class":    None,
        "emergency_ids":      set(),
        "emergency_hold_timer": 0.0,
        "ever_detected":      False,   # True only after first real crossing
    }

state = {
    "A": make_state("North", "north_south", 1, fps_a,
                    frame_width_a, frame_height_a,
                    line_y_a, queue_y1_a, queue_y2_a),
    "B": make_state("East",  "east_west",   2, fps_b,
                    frame_width_b, frame_height_b,
                    line_y_b, queue_y1_b, queue_y2_b),
}

# ======================================================
# ADAPTIVE PARAMETERS
# ======================================================
FLOW_WINDOW_SECONDS = 5
COGNITION_WINDOW    = 30   # seconds between cognition records (real-time)
                           # timer only starts after first vehicle is counted

MIN_GREEN     = 10
MAX_GREEN     = 45
BUFFER        = 4
SWITCH_MARGIN = 0.2

WF = 0.50
WQ = 0.35
WS = 0.15

EMERGENCY_HOLD_SECONDS = 8

DISPLAY_EVERY_N = 2
display_counter = 0

# ======================================================
# PHASE / SYSTEM STATE
# ======================================================
phase                = None
phase_timer          = 0.0
current_green_target = MIN_GREEN
system_active        = False

# Cognition timers — None means "not started yet"
# They start the moment the first vehicle is counted on each camera
cognition_timer_a    = None
cognition_timer_b    = None

# ======================================================
# HELPERS
# ======================================================
def clamp(v, lo, hi): return max(lo, min(hi, v))


def compute_congestion_level(count):
    avg = count / 10.0
    if avg < 2:  return 1
    if avg < 5:  return 2
    return 3


def cleanup_old_crossings(cam_state, now_sec):
    cam_state["cross_times"] = [
        t for t in cam_state["cross_times"]
        if now_sec - t <= FLOW_WINDOW_SECONDS
    ]
    cam_state["recent_flow"] = len(cam_state["cross_times"]) / FLOW_WINDOW_SECONDS


def compute_demand(cam_state):
    cam_state["demand"] = (
        WF * cam_state["recent_flow"] +
        WQ * cam_state["queue_count"] +
        WS * cam_state["starvation"]
    )


def compute_green_times(da, db):
    total = da + db
    if total <= 0:
        return MIN_GREEN, MIN_GREEN
    u  = 2 * MIN_GREEN
    ga = clamp(u * (da / total), MIN_GREEN, MAX_GREEN)
    gb = clamp(u * (db / total), MIN_GREEN, MAX_GREEN)
    return ga, gb


# ======================================================
# SIGNAL HELPERS
# ======================================================
def send_both_signals(green_cam, green_duration, red_duration,
                      green_flow, red_flow, mode="ai-controlled"):
    """
    Sends signal data for BOTH directions every phase switch.
    The green camera gets traffic_mode='green', the other gets 'red'.
    """
    red_cam = "B" if green_cam == "A" else "A"

    send_signal_data(
        direction    = state[green_cam]["direction"],
        green        = green_duration,
        red          = red_duration,
        flow         = green_flow,
        traffic_mode = "green",
        mode         = mode
    )
    send_signal_data(
        direction    = state[red_cam]["direction"],
        green        = red_duration,
        red          = green_duration,
        flow         = red_flow,
        traffic_mode = "red",
        mode         = mode
    )


# ======================================================
# EMERGENCY LOGIC
# ======================================================
def update_emergency_state(cam_state, detected_ids, detected_class, dt):
    if detected_ids:
        cam_state["emergency_ids"]        = detected_ids
        cam_state["emergency_class"]      = detected_class
        cam_state["emergency_active"]     = True
        cam_state["emergency_hold_timer"] = 0.0
    elif cam_state["emergency_active"]:
        cam_state["emergency_hold_timer"] += dt
        if cam_state["emergency_hold_timer"] >= EMERGENCY_HOLD_SECONDS:
            cam_state["emergency_active"] = False
            cam_state["emergency_class"]  = None
            cam_state["emergency_ids"]    = set()
            print(f"[EMERGENCY] {cam_state['name']}: cleared")


def resolve_emergency_phase(sa, sb):
    ea, eb = sa["emergency_active"], sb["emergency_active"]
    if not ea and not eb:  return None
    if ea and not eb:      return "A"
    if eb and not ea:      return "B"
    def pri(c): return 0 if c == "ambulance" else 1
    pa, pb = pri(sa["emergency_class"]), pri(sb["emergency_class"])
    if pa != pb:
        return "A" if pa < pb else "B"
    return "A" if sa["emergency_hold_timer"] <= sb["emergency_hold_timer"] else "B"


# ======================================================
# DRAW HELPERS
# ======================================================
def draw_label(frame, text, x, y, font_scale=0.55, thickness=1,
               text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 3
    cv2.rectangle(frame,
                  (x - pad,      y - th - pad),
                  (x + tw + pad, y + bl + pad),
                  bg_color, -1)
    cv2.putText(frame, text, (x, y), font, font_scale,
                text_color, thickness, cv2.LINE_AA)


def draw_bounding_box(frame, x1, y1, x2, y2, class_name, obj_id, conf=None):
    color      = CLASS_COLORS.get(class_name, (200, 200, 200))
    is_emerg   = class_name in EMERGENCY_CLASSES
    thickness  = 3 if is_emerg else 2
    corner_len = max(12, int((x2 - x1) * 0.15))

    for pts in [
        [(x1, y1 + corner_len), (x1, y1), (x1 + corner_len, y1)],
        [(x2 - corner_len, y1), (x2, y1), (x2, y1 + corner_len)],
        [(x1, y2 - corner_len), (x1, y2), (x1 + corner_len, y2)],
        [(x2 - corner_len, y2), (x2, y2), (x2, y2 - corner_len)],
    ]:
        cv2.polylines(frame, [numpy.array(pts)],
                      False, color, thickness, cv2.LINE_AA)

    if is_emerg:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

    conf_str = f" {conf:.2f}" if conf is not None else ""
    draw_label(frame, f"{class_name} #{obj_id}{conf_str}",
               x1, y1 - 4, font_scale=0.52, thickness=1,
               text_color=color, bg_color=(20, 20, 20))

    cv2.circle(frame, ((x1 + x2) // 2, y2), 4, color, -1, cv2.LINE_AA)


# ======================================================
# FRAME PROCESSING
# ======================================================
def process_camera_frame(frame, results, cam_state, now_sec):
    cam_state["queue_ids"]       = set()
    detected_emergency_ids       = set()
    detected_emergency_class     = None

    if not results or results[0].boxes is None:
        cam_state["queue_count"] = 0
        cleanup_old_crossings(cam_state, now_sec)
        return frame, detected_emergency_ids, detected_emergency_class

    boxes = results[0].boxes
    confs = boxes.conf.tolist() if boxes.conf is not None else [None] * len(boxes)

    for box, conf in zip(boxes, confs):
        if box.id is None:
            continue
        obj_id     = int(box.id)
        cls_id     = int(box.cls)
        if cls_id >= len(names):
            continue
        class_name = names[cls_id]
        if class_name not in weights:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cy = y2

        if class_name in EMERGENCY_CLASSES:
            detected_emergency_ids.add(obj_id)
            detected_emergency_class = class_name

        if obj_id in cam_state["last_y"]:
            prev_y   = cam_state["last_y"][obj_id]
            smooth_y = int(0.7 * prev_y + 0.3 * cy)
        else:
            prev_y   = cy
            smooth_y = cy

        crossed = prev_y < cam_state["line_y"] and smooth_y >= cam_state["line_y"]
        if crossed:
            if obj_id not in cam_state["counted_ids_window"]:
                cam_state["counted_ids_window"].add(obj_id)
                cam_state["cognition_counter"] += 1
                cam_state["cross_times"].append(now_sec)
                cam_state["class_counts"][class_name] = (
                    cam_state["class_counts"].get(class_name, 0) + 1
                )
                print(f"[COUNT] {cam_state['name']} | ID={obj_id} | {class_name}")

            if obj_id not in cam_state["counted_ids_total"]:
                cam_state["counted_ids_total"].add(obj_id)
                cam_state["total_count"] += 1
                cam_state["ever_detected"] = True   # only flips on real crossing

        if cam_state["queue_y1"] <= cy <= cam_state["queue_y2"]:
            cam_state["queue_ids"].add(obj_id)

        cam_state["last_y"][obj_id] = smooth_y

        draw_bounding_box(frame, x1, y1, x2, y2, class_name, obj_id,
                          conf=float(conf) if conf is not None else None)

    cam_state["queue_count"] = len(cam_state["queue_ids"])
    cleanup_old_crossings(cam_state, now_sec)
    return frame, detected_emergency_ids, detected_emergency_class


# ======================================================
# OVERLAY
# ======================================================
def draw_camera_overlay(frame, cam_state, is_green, green_left, red_left,
                        other_name, is_emergency_override, waiting):
    h, w         = frame.shape[:2]
    scale_factor = max(w, h) / 640.0

    small    = max(0.45, 0.50 * scale_factor)
    medium   = max(0.52, 0.60 * scale_factor)
    line_gap = int(18 * scale_factor)
    pad_x    = int(8  * scale_factor)
    panel_w  = int(240 * scale_factor)
    panel_h  = int(230 * scale_factor)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.50, frame, 0.50, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    y    = int(16 * scale_factor)

    def put(text, fscale, color, bold=False):
        nonlocal y
        thick = max(1, int(2 * scale_factor)) if bold else max(1, int(1 * scale_factor))
        cv2.putText(frame, text, (pad_x, y), font, fscale,
                    color, thick, cv2.LINE_AA)
        y += line_gap + int(fscale * 10)

    if waiting:
        put(f"[ {cam_state['name']} ]", medium, (255, 255, 255), bold=True)
        put("WAITING...",               medium, (255, 200,   0), bold=True)
        put("No vehicle counted yet",   small,  (200, 200, 200))
    else:
        status_color = (0, 255, 0) if is_green else (0, 0, 255)
        status_text  = "GREEN" if is_green else "RED"

        put(f"[ {cam_state['name']} ]",                     medium, (255, 255, 255), bold=True)
        put(f"Status : {status_text}",                      medium, status_color,    bold=True)
        put(f"Green  : {green_left:.1f}s",                  small,  (200, 255, 200))
        put(f"Red    : {red_left:.1f}s",                    small,  (200, 200, 255))
        put(f"Count  : {cam_state['total_count']}",         small,  (255, 255, 255))
        put(f"Flow   : {cam_state['recent_flow']:.2f} v/s", small,  (255, 255, 255))
        put(f"Queue  : {cam_state['queue_count']}",         small,  (255, 255, 255))
        put(f"Demand : {cam_state['demand']:.2f}",          small,  (255, 255, 255))

    if cam_state["emergency_active"]:
        bh = int(28 * scale_factor)
        cv2.rectangle(frame, (0, h - bh), (w, h), (0, 0, 180), -1)
        em_text  = f"EMERGENCY: {cam_state['emergency_class'].upper()} DETECTED"
        em_scale = max(0.4, 0.55 * scale_factor)
        (tw, _), _ = cv2.getTextSize(em_text, font, em_scale, 2)
        cv2.putText(frame, em_text,
                    ((w - tw) // 2, h - int(8 * scale_factor)),
                    font, em_scale, (255, 255, 255), 2, cv2.LINE_AA)

    if is_emergency_override:
        ov_scale = max(0.35, 0.45 * scale_factor)
        (tw, _), _ = cv2.getTextSize("OVERRIDE ACTIVE", font, ov_scale, 1)
        draw_label(frame, "OVERRIDE ACTIVE",
                   (w - tw) // 2, int(16 * scale_factor),
                   font_scale=ov_scale,
                   text_color=(0, 0, 255), bg_color=(255, 255, 255))

    lthick = max(1, int(2 * scale_factor))
    cv2.line(frame,
             (0, cam_state["line_y"]), (w, cam_state["line_y"]),
             (0, 255, 80), lthick, cv2.LINE_AA)
    draw_label(frame, "COUNT",
               w - int(65 * scale_factor),
               cam_state["line_y"] - int(4 * scale_factor),
               font_scale=max(0.3, 0.40 * scale_factor),
               text_color=(0, 255, 80), bg_color=(20, 20, 20))

    return frame


# ======================================================
# MAIN LOOP
# ======================================================
last_time = cv2.getTickCount() / cv2.getTickFrequency()

print("🚦 Traffic system started — waiting for first vehicle to cross the line...")

while cap_a.isOpened() and cap_b.isOpened():
    ret_a, frame_a = cap_a.read()
    ret_b, frame_b = cap_b.read()

    if not ret_a or not ret_b:
        if VIDEO_A_PATH.startswith("http"):
            continue
        break

    now = cv2.getTickCount() / cv2.getTickFrequency()
    dt  = now - last_time
    last_time = now

    results_a = model_a.track(frame_a, persist=True, verbose=False, tracker=TRACKER_CFG)
    results_b = model_b.track(frame_b, persist=True, verbose=False, tracker=TRACKER_CFG)

    frame_a, em_ids_a, em_cls_a = process_camera_frame(frame_a, results_a, state["A"], now)
    frame_b, em_ids_b, em_cls_b = process_camera_frame(frame_b, results_b, state["B"], now)

    update_emergency_state(state["A"], em_ids_a, em_cls_a, dt)
    update_emergency_state(state["B"], em_ids_b, em_cls_b, dt)

    # ── Start cognition timer the moment first crossing happens ──
    if cognition_timer_a is None and state["A"]["ever_detected"]:
        cognition_timer_a = now
        print("[COGNITION] Camera A timer started after first crossing")

    if cognition_timer_b is None and state["B"]["ever_detected"]:
        cognition_timer_b = now
        print("[COGNITION] Camera B timer started after first crossing")

    # ── Send cognition every 30 real-time seconds ────────────────
    if cognition_timer_a is not None and now - cognition_timer_a >= COGNITION_WINDOW:
        flush_cognition("A")
        cognition_timer_a = now

    if cognition_timer_b is not None and now - cognition_timer_b >= COGNITION_WINDOW:
        flush_cognition("B")
        cognition_timer_b = now

    # ======================================================
    # STARTUP LOGIC — first crossing decides who goes green
    # ======================================================
    if not system_active:
        if state["A"]["ever_detected"] or state["B"]["ever_detected"]:
            system_active = True

            if state["A"]["ever_detected"] and not state["B"]["ever_detected"]:
                phase = "A_GREEN"
                print("[START] First crossing on North → North gets GREEN")
            elif state["B"]["ever_detected"] and not state["A"]["ever_detected"]:
                phase = "B_GREEN"
                print("[START] First crossing on East → East gets GREEN")
            else:
                phase = "A_GREEN" if (
                    state["A"]["demand"] >= state["B"]["demand"]
                ) else "B_GREEN"
                print(f"[START] Simultaneous crossing → {phase}")

            phase_timer = 0.0

            # Send initial signal for both directions
            green_cam = "A" if phase == "A_GREEN" else "B"
            send_both_signals(
                green_cam    = green_cam,
                green_duration = MIN_GREEN,
                red_duration   = MIN_GREEN + BUFFER,
                green_flow   = 0,
                red_flow     = 0
            )

    # ── Still waiting — draw overlay and continue ─────────────────
    if not system_active:
        frame_a = draw_camera_overlay(
            frame_a, state["A"], False, 0, 0,
            state["B"]["name"], False, waiting=True)
        frame_b = draw_camera_overlay(
            frame_b, state["B"], False, 0, 0,
            state["A"]["name"], False, waiting=True)

        display_counter += 1
        if display_counter % DISPLAY_EVERY_N == 0:
            th  = 640
            sa  = th / frame_a.shape[0]
            sb  = th / frame_b.shape[0]
            sha = cv2.resize(frame_a, (int(frame_a.shape[1] * sa), th))
            shb = cv2.resize(frame_b, (int(frame_b.shape[1] * sb), th))
            cv2.imshow("Traffic System", cv2.hconcat([sha, shb]))

        if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
            break
        continue

    # ======================================================
    # NORMAL OPERATION
    # ======================================================

    if phase == "A_GREEN":
        state["A"]["starvation"] = 0.0
        state["B"]["starvation"] += dt
    else:
        state["B"]["starvation"] = 0.0
        state["A"]["starvation"] += dt

    compute_demand(state["A"])
    compute_demand(state["B"])
    green_a, green_b = compute_green_times(
        state["A"]["demand"], state["B"]["demand"])

    phase_timer += dt

    # ── Emergency override ────────────────────────────────
    emergency_cam      = resolve_emergency_phase(state["A"], state["B"])
    emergency_override = emergency_cam is not None

    if emergency_override:
        required_phase = f"{emergency_cam}_GREEN"
        other_cam      = "B" if emergency_cam == "A" else "A"

        if phase != required_phase:
            print(f"[EMERGENCY] Forcing {emergency_cam}_GREEN — "
                  f"{state[emergency_cam]['emergency_class']} on "
                  f"{state[emergency_cam]['name']}")

            send_both_signals(
                green_cam      = emergency_cam,
                green_duration = EMERGENCY_HOLD_SECONDS,
                red_duration   = MAX_GREEN + BUFFER,
                green_flow     = round(state[emergency_cam]["recent_flow"] * FLOW_WINDOW_SECONDS),
                red_flow       = round(state[other_cam]["recent_flow"] * FLOW_WINDOW_SECONDS),
                mode           = "emergency-override"
            )

            phase       = required_phase
            phase_timer = 0.0
            state[other_cam]["counted_ids_window"].clear()

        current_green_target = EMERGENCY_HOLD_SECONDS

    else:
        # ── Normal adaptive phase logic ───────────────────
        if phase == "A_GREEN":
            current_green_target = green_a
            if phase_timer >= MIN_GREEN:
                should_switch = (
                    phase_timer >= MAX_GREEN or
                    state["B"]["demand"] > state["A"]["demand"] + SWITCH_MARGIN
                )
                if should_switch:
                    # B is about to go green
                    send_both_signals(
                        green_cam      = "B",
                        green_duration = green_b,
                        red_duration   = green_b + BUFFER,
                        green_flow     = round(state["B"]["recent_flow"] * FLOW_WINDOW_SECONDS),
                        red_flow       = round(state["A"]["recent_flow"] * FLOW_WINDOW_SECONDS)
                    )
                    phase       = "B_GREEN"
                    phase_timer = 0.0
                    state["A"]["counted_ids_window"].clear()
        else:
            current_green_target = green_b
            if phase_timer >= MIN_GREEN:
                should_switch = (
                    phase_timer >= MAX_GREEN or
                    state["A"]["demand"] > state["B"]["demand"] + SWITCH_MARGIN
                )
                if should_switch:
                    # A is about to go green
                    send_both_signals(
                        green_cam      = "A",
                        green_duration = green_a,
                        red_duration   = green_a + BUFFER,
                        green_flow     = round(state["A"]["recent_flow"] * FLOW_WINDOW_SECONDS),
                        red_flow       = round(state["B"]["recent_flow"] * FLOW_WINDOW_SECONDS)
                    )
                    phase       = "A_GREEN"
                    phase_timer = 0.0
                    state["B"]["counted_ids_window"].clear()

    # ── Display timing ────────────────────────────────────
    if phase == "A_GREEN":
        green_left_a = max(0.0, current_green_target - phase_timer)
        red_left_a   = 0.0
        green_left_b = 0.0
        red_left_b   = green_left_a + BUFFER
    else:
        green_left_b = max(0.0, current_green_target - phase_timer)
        red_left_b   = 0.0
        green_left_a = 0.0
        red_left_a   = green_left_b + BUFFER

    frame_a = draw_camera_overlay(
        frame_a, state["A"], phase == "A_GREEN",
        green_left_a, red_left_a, state["B"]["name"],
        emergency_override and emergency_cam == "A",
        waiting=False
    )
    frame_b = draw_camera_overlay(
        frame_b, state["B"], phase == "B_GREEN",
        green_left_b, red_left_b, state["A"]["name"],
        emergency_override and emergency_cam == "B",
        waiting=False
    )

    display_counter += 1
    if display_counter % DISPLAY_EVERY_N == 0:
        target_h = 640
        sa       = target_h / frame_a.shape[0]
        sb       = target_h / frame_b.shape[0]
        show_a   = cv2.resize(frame_a, (int(frame_a.shape[1] * sa), target_h))
        show_b   = cv2.resize(frame_b, (int(frame_b.shape[1] * sb), target_h))
        cv2.imshow("Traffic System", cv2.hconcat([show_a, show_b]))

    if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
        break

cap_a.release()
cap_b.release()
cv2.destroyAllWindows()
print("🛑 Traffic system stopped.")