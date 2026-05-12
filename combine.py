# =========================================================
# MAXIMUM SPEED — 3-thread pipeline: decode / infer / encode
# run simultaneously so no stage waits for another.
# =========================================================
from ultralytics import YOLO
import cv2
import numpy as np
import joblib
import mediapipe as mp
from collections import Counter
import sys, os, threading
from queue import Queue
import torch

# ── CPU tuning ────────────────────────────────────────────
torch.set_num_threads(os.cpu_count() or 4)
cv2.setNumThreads(os.cpu_count() or 4)

VIDEO_PATH  = "test_video.mp4"
OUTPUT_PATH = "static/output_result.mp4"
os.makedirs("static", exist_ok=True)

# ── Read video properties ONCE ────────────────────────────
_cap = cv2.VideoCapture(VIDEO_PATH)
VIDEO_FPS = _cap.get(cv2.CAP_PROP_FPS) or 25.0
SRC_W     = int(_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
SRC_H     = int(_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
_cap.release()

TARGET_H = 480
if SRC_H > TARGET_H:
    _s   = TARGET_H / SRC_H
    OUT_W, OUT_H = int(SRC_W * _s), TARGET_H
else:
    OUT_W, OUT_H = SRC_W, SRC_H
NEEDS_RESIZE = (OUT_W != SRC_W or OUT_H != SRC_H)

FONT       = cv2.FONT_HERSHEY_SIMPLEX
SENTINEL   = None   # signals end-of-stream


def get_writer(path, fps, w, h):
    for cc in ["avc1", "H264", "X264", "mp4v"]:
        wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*cc), fps, (w, h))
        if wr.isOpened():
            return wr
    return cv2.VideoWriter(path, -1, fps, (w, h))


# =========================================================
# ⚡ PHASE 1 — YOLO unsafe scan (fastest possible settings)
# =========================================================
yolo_model = YOLO("yolov8n.pt")

# Use half precision if a CUDA GPU is available
USE_GPU = torch.cuda.is_available()
if USE_GPU:
    yolo_model = yolo_model.half()

unsafe, unsafe_reason = False, ""

for result in yolo_model.predict(
    source=VIDEO_PATH,
    conf=0.25,
    verbose=False,
    vid_stride=15,          # ⚡ scan every 15th frame
    stream=True,
    imgsz=160,              # ⚡ tiny — enough for vehicle detection
    half=USE_GPU,
):
    if unsafe:
        break
    if result.boxes is not None:
        for box in result.boxes:
            if yolo_model.names[int(box.cls[0])] in ("car","motorcycle","bus","truck"):
                unsafe       = True
                unsafe_reason = yolo_model.names[int(box.cls[0])].upper() + " DETECTED"
                break


# =========================================================
# 🚨 UNSAFE — 2-thread pipeline: decode → draw+encode
# =========================================================
if unsafe:
    print("UNSAFE -", unsafe_reason, flush=True)

    raw_q = Queue(maxsize=128)

    def _reader(q):
        cap = cv2.VideoCapture(VIDEO_PATH)
        while cap.isOpened():
            ok, frame = cap.read()
            q.put(frame if ok else SENTINEL)
            if not ok:
                break
        cap.release()

    threading.Thread(target=_reader, args=(raw_q,), daemon=True).start()
    out = get_writer(OUTPUT_PATH, VIDEO_FPS, OUT_W, OUT_H)

    while True:
        frame = raw_q.get()
        if frame is SENTINEL:
            break
        if NEEDS_RESIZE:
            frame = cv2.resize(frame, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
        fh, fw = frame.shape[:2]
        cv2.rectangle(frame, (10, 10), (fw-10, fh-10), (0,0,255), 4)
        cv2.putText(frame, f'UNSAFE: {unsafe_reason}', (50,80), FONT, 1.2, (0,0,255), 3)
        out.write(frame)

    out.release()
    sys.exit()


# =========================================================
# 🟢 SAFE — 3-thread pipeline: decode / infer / encode
# =========================================================
print("NOTHING UNSAFE\nRunning SAFE activity detection...\n", flush=True)

activity_model = joblib.load("activity_model.pkl")

SAMPLE_EVERY = 15           # ⚡ pose inference on every 15th frame
WARMUP_FRAMES = 60          # collect predictions for first ~2.4 s then lock label

# Preallocate feature buffer — no per-frame numpy allocation
_feat_buf = np.empty((1, 8), dtype=np.float32)

pose = mp.solutions.pose.Pose(
    model_complexity=0,
    smooth_landmarks=False,
    enable_segmentation=False,
    min_detection_confidence=0.4,   # ⚡ slightly lower = faster early exit
    min_tracking_confidence=0.4,
)


def _angle(ax, ay, bx, by, cx, cy):
    """Compute joint angle without any numpy allocation."""
    r = (  (cx-bx)*(cy-by) - (cx-bx)*(ay-by)  # wrong — use atan2 below
         + (cx-bx)*(cy-by))                     # placeholder; real below
    r = abs(
        np.degrees(
            np.arctan2(cy-by, cx-bx) - np.arctan2(ay-by, ax-bx)
        )
    )
    return 360 - r if r > 180 else r


def fill_features(lm, buf):
    """Fill preallocated (1,8) buffer. No heap allocation."""
    def a(i,j,k):
        return _angle(lm[i].x,lm[i].y, lm[j].x,lm[j].y, lm[k].x,lm[k].y)
    buf[0,0] = a(11,13,15)
    buf[0,1] = a(12,14,16)
    buf[0,2] = a(23,25,27)
    buf[0,3] = a(24,26,28)
    buf[0,4] = a(13,11,23)
    buf[0,5] = a(14,12,24)
    buf[0,6] = a(11,23,25)
    buf[0,7] = a(12,24,26)


# ── Queues ────────────────────────────────────────────────
# raw_q:  raw frames from disk
# ann_q:  (annotated_frame, bbox_or_None) ready to encode
raw_q = Queue(maxsize=128)
ann_q = Queue(maxsize=64)

# ── Shared state ──────────────────────────────────────────
_state_lock    = threading.Lock()
_final_activity = [None]        # set once by inference thread
_label_ready   = threading.Event()


# ── Thread 1: decode ─────────────────────────────────────
def reader_thread():
    cap = cv2.VideoCapture(VIDEO_PATH)
    while cap.isOpened():
        ok, frame = cap.read()
        raw_q.put(frame if ok else SENTINEL)
        if not ok:
            break
    cap.release()


# ── Thread 2: inference ──────────────────────────────────
def inference_thread():
    predictions  = []
    frame_idx    = 0
    last_bbox    = None
    warmup_done  = False
    buf_frames   = []           # hold raw frames until label is decided

    while True:
        frame = raw_q.get()
        if frame is SENTINEL:
            # flush buffered frames
            if not warmup_done:
                act = Counter(predictions).most_common(1)[0][0] if predictions else "Unknown"
                with _state_lock:
                    _final_activity[0] = act
                _label_ready.set()
                print("FINAL ACTIVITY DETECTED:", act, flush=True)
            for bf, bb in buf_frames:
                ann_q.put((bf, bb))
            ann_q.put(SENTINEL)
            return

        # ── resize once per frame ──────────────────────────
        if NEEDS_RESIZE:
            frame = cv2.resize(frame, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
        fh, fw = frame.shape[:2]

        # ── pose inference on sampled frames ──────────────
        if frame_idx % SAMPLE_EVERY == 0:
            # Fast BGR→RGB: flip channel axis in-place view (no copy)
            rgb = frame[:, :, ::-1]
            results = pose.process(rgb)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                xs = [int(l.x * fw) for l in lm]
                ys = [int(l.y * fh) for l in lm]
                last_bbox = (
                    max(0, min(xs)-20), max(0, min(ys)-20),
                    min(fw, max(xs)+20), min(fh, max(ys)+20),
                )
                if not warmup_done:
                    fill_features(lm, _feat_buf)
                    predictions.append(activity_model.predict(_feat_buf)[0])

        frame_idx += 1

        # ── warmup buffering ──────────────────────────────
        if not warmup_done:
            buf_frames.append((frame, last_bbox))
            if frame_idx >= WARMUP_FRAMES:
                act = Counter(predictions).most_common(1)[0][0] if predictions else "Unknown"
                with _state_lock:
                    _final_activity[0] = act
                _label_ready.set()
                print("FINAL ACTIVITY DETECTED:", act, flush=True)
                warmup_done = True
                for bf, bb in buf_frames:
                    ann_q.put((bf, bb))
                buf_frames.clear()
        else:
            ann_q.put((frame, last_bbox))


# ── Thread 3: encode (writer lives here) ─────────────────
def writer_thread():
    _label_ready.wait()         # block until label is decided
    act = _final_activity[0]
    out = get_writer(OUTPUT_PATH, VIDEO_FPS, OUT_W, OUT_H)

    while True:
        item = ann_q.get()
        if item is SENTINEL:
            break
        frame, bbox = item
        if bbox:
            x0, y0, x1, y1 = bbox
            cv2.rectangle(frame, (x0,y0), (x1,y1), (0,255,0), 2)
            cv2.putText(frame, f'SAFE Activity: {act}',
                        (x0, max(y0-10, 20)), FONT, 0.9, (0,255,0), 2)
        out.write(frame)

    out.release()


# ── Launch all 3 threads ─────────────────────────────────
t1 = threading.Thread(target=reader_thread,    daemon=True)
t2 = threading.Thread(target=inference_thread, daemon=True)
t3 = threading.Thread(target=writer_thread,    daemon=True)

t1.start(); t2.start(); t3.start()
t1.join();  t2.join();  t3.join()