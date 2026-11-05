"""Step 11. Demo: driver state in real time (webcam or video file).
Models: models\\eb_drowsy.joblib and models\\eb_attention.joblib (from 10_train_final.py).

Features are computed EXACTLY as in training:
  - EAR, MAR, head angles, face crop: formulas from 02_extract_features.py (copied below);
  - blendshapes: FaceLandmarker with output_face_blendshapes, as in 02b, in the order stored in the model;
  - embeddings: MobileNetV2 (imagenet, avg pool, 224x224, preprocess_input), as in m0b_honest_colab.ipynb;
  - 5 frames per second; calibration = median of the first 50 frames with a face (10 s), as in training;
  - 1 s window = last 5 frames with a face; no face -> fixed answer (alert / not_looking_road).
The first 10 s are calibration: sit still and look ahead (as at the road).

Run (from the project root):
  python code\\11_demo_live.py                         webcam 0
  python code\\11_demo_live.py --camera 1              another camera
  python code\\11_demo_live.py --video "path.mp4"      video file
  python code\\11_demo_live.py --log results\\live.csv --record results\\live.mp4
Keys in the window: q - quit, c - recalibrate.
At the end the per-stage timing is printed (Section 6.5 of the paper).

Needs: pip install tensorflow
"""
import argparse
import os
import sys
import time
from collections import deque

import cv2
import joblib
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")

# ---- formulas from 02_extract_features.py (do not change) ----
R_EYE = [33, 160, 158, 133, 153, 144]
L_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_V = [(81, 178), (13, 14), (311, 402)]
MOUTH_H = (78, 308)
POSE_IDX = [1, 152, 263, 33, 291, 61]
MODEL_3D = np.array([(0.0, 0.0, 0.0), (0.0, -63.6, -12.5), (-43.3, 32.7, -26.0),
                     (43.3, 32.7, -26.0), (-28.9, -28.9, -24.1), (28.9, -28.9, -24.1)], dtype=np.float64)


def dist(a, b):
    return float(np.linalg.norm(a - b))


def ear(pts, idx):
    p = [pts[i] for i in idx]
    return (dist(p[1], p[5]) + dist(p[2], p[4])) / (2.0 * dist(p[0], p[3]) + 1e-9)


def mar(pts):
    v = sum(dist(pts[a], pts[b]) for a, b in MOUTH_V)
    return v / (3.0 * dist(pts[MOUTH_H[0]], pts[MOUTH_H[1]]) + 1e-9)


def head_pose(pts, w, h):
    img = np.array([pts[i] for i in POSE_IDX], dtype=np.float64)
    cam = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(MODEL_3D, img, cam, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return np.nan, np.nan, np.nan
    rot, _ = cv2.Rodrigues(rvec)
    angles = cv2.RQDecomp3x3(rot)[0]
    wrap = lambda a: float((a + 90.0) % 180.0 - 90.0)
    return wrap(angles[1]), wrap(angles[0]), wrap(angles[2])


def crop_face(frame, pts, margin=0.25, size=224):
    h, w = frame.shape[:2]
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0) * (1 + margin)
    xa, ya = int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0))
    xb, yb = int(min(cx + side / 2, w)), int(min(cy + side / 2, h))
    if xb - xa < 10 or yb - ya < 10:
        return None
    return cv2.resize(frame[ya:yb, xa:xb], (size, size))
# ---- end of formulas from 02 ----


class OnlineFeatures:
    """Calibration + window as in training (09/10): computed on frames with a face only."""
    BASE = ["ear", "mar", "yaw", "pitch", "roll"]

    def __init__(self, n_cal, window):
        self.n_cal, self.window = n_cal, window
        self.reset()

    def reset(self):
        self.cal_base, self.cal_blend, self.cal_emb = [], [], []
        self.med = None
        self.hist = deque(maxlen=self.window)
        self.last_base = None
        self.nan_run = 0

    @property
    def calibrated(self):
        return self.med is not None

    def ffill(self, base):
        """Like groupby.ffill(limit=3): missing angles are filled with the previous value at most 3 times in a row."""
        if np.isnan(base).any():
            if self.last_base is not None and self.nan_run < 3:
                self.nan_run += 1
                base = np.where(np.isnan(base), self.last_base, base)
            else:
                return None
        else:
            self.nan_run = 0
        self.last_base = base
        return base

    def add(self, base, blend, emb):
        """Returns a feature dict, or None while calibrating."""
        if not self.calibrated:
            self.cal_base.append(base); self.cal_blend.append(blend); self.cal_emb.append(emb)
            if len(self.cal_base) >= self.n_cal:
                b = np.array(self.cal_blend, np.float64)
                self.med = dict(base=np.median(np.array(self.cal_base), axis=0),
                                blend=np.nanmedian(b, axis=0) if np.isfinite(b).any() else np.zeros(52),
                                emb=np.median(np.array(self.cal_emb), axis=0))
                # calibration frames also enter the window history (in training the window runs over the whole recording)
                for x in self.cal_base[-self.window:]:
                    self.hist.append(x - self.med["base"])
            return None
        bc = base - self.med["base"]
        self.hist.append(bc)
        H = np.array(self.hist)
        k = self.window
        feats = dict(zip(self.BASE, bc))
        std = lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        for j, c in [(0, "ear"), (1, "mar")]:
            feats[f"{c}_mean{k}"] = H[:, j].mean(); feats[f"{c}_min{k}"] = H[:, j].min()
            feats[f"{c}_max{k}"] = H[:, j].max(); feats[f"{c}_std{k}"] = std(H[:, j])
        for j, c in [(2, "yaw"), (3, "pitch")]:
            feats[f"{c}_mean{k}"] = H[:, j].mean(); feats[f"{c}_std{k}"] = std(H[:, j])
        bs_ok = float(np.isfinite(blend).all())
        bcal = np.nan_to_num(blend - self.med["blend"], nan=0.0)
        for i in range(52):
            feats[f"bs_{i}"] = bcal[i]
        feats["bs_ok"] = bs_ok
        return feats, emb - self.med["emb"]


def make_vector(bundle, feats, emb_c):
    xa = [feats[n] for n in bundle["feats_A"]]
    xb = [feats[n] for n in bundle["feats_B"]]
    z = bundle["pca"].transform(bundle["scaler"].transform(emb_c[None, :]))[0]
    return np.r_[xa, xb, z].astype(np.float32)[None, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--video", default="")
    ap.add_argument("--models", default="models")
    ap.add_argument("--log", default="", help="CSV with the output for every frame")
    ap.add_argument("--record", default="", help="save the raw camera video (for liveness experiments)")
    ap.add_argument("--no-window", action="store_true", help="no display window (for a video file)")
    ap.add_argument("--skip", type=float, default=0.0, help="video file: skip this many seconds at the start")
    a = ap.parse_args()

    models = {}
    for t in ["drowsy", "attention"]:
        fn = os.path.join(a.models, f"eb_{t}.joblib")
        if not os.path.exists(fn):
            sys.exit(f"{fn} not found: first run python code\\10_train_final.py --task {t}")
        b = joblib.load(fn)
        b["rf"].n_jobs = 1  # faster for a single frame
        models[t] = b
    ref = models["drowsy"]
    fps, n_cal, window = ref["fps"], int(round(ref["calib_sec"] * ref["fps"])), ref["window"]
    blend_names = ref["blend_names"]

    try:
        import tensorflow as tf
    except ImportError:
        sys.exit("tensorflow not found: pip install tensorflow")
    import mediapipe as mp
    net = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet",
                                            input_shape=(224, 224, 3), pooling="avg")
    prep = tf.keras.applications.mobilenet_v2.preprocess_input
    # compiled call: no Keras overhead per frame (same weights and output)
    infer = tf.function(lambda x: net(x, training=False),
                        input_signature=[tf.TensorSpec([1, 224, 224, 3], tf.float32)])
    infer(tf.zeros((1, 224, 224, 3)))  # warm-up

    if not os.path.exists(MODEL_PATH):
        sys.exit(f"{MODEL_PATH} not found (the same file as for 02/02b)")
    lm = mp.tasks.vision.FaceLandmarker.create_from_options(mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_faces=1, output_face_blendshapes=True))

    is_file = bool(a.video)
    cap = cv2.VideoCapture(a.video if is_file else a.camera)
    if not cap.isOpened():
        sys.exit("Could not open the camera/video")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(src_fps / fps)), 1)
    writer = None
    log = open(a.log, "w", encoding="utf-8") if a.log else None
    if log:
        log.write("t,frame_idx,face,ear,mar,yaw,pitch,roll,calibrated,drowsy,attention,"
                  "p_alert,p_eyes_closed,p_yawning,p_not_looking_road\n")

    of = OnlineFeatures(n_cal, window)
    times = {"face (MediaPipe)": [], "features": [], "MobileNetV2": [], "RF (2 models)": [], "total": []}
    t_start, last_proc, i = time.time(), -1e9, 0
    state = {"drowsy": "...", "attention": "...", "p": {}}
    face_now = False
    print("First 10 s: calibration - sit still and look ahead. q - quit, c - recalibrate.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        if is_file and i / src_fps < a.skip:
            i += 1
            continue
        if a.record and not is_file:
            if writer is None:
                h0, w0 = frame.shape[:2]
                writer = cv2.VideoWriter(a.record, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w0, h0))
            writer.write(frame)
        take = (i % step == 0) if is_file else (now - last_proc >= 1.0 / fps)
        ts_ms = (i / src_fps * 1000.0) if is_file else (now - t_start) * 1000.0
        if take:
            last_proc = now
            t0 = time.perf_counter()
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)),
                                      int(ts_ms))
            t1 = time.perf_counter()
            face_now, base, row = False, None, [np.nan] * 5
            if res.face_landmarks:
                pts = np.array([(p.x * w, p.y * h) for p in res.face_landmarks[0]])
                crop = crop_face(frame, pts)
                if crop is not None:
                    e_ = (ear(pts, R_EYE) + ear(pts, L_EYE)) / 2
                    yaw, pitch, roll = head_pose(pts, w, h)
                    raw = np.array([e_, mar(pts), yaw, pitch, roll], np.float64)
                    row = list(raw)
                    base = of.ffill(raw)
                    blend = np.full(52, np.nan)
                    if res.face_blendshapes:
                        sc = {c.category_name: c.score for c in res.face_blendshapes[0]}
                        names = blend_names or [c.category_name for c in res.face_blendshapes[0]]
                        blend = np.array([sc.get(n, np.nan) for n in names[:52]], np.float64)
            t2 = time.perf_counter()
            t3 = t4 = t2
            if base is not None:
                face_now = True
                # as in 02: the face crop was saved as JPG and read back as RGB
                _, jpg = cv2.imencode(".jpg", crop)
                crop_rgb = cv2.cvtColor(cv2.imdecode(jpg, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                emb = infer(tf.constant(prep(crop_rgb.astype(np.float32))[None])).numpy()[0]
                t3 = time.perf_counter()
                out = of.add(base, blend, emb)
                if out is not None:
                    feats, emb_c = out
                    for t, b in models.items():
                        x = make_vector(b, feats, emb_c)
                        pr = b["rf"].predict_proba(x)[0]
                        state[t] = b["classes"][int(pr.argmax())]
                        state["p"].update(dict(zip(b["classes"], pr)))
                    t4 = time.perf_counter()
            if not face_now and of.calibrated:
                state["drowsy"], state["attention"] = models["drowsy"]["noface_label"], models["attention"]["noface_label"]
                state["p"] = {}
            if of.calibrated and face_now:
                times["face (MediaPipe)"].append(t1 - t0); times["features"].append(t2 - t1)
                times["MobileNetV2"].append(t3 - t2); times["RF (2 models)"].append(t4 - t3)
                times["total"].append(t4 - t0)
            if log:
                p = state["p"]
                log.write(f"{now - t_start:.3f},{i},{int(face_now)}," + ",".join(f"{v:.5f}" for v in row) +
                          f",{int(of.calibrated)},{state['drowsy'] if of.calibrated else ''},"
                          f"{state['attention'] if of.calibrated else ''}," +
                          ",".join(f"{p.get(c, np.nan):.4f}" for c in
                                   ["alert", "eyes_closed", "yawning", "not_looking_road"]) + "\n")
        if not a.no_window:
            disp = frame.copy()
            if not of.calibrated:
                txt = [f"Calibrating {len(of.cal_base)}/{n_cal}: sit still, look ahead"]
                col = (0, 200, 255)
            else:
                d, at = state["drowsy"], state["attention"]
                txt = [f"drowsy: {d}", f"attention: {at}", "face: " + ("yes" if face_now else "NO")]
                col = (0, 0, 255) if (d != "alert" or at != "looking_road") else (0, 200, 0)
            for k, s in enumerate(txt):
                cv2.putText(disp, s, (15, 35 + 35 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
            cv2.imshow("driver state (q - exit, c - recalibrate)", disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                of.reset(); state.update(drowsy="...", attention="...", p={})
                print("Recalibrating...")
        i += 1

    cap.release()
    if writer is not None:
        writer.release()
    if log:
        log.close()
    cv2.destroyAllWindows()
    print("\nSpeed per stage (ms per frame, median; frames with a face after calibration only):")
    for k, v in times.items():
        if v:
            print(f"  {k:18s} {1000 * np.median(v):7.1f}")
    if times["total"]:
        print(f"  => up to {1 / np.median(times['total']):.0f} frames/s on this computer "
              f"(the system uses {fps} frames/s)")


if __name__ == "__main__":
    main()
