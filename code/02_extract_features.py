"""Step 2. Video -> frames (face crops) + EAR, MAR and head-pose features.

Install:  pip install opencv-python mediapipe numpy pandas
Run:
  python 02_extract_features.py --input video_or_folder --out out --fps 5 --pattern _rgb_face
Output in the out folder:
  frames/<video_id>/<number>.jpg   224x224 face crop
  features.csv                     video_id, frame_idx, time_s, face_found, ear, mar, yaw, pitch, roll
Labels (class, person) are added later by 03_build_labels.py.
"""
import os, argparse, urllib.request
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")


def make_detector():
    """Returns a function (rgb_frame, time_ms) -> list of landmarks or None.
    Works with both the old mediapipe interface (solutions) and the new one (tasks)."""
    if hasattr(mp, "solutions"):
        mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)

        def det(rgb, ts_ms):
            r = mesh.process(rgb)
            return r.multi_face_landmarks[0].landmark if r.multi_face_landmarks else None
        return det

    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    if not os.path.exists(MODEL_PATH):
        print("Downloading the face model (once)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO, num_faces=1)
    landmarker = vision.FaceLandmarker.create_from_options(opts)

    def det(rgb, ts_ms):
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        r = landmarker.detect_for_video(img, int(ts_ms))
        return r.face_landmarks[0] if r.face_landmarks else None
    return det


VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".webm"}

# MediaPipe Face Mesh landmark indices
R_EYE = [33, 160, 158, 133, 153, 144]
L_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_V = [(81, 178), (13, 14), (311, 402)]
MOUTH_H = (78, 308)
# points for head pose: nose tip, chin, outer eye corners, mouth corners
POSE_IDX = [1, 152, 263, 33, 291, 61]
MODEL_3D = np.array([
    (0.0, 0.0, 0.0),          # nose tip
    (0.0, -63.6, -12.5),      # chin
    (-43.3, 32.7, -26.0),     # left eye (outer corner)
    (43.3, 32.7, -26.0),      # right eye (outer corner)
    (-28.9, -28.9, -24.1),    # left mouth corner
    (28.9, -28.9, -24.1),     # right mouth corner
], dtype=np.float64)


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
    angles = cv2.RQDecomp3x3(rot)[0]  # degrees: pitch, yaw, roll (approximately)
    # wrap angles into (-90, 90]: otherwise "jumps" such as -175 instead of ~5 occur
    wrap = lambda a: float((a + 90.0) % 180.0 - 90.0)
    return wrap(angles[1]), wrap(angles[0]), wrap(angles[2])  # yaw, pitch, roll


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


def process_video(path, vid_id, out_dir, target_fps, detector):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("  could not open:", path)
        return []
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(int(round(src_fps / target_fps)), 1)
    fdir = os.path.join(out_dir, "frames", vid_id)
    os.makedirs(fdir, exist_ok=True)
    rows, i, saved = [], 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            h, w = frame.shape[:2]
            lms = detector(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), i / src_fps * 1000.0)
            row = dict(video_id=vid_id, frame_idx=i, time_s=round(i / src_fps, 3),
                       face_found=0, ear=np.nan, mar=np.nan,
                       yaw=np.nan, pitch=np.nan, roll=np.nan)
            if lms is not None:
                pts = np.array([(p.x * w, p.y * h) for p in lms])
                e = (ear(pts, R_EYE) + ear(pts, L_EYE)) / 2
                yaw, pitch, roll = head_pose(pts, w, h)
                face = crop_face(frame, pts)
                if face is not None:
                    cv2.imwrite(os.path.join(fdir, f"{saved:06d}.jpg"), face)
                    saved += 1
                    row.update(face_found=1, ear=e, mar=mar(pts),
                               yaw=yaw, pitch=pitch, roll=roll)
            rows.append(row)
        i += 1
    cap.release()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--pattern", default="",
                    help="use only files whose name contains this string, e.g. _rgb_face")
    a = ap.parse_args()

    if os.path.isdir(a.input):
        vids = []
        for dp, _, fs in os.walk(a.input):
            for f in fs:
                if os.path.splitext(f)[1].lower() in VIDEO_EXT and a.pattern in f:
                    vids.append(os.path.join(dp, f))
        vids.sort()
        base = a.input
    else:
        vids, base = [a.input], os.path.dirname(a.input)

    os.makedirs(a.out, exist_ok=True)
    csv_path = os.path.join(a.out, "features.csv")
    done = set()
    if os.path.exists(csv_path):  # restartable: finished videos are skipped
        done = set(pd.read_csv(csv_path, usecols=["video_id"])["video_id"].unique())

    for n, p in enumerate(vids, 1):
        vid_id = os.path.splitext(os.path.relpath(p, base))[0].replace(os.sep, "__")
        if vid_id in done:
            continue
        print(f"[{n}/{len(vids)}] {vid_id}")
        rows = process_video(p, vid_id, a.out, a.fps, make_detector())  # new detector for every video
        if rows:
            pd.DataFrame(rows).to_csv(csv_path, mode="a", header=not os.path.exists(csv_path),
                                      index=False)
    print("Done:", csv_path)


if __name__ == "__main__":
    main()
