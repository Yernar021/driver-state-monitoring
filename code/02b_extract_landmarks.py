"""Step 2b. Save MediaPipe face landmarks and blendshapes (used by the GRU, the EB models and liveness).
02_extract_features.py is left unchanged.

For each video, EXACTLY the frames already in features.csv are processed (same frame_idx),
so everything matches labels.csv row by row on (video_id, frame_idx).

Saved per video (one file: data_processed\\dmd\\landmarks\\<video_id>.npz):
  frame_idx   (N,)        frame number in the source video
  face        (N,)        1 = face found
  pts         (N, 90, 3)  x, y (fractions of frame width/height), z for 90 points: eyes, irises, lips, rigid points
  blend       (N, 52)     MediaPipe blendshapes (eyeBlinkLeft, jawOpen, ...), 0..1
  pose        (N, 4, 4)   facial transformation matrix
  point_ids, blend_names  which points and which blendshapes

Run (from the project root; can be left overnight):
  python code\\02b_extract_landmarks.py --input data_raw\\dmd\\extracted --features data_processed\\dmd\\features.csv --out data_processed\\dmd\\landmarks --workers 2
Restartable: finished videos are skipped. An error in one video does not stop the others.
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")

LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
IRIS = [468, 469, 470, 471, 472, 473, 474, 475, 476, 477]
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
RIGID = [1, 4, 6, 10, 152, 168, 234, 454]  # nose, nose bridge, forehead, chin, cheekbones
POINT_IDS = np.array(LEFT_EYE + RIGHT_EYE + IRIS + LIPS_OUTER + LIPS_INNER + RIGID, dtype=np.int16)
N_PTS, N_BLEND = len(POINT_IDS), 52


def make_landmarker():
    import mediapipe as mp
    opts = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(opts)


def video_path(base, vid):
    return os.path.join(base, *vid.split("__")) + ".mp4"


def process(job):
    vid, frames, base, out_dir = job
    out_path = os.path.join(out_dir, vid + ".npz")
    if os.path.exists(out_path):
        return vid, "skip", 0.0
    t0 = time.time()
    try:
        import cv2
        import mediapipe as mp
        path = video_path(base, vid)
        if not os.path.exists(path):
            return vid, f"NO VIDEO: {path}", 0.0
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        want = set(int(f) for f in frames)
        last = max(want)
        lm = make_landmarker()
        n = len(frames)
        order = {f: k for k, f in enumerate(sorted(want))}
        fidx = np.array(sorted(want), dtype=np.int32)
        face = np.zeros(n, dtype=np.uint8)
        pts = np.full((n, N_PTS, 3), np.nan, dtype=np.float32)
        blend = np.full((n, N_BLEND), np.nan, dtype=np.float32)
        pose = np.full((n, 4, 4), np.nan, dtype=np.float32)
        blend_names = []
        i = 0
        while i <= last:
            if i in want:
                ok, img = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                          int(i * 1000.0 / fps))
                k = order[i]
                if res.face_landmarks:
                    allp = res.face_landmarks[0]
                    if len(allp) > int(POINT_IDS.max()):
                        face[k] = 1
                        pts[k] = [(allp[j].x, allp[j].y, allp[j].z) for j in POINT_IDS]
                    if res.face_blendshapes:
                        cats = res.face_blendshapes[0]
                        if not blend_names:
                            blend_names = [c.category_name for c in cats]
                        blend[k, :len(cats)] = [c.score for c in cats][:N_BLEND]
                    if res.facial_transformation_matrixes:
                        pose[k] = np.asarray(res.facial_transformation_matrixes[0], dtype=np.float32)
            else:
                if not cap.grab():
                    break
            i += 1
        cap.release()
        lm.close()
        tmp = out_path + ".tmp.npz"
        np.savez_compressed(tmp, frame_idx=fidx, face=face, pts=pts, blend=blend, pose=pose,
                            point_ids=POINT_IDS, blend_names=np.array(blend_names))
        os.replace(tmp, out_path)  # the file appears only when complete
        return vid, f"ok, face {face.mean():.2f}", time.time() - t0
    except Exception as e:  # one bad video must not stop the whole run
        return vid, f"ERROR: {e!r}", time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="extracted folder (the same as for 02)")
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--only", default="", help="process only videos whose name contains this string (for testing)")
    a = ap.parse_args()

    if not os.path.exists(MODEL_PATH):
        sys.exit(f"Model not found: {MODEL_PATH} (the same one 02 uses)")
    os.makedirs(a.out, exist_ok=True)
    f = pd.read_csv(a.features, usecols=["video_id", "frame_idx"])
    groups = f.groupby("video_id")["frame_idx"].apply(list)
    vids = [v for v in groups.index if a.only in v]
    vids.sort(key=lambda v: (not v.startswith("drowsiness"), v))  # drowsiness first
    jobs = [(v, groups[v], a.input, a.out) for v in vids]
    todo = sum(not os.path.exists(os.path.join(a.out, v + ".npz")) for v in vids)
    print(f"Videos: {len(vids)}, left to process: {todo}, workers: {a.workers}", flush=True)

    t0, done = time.time(), 0
    with Pool(a.workers) as pool:
        for vid, status, sec in pool.imap_unordered(process, jobs):
            if status == "skip":
                continue
            done += 1
            print(f"[{done}/{todo}] {status} | {sec/60:.1f} min | {vid}", flush=True)
    print(f"Done in {(time.time()-t0)/60:.0f} min: {a.out}")


if __name__ == "__main__":
    main()
