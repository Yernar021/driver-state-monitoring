"""Step 16. Learned attack detector: live person or attack (printed photo / photo on a screen / video on a screen).
The decision is made on a 5 s window (1 s step).

Window features (fixed in advance):
  M (motion):  blinks (eyeBlink blendshapes and EAR), non-rigid landmark motion
               (residual after affine alignment: close to 0 for a flat photo), rigid motion
               (face shift/scale, head-angle spread), facial activity (spread of blendshapes);
  I (image):   sharpness, saturation, brightness and its flicker, glare, high frequencies and spectral peaks (moire),
               computed on the 224x224 face crop, as in 02.
  MI = M + I.
Model: random forest (300 trees), as everywhere.

Rules fixed in advance:
  - frames with two large faces (the second > 30% of the first's area) are not used; the largest face is taken;
  - a window is used if a single face is found in at least 50% of its frames;
  - the first and last 3 s of every recording are discarded (start/stop of recording);
  - only the first 60 s after trimming are used from each recording (all recordings weigh the same);
  - a recording is excluded entirely if a face is found in less than 50% of its frames (the list is printed).
Evaluation:
  - if at least 3 people have both live and attack recordings: by person (the test person is not in training);
  - otherwise: PRELIMINARY time split inside each recording (first half train, second half test,
    windows do not overlap), i.e. same people and recordings, so the result is optimistic.
Metrics (ISO/IEC 30107-3): APCER - share of attack windows accepted as live (per attack type);
  BPCER - share of live windows rejected as attacks; ACER = (max APCER + BPCER) / 2.

Run (from the project root):
  python code\\16_liveness.py
Needs: face_landmarker.task in the code folder (the same file as for 02/02b/11).
Per-frame features are cached in results\\spoof\\frames_<file>.npz (re-runs are fast).
"""
import argparse
import os
import re
import sys
import time

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

TYPES = ["live_normal", "live_still", "photo_print", "photo_screen", "video_replay"]
ATTACKS = ["photo_print", "photo_screen", "video_replay"]
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".3gp"}
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
R_EYE = [33, 160, 158, 133, 153, 144]
L_EYE = [362, 385, 387, 263, 373, 380]
POSE_IDX = [1, 152, 263, 33, 291, 61]
MODEL_3D = np.array([(0.0, 0.0, 0.0), (0.0, -63.6, -12.5), (-43.3, 32.7, -26.0),
                     (43.3, 32.7, -26.0), (-28.9, -28.9, -24.1), (28.9, -28.9, -24.1)], dtype=np.float64)
IMG_NAMES = ["sharp_log", "sat", "val", "glare", "hf_ratio", "spec_peak_log", "b", "g", "r"]
_yy, _xx = np.mgrid[-112:112, -112:112] / 224.0
RADIUS = np.sqrt(_yy ** 2 + _xx ** 2)


def parse(name):
    m = re.match(r"^(p\d+)_(" + "|".join(TYPES) + r")", name, re.I)
    return (m.group(1).lower(), m.group(2).lower()) if m else None


def ear(pts, idx):
    p = [pts[i] for i in idx]
    d = lambda a, b: float(np.linalg.norm(a - b))
    return (d(p[1], p[5]) + d(p[2], p[4])) / (2.0 * d(p[0], p[3]) + 1e-9)


def head_pose(pts, w, h):
    img = np.array([pts[i] for i in POSE_IDX], dtype=np.float64)
    cam = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(MODEL_3D, img, cam, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return np.nan, np.nan, np.nan
    angles = cv2.RQDecomp3x3(cv2.Rodrigues(rvec)[0])[0]
    wrap = lambda a: float((a + 90.0) % 180.0 - 90.0)
    return wrap(angles[1]), wrap(angles[0]), wrap(angles[2])


def crop_face(frame, pts, margin=0.25, size=224):
    h, w = frame.shape[:2]
    x0, y0 = pts.min(axis=0); x1, y1 = pts.max(axis=0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0) * (1 + margin)
    xa, ya = int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0))
    xb, yb = int(min(cx + side / 2, w)), int(min(cy + side / 2, h))
    if xb - xa < 10 or yb - ya < 10:
        return None
    return cv2.resize(frame[ya:yb, xa:xb], (size, size))


def img_feats(crop):
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    F = np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean())))
    band = F[(RADIUS > 0.15) & (RADIUS < 0.45)]
    b, gg, r = crop.reshape(-1, 3).mean(0) / 255.0
    return [np.log1p(cv2.Laplacian(g, cv2.CV_32F).var()), hsv[..., 1].mean() / 255, hsv[..., 2].mean() / 255,
            float((hsv[..., 2] > 245).mean()), float(F[RADIUS > 0.25].sum() / (F.sum() + 1e-9)),
            float(np.log(band.max() / (np.median(band) + 1e-9))), b, gg, r]


def extract(path, target_fps, max_w=1280):
    """Per-frame features of one video."""
    import mediapipe as mp
    lm = mp.tasks.vision.FaceLandmarker.create_from_options(mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_faces=2, output_face_blendshapes=True))
    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"    video: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}, "
          f"{src_fps:.0f} frames/s", flush=True)
    step = max(int(round(src_fps / target_fps)), 1)
    out = {k: [] for k in ["t", "face", "two", "pts", "blend", "ear", "pose", "img"]}
    names, i = None, 0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    t_start = time.time()
    while True:
        if i % step != 0:            # skipped frames: grab only, no image copy (faster)
            if not cap.grab():
                break
            i += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        if i % int(round(src_fps * 10)) == 0 and i:   # progress every 10 s of video
            print(f"    {i / src_fps:.0f} s of {n_total / src_fps:.0f} s ({time.time() - t_start:.0f} s elapsed)",
                  flush=True)
        if True:
            if frame.shape[1] > max_w:
                s = max_w / frame.shape[1]
                frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            h, w = frame.shape[:2]
            res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                               data=np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))),
                                      int(i / src_fps * 1000))
            face, two, pts, bl, e, pose, im = 0, 0, np.full((478, 2), np.nan), np.full(52, np.nan), np.nan, \
                [np.nan] * 3, [np.nan] * len(IMG_NAMES)
            if res.face_landmarks:
                cand = [np.array([(p.x * w, p.y * h) for p in f]) for f in res.face_landmarks]
                areas = [np.ptp(c[:, 0]) * np.ptp(c[:, 1]) for c in cand]
                order = np.argsort(areas)[::-1]
                k = order[0]
                two = int(len(cand) > 1 and areas[order[1]] > 0.3 * areas[k])
                p = cand[k][:478]
                crop = crop_face(frame, p)
                if crop is not None:
                    face, pts = 1, p
                    if res.face_blendshapes:
                        cats = res.face_blendshapes[k]
                        if names is None:
                            names = [c.category_name for c in cats]
                        sc = {c.category_name: c.score for c in cats}
                        bl = np.array([sc.get(n, np.nan) for n in names[:52]])
                    e = (ear(p, R_EYE) + ear(p, L_EYE)) / 2
                    pose = list(head_pose(p, w, h))
                    im = img_feats(crop)
            for key, v in zip(["t", "face", "two", "pts", "blend", "ear", "pose", "img"],
                              [i / src_fps, face, two, pts, bl, e, pose, im]):
                out[key].append(v)
        i += 1
    cap.release()
    lm.close()
    d = {k: np.array(v, dtype=np.float32 if k not in ("face", "two") else np.int8) for k, v in out.items()}
    d["blend_names"] = np.array(names or [])
    d["fps"] = np.float32(src_fps / step)
    return d


def affine_residual(P_from, P_to):
    """RMS residual after affine alignment of the points (frame -> target)."""
    X = np.c_[P_from, np.ones(len(P_from))]
    A, *_ = np.linalg.lstsq(X, P_to, rcond=None)
    return float(np.sqrt(((X @ A - P_to) ** 2).sum(1).mean()))


def window_features(d, a_, b_):
    """Features of the frame window [a_, b_). Returns (dict M, dict I) or None."""
    valid = (d["face"][a_:b_] == 1) & (d["two"][a_:b_] == 0)
    if valid.mean() < 0.5:
        return None
    sl = np.arange(a_, b_)[valid]
    P = d["pts"][sl].astype(np.float64)
    io = np.linalg.norm(P[:, 33] - P[:, 263], axis=1)
    s = float(np.median(io)) + 1e-9
    M = P.mean(0)
    res_mean = np.array([affine_residual(P[j], M) for j in range(len(P))]) / s
    res_step = np.array([affine_residual(P[j], P[j - 1]) for j in range(1, len(P))]) / s if len(P) > 1 \
        else np.zeros(1)
    c = P.mean(1)
    names = [str(x) for x in d["blend_names"]]
    B = d["blend"][sl].astype(np.float64)
    bstd = np.nanstd(B, axis=0) if np.isfinite(B).any() else np.zeros(52)
    ib = [names.index(n) for n in ["eyeBlinkLeft", "eyeBlinkRight"] if n in names]
    blink = np.nanmean(B[:, ib], axis=1) if ib else np.zeros(len(sl))
    blink = np.nan_to_num(blink)
    bl_cnt = int(((blink[1:] >= 0.5) & (blink[:-1] < 0.5)).sum())
    E = d["ear"][sl].astype(np.float64)
    med_e = np.nanmedian(E) + 1e-9
    low = E < 0.8 * med_e
    ear_cnt = int((low[1:] & ~low[:-1]).sum())
    pose = d["pose"][sl].astype(np.float64)
    jaw = bstd[names.index("jawOpen")] if "jawOpen" in names else 0.0
    m = {"blink_count": bl_cnt, "blink_max": float(blink.max()), "blink_std": float(blink.std()),
         "ear_dip_count": ear_cnt, "ear_std_rel": float(np.nanstd(E) / med_e), "ear_min_rel": float(np.nanmin(E) / med_e),
         "nonrigid_mean": float(res_mean.mean()), "nonrigid_max": float(res_mean.max()),
         "nonrigid_std": float(res_mean.std()), "nonrigid_step_mean": float(res_step.mean()),
         "nonrigid_step_max": float(res_step.max()),
         "center_std": float(np.linalg.norm(c.std(0)) / s), "scale_std": float(io.std() / s),
         "yaw_std": float(np.nanstd(pose[:, 0])), "pitch_std": float(np.nanstd(pose[:, 1])),
         "roll_std": float(np.nanstd(pose[:, 2])),
         "blend_std_mean": float(np.nanmean(bstd)), "blend_std_max": float(np.nanmax(bstd)), "jaw_std": float(jaw),
         "face_frac": float(valid.mean())}
    I = d["img"][sl].astype(np.float64)
    i = {f"{n}_mean": float(np.nanmean(I[:, k])) for k, n in enumerate(IMG_NAMES)}
    i.update({f"{n}_std": float(np.nanstd(I[:, k])) for k, n in enumerate(IMG_NAMES)})
    return m, i


def build_windows(cache, win_s, step_s, trim_s=3.0, max_s=60.0):
    rows = []
    for name, (person, typ, d) in cache.items():
        fps = float(d["fps"]); n = len(d["t"])
        W, S = int(round(win_s * fps)), max(int(round(step_s * fps)), 1)
        cut = int(round(trim_s * fps))          # drop the start and end of the recording
        dur = n / fps
        end = min(n - cut, cut + int(round(max_s * fps)))   # equal length: at most max_s after trimming
        for a_ in range(cut, end - W + 1, S):
            r = window_features(d, a_, a_ + W)
            if r is None:
                continue
            m, i = r
            rows.append({"video": name, "person": person, "type": typ, "live": int(typ.startswith("live")),
                         "t0": a_ / fps, "t1": (a_ + W) / fps, "dur": dur, **m, **i})
    return pd.DataFrame(rows)


def metrics(df, pred_live):
    out = {}
    live = df["live"].values == 1
    for t in ["live_normal", "live_still"]:
        m = df["type"].values == t
        out[f"BPCER {t} %"] = 100 * float((pred_live[m] == 0).mean()) if m.any() else np.nan
    out["BPCER %"] = 100 * float((pred_live[live] == 0).mean()) if live.any() else np.nan
    ap = []
    for t in ATTACKS:
        m = df["type"].values == t
        v = 100 * float((pred_live[m] == 1).mean()) if m.any() else np.nan
        out[f"APCER {t} %"] = v
        if m.any():
            ap.append(v)
    out["ACER %"] = (max(ap) + out["BPCER %"]) / 2 if ap and live.any() else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=r"data_raw\spoof")
    ap.add_argument("--cache", default=r"results\spoof")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--win", type=float, default=5.0)
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--trim", type=float, default=3.0, help="discard this many seconds at the start and end of each recording")
    ap.add_argument("--max-sec", type=float, default=60.0, help="use at most this many seconds of each recording after trimming")
    a = ap.parse_args()
    os.makedirs(a.cache, exist_ok=True)
    t0 = time.time()

    files = sorted(f for f in os.listdir(a.input) if os.path.splitext(f)[1].lower() in VIDEO_EXT)
    cache = {}
    for f in files:
        pt = parse(f)
        if pt is None:
            print("skipping (name does not match p<number>_<type>):", f)
            continue
        name = os.path.splitext(f)[0]
        fn = os.path.join(a.cache, f"frames_{name}.npz")
        if not os.path.exists(fn):
            if not os.path.exists(MODEL_PATH):
                sys.exit(f"{MODEL_PATH} not found")
            print(f"extracting {f} ...", flush=True)
            np.savez_compressed(fn, **extract(os.path.join(a.input, f), a.fps))
        d = dict(np.load(fn))
        c_ = int(round(a.trim * float(d["fps"])))
        core = slice(c_, max(len(d["face"]) - c_, c_))
        ff = float((d["face"][core] == 1).mean()) if len(d["face"][core]) else 0.0
        tw = float((d["two"] == 1).mean()) if len(d["two"]) else 0.0
        print(f"  {name}: {len(d['t'])} frames ({float(d['fps']):.1f}/s), face {100 * ff:.0f}%, "
              f"two faces {100 * tw:.1f}%")
        if ff < 0.5:
            print(f"  -> EXCLUDED (face found in less than 50% of frames)")
            continue
        cache[name] = (pt[0], pt[1], d)

    W = build_windows(cache, a.win, a.step, a.trim, a.max_sec)
    if W.empty:
        sys.exit("No windows for training")
    W.to_csv(os.path.join(a.cache, "liveness_windows.csv"), index=False)
    feat_M = ["blink_count", "blink_max", "blink_std", "ear_dip_count", "ear_std_rel", "ear_min_rel",
              "nonrigid_mean", "nonrigid_max", "nonrigid_std", "nonrigid_step_mean", "nonrigid_step_max",
              "center_std", "scale_std", "yaw_std", "pitch_std", "roll_std",
              "blend_std_mean", "blend_std_max", "jaw_std", "face_frac"]
    feat_I = [f"{n}_{s}" for n in IMG_NAMES for s in ("mean", "std")]
    sets = {"M (motion)": feat_M, "I (image)": feat_I, "MI (both)": feat_M + feat_I}

    print("\nWindows by type and person:")
    print(W.pivot_table(index="person", columns="type", values="t0", aggfunc="count", fill_value=0).to_string())

    both = W.groupby("person")["live"].agg(lambda s: s.nunique() == 2)
    persons_ok = sorted(both[both].index)
    lopo = len(persons_ok) >= 3
    if lopo:
        print(f"\nEvaluation BY PERSON ({len(persons_ok)} people with live and attack recordings).")
        splits = [(p, W.index[W["person"] != p], W.index[W["person"] == p]) for p in persons_ok]
    else:
        print("\nWARNING: fewer than 3 people with live and attack recordings -> PRELIMINARY evaluation: time split "
              "inside each recording (first half train, second half test). Same people and recordings, "
              "so the result is optimistic; more people are needed.")
        mid = W["dur"] / 2
        tr, te = W.index[W["t1"] <= mid], W.index[W["t0"] >= mid]
        splits = [("time", tr, te)]

    rows, pp = [], []
    for sname, feats in sets.items():
        pred = pd.Series(np.nan, index=W.index)
        for p, tr, te in splits:
            if W.loc[tr, "live"].nunique() < 2:
                continue
            clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                                         random_state=0, n_jobs=a.jobs)
            clf.fit(W.loc[tr, feats].fillna(0).values, W.loc[tr, "live"].values)
            pred[te] = clf.predict(W.loc[te, feats].fillna(0).values)
            if lopo:
                pp.append({"set": sname, "person": p, **metrics(W.loc[te], pred[te].values.astype(int))})
        ev = pred.notna()
        r = metrics(W[ev], pred[ev].values.astype(int))
        # per-recording decision: majority of windows
        vid = W[ev].assign(pred=pred[ev].values).groupby("video").agg(live=("live", "first"), pred=("pred", "mean"))
        r["recordings correct"] = f"{int(((vid['pred'] >= 0.5).astype(int) == vid['live']).sum())}/{len(vid)}"
        r["test windows"] = int(ev.sum())
        rows.append({"set": sname, **{k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()}})
        W.loc[ev, f"pred_{sname.split()[0]}"] = pred[ev]

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    pcols = [c for c in ["pred_M", "pred_I", "pred_MI"] if c in W]
    if pcols:
        pv = W.groupby("video").agg(type=("type", "first"), windows=("t0", "size"),
                                    **{f"{c[5:]}: live %": (c, "mean") for c in pcols})
        for c in pv.columns[2:]:
            pv[c] = (100 * pv[c]).round(0)
        print("\nPer recording: share of windows the detector called LIVE (should be ~100 for live, ~0 for attacks):")
        print(pv.to_string())
        pv.to_csv(os.path.join(a.cache, "liveness_per_video.csv"))
    print("\nResults (5 s windows; APCER - attack accepted as live, BPCER - live rejected as attack):")
    print(res.to_string(index=False))
    res.to_csv(os.path.join(a.cache, "liveness_results.csv"), index=False)
    if pp:
        pp = pd.DataFrame(pp).round(1)
        print("\nPer person:")
        print(pp.to_string(index=False))
        pp.to_csv(os.path.join(a.cache, "liveness_per_person.csv"), index=False)

    # feature importance (trained on all windows) - for description only, not for evaluation
    clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                                 random_state=0, n_jobs=a.jobs).fit(W[feat_M + feat_I].fillna(0).values, W["live"])
    imp = pd.Series(clf.feature_importances_, index=feat_M + feat_I).sort_values(ascending=False)
    print("\nMost important features (MI, trained on all windows):")
    print(imp.head(10).round(3).to_string())
    print("\nMean motion features by type:")
    print(W.groupby("type")[["blink_count", "nonrigid_mean", "nonrigid_step_mean", "center_std",
                             "blend_std_mean"]].mean().round(4).to_string())
    print(f"\nSaved to {a.cache}\\ | {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
