"""Step 10. Final EB model: train on ALL people and save to a file (for the demo and the attack tests).
EB = geometry (EAR, MAR, head angles) + 1 s window + 52 blendshapes + PCA32 of MobileNetV2 embeddings,
all after calibration on the first 10 s of the recording; random forest. Same settings as 09_gaze_check.py.
For drowsiness the talking frames are added as alert (EB+T, step 12) unless --no-talking is given.

This script gives NO quality estimate (it trains on everyone); the LOPO evaluation is in 09/12/13.

Run (from the project root):
  python code\\10_train_final.py --task drowsy
  python code\\10_train_final.py --task attention
Output: models\\eb_<task>.joblib
"""
import argparse
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

BASE = ["ear", "mar", "yaw", "pitch", "roll"]
TASKS = {"drowsy": ("drowsy_label", ["alert", "eyes_closed", "yawning"]),
         "attention": ("attention_label", ["looking_road", "not_looking_road"])}
NOFACE_RULE = {"drowsy": "alert", "attention": "not_looking_road"}
BS = [f"bs_{i}" for i in range(52)]
WINDOW, CALIB_SEC, FPS, N_PCA, PCA_FIT_MAX = 5, 10, 5, 32, 20000


def add_window_features(df, k):
    df = df.sort_values(["video_id", "frame_idx"]).copy()
    g = df.groupby("video_id")
    feats = list(BASE)
    for c in ["ear", "mar"]:
        for name in ["mean", "min", "max", "std"]:
            col = f"{c}_{name}{k}"
            df[col] = g[c].transform(lambda s, n=name: getattr(s.rolling(k, min_periods=1), n)())
            feats.append(col)
    for c in ["yaw", "pitch"]:
        for name in ["mean", "std"]:
            col = f"{c}_{name}{k}"
            df[col] = g[c].transform(lambda s, n=name: getattr(s.rolling(k, min_periods=1), n)())
            feats.append(col)
    wcols = [x for x in feats if x not in BASE]
    df[wcols] = df[wcols].fillna(0)
    return df, feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), required=True)
    ap.add_argument("--labels", default=r"data_processed\dmd\labels.csv")
    ap.add_argument("--landmarks", default=r"data_processed\dmd\landmarks")
    ap.add_argument("--emb", default=r"results\emb_all_v2.npy")
    ap.add_argument("--out", default="models")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--no-talking", action="store_true",
                    help="for drowsiness, do NOT add talking frames (plain EB)")
    a = ap.parse_args()
    t0 = time.time()
    label_col, classes = TASKS[a.task]
    os.makedirs(a.out, exist_ok=True)

    lab = pd.read_csv(a.labels, keep_default_na=False, na_values=[""], low_memory=False)
    for c in BASE:
        lab[c] = pd.to_numeric(lab[c], errors="coerce")
    lab["face_found"] = pd.to_numeric(lab["face_found"], errors="coerce").fillna(0)
    face = lab[lab["face_found"] == 1].sort_values(["video_id", "frame_idx"]).copy()
    face["eidx"] = np.arange(len(face))
    n_emb = len(np.load(a.emb, mmap_mode="r"))
    if n_emb != len(face):
        raise SystemExit(f"Embedding cache ({n_emb}) does not match the frames ({len(face)})")

    # blendshapes (+ their names, so the demo uses the same order)
    Bm = np.full((len(face), 52), np.nan, np.float32)
    pos = {k: i for i, k in enumerate(zip(face["video_id"], face["frame_idx"]))}
    blend_names = None
    for vid in face["video_id"].unique():
        fn = os.path.join(a.landmarks, vid + ".npz")
        if not os.path.exists(fn):
            continue
        d = np.load(fn)
        if blend_names is None and len(d["blend_names"]):
            blend_names = [str(x) for x in d["blend_names"]]
        rows = np.array([pos.get((vid, int(x)), -1) for x in d["frame_idx"]])
        ok = (rows >= 0) & (d["face"] == 1)
        Bm[rows[ok]] = d["blend"][ok][:, :52]
    face[BS] = Bm
    del Bm

    n_cal = int(round(CALIB_SEC * FPS))
    f = face
    f[BASE] = f.groupby("video_id")[BASE].ffill(limit=3)
    f = f.dropna(subset=BASE).sort_values(["video_id", "frame_idx"])
    f["is_cal"] = (f.groupby("video_id").cumcount() < n_cal).values
    f["bs_ok"] = f[BS].notna().all(axis=1).astype(float)
    ref = f[f["is_cal"]].groupby("video_id")[BASE].median()
    f[BASE] = f[BASE].values - ref.reindex(f["video_id"]).values
    ref_b = f[f["is_cal"]].groupby("video_id")[BS].median()
    f[BS] = (f[BS].values - ref_b.reindex(f["video_id"]).values)
    f[BS] = f[BS].fillna(0.0)
    f, feats_A = add_window_features(f, WINDOW)
    feats_B = BS + ["bs_ok"]

    lab_mask = f[label_col].isin(classes).values
    if a.task == "drowsy" and not a.no_talking:
        # EB+T (step 12): talking frames without a drowsiness label are added as alert
        tk = f["talking"].fillna("").astype(str).str.strip().str.lower()
        talk = (tk.str.contains("talk") & ~tk.str.contains("not")).values & ~lab_mask
        f.loc[talk, label_col] = "alert"
        lab_mask = lab_mask | talk
        print(f"Talking frames added as alert: {int(talk.sum())}", flush=True)
    tr = np.flatnonzero(lab_mask)
    vids = f["video_id"].values
    E_all = np.load(a.emb, mmap_mode="r")
    Ec = np.asarray(E_all[f["eidx"].values[tr]], dtype=np.float32)
    # embedding calibration: median of the first 10 s of the recording (all frames of the recording, as in 09)
    cal_rows = np.flatnonzero(f["is_cal"].values)
    for v in np.unique(vids[tr]):
        med = np.median(np.asarray(E_all[f["eidx"].values[cal_rows[vids[cal_rows] == v]]], np.float32), axis=0)
        Ec[vids[tr] == v] -= med
    del E_all

    sub = np.random.RandomState(0).permutation(len(tr))[:PCA_FIT_MAX]
    sc = StandardScaler().fit(Ec[sub])
    pca = PCA(N_PCA, random_state=0).fit(sc.transform(Ec[sub]))
    Z = np.vstack([pca.transform(sc.transform(Ec[j:j + 10000])) for j in range(0, len(Ec), 10000)])
    X = np.hstack([f[feats_A].values[tr], f[feats_B].values[tr], Z]).astype(np.float32)
    y = f[label_col].values[tr]
    n_people = len(set(f["person"].values[tr]))
    print(f"Task: {a.task} | training frames: {len(y)} | people: {n_people} | features: {X.shape[1]}",
          flush=True)
    print(pd.Series(y).value_counts().to_string(), flush=True)

    clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                                 random_state=0, n_jobs=a.jobs).fit(X, y)
    bundle = dict(task=a.task, classes=list(clf.classes_), noface_label=NOFACE_RULE[a.task],
                  feats_A=feats_A, feats_B=feats_B, blend_names=blend_names,
                  window=WINDOW, calib_sec=CALIB_SEC, fps=FPS,
                  scaler=sc, pca=pca, rf=clf,
                  note="Features: [feats_A, feats_B, PCA32]; all after subtracting the median of the first calib_sec "
                       "seconds of the recording (frames with a face). Embeddings: MobileNetV2 imagenet, avg pool, "
                       "224x224, on the same face crops that 02_extract_features.py saved.")
    out = os.path.join(a.out, f"eb_{a.task}.joblib")
    joblib.dump(bundle, out, compress=3)
    print(f"Saved: {out} ({os.path.getsize(out) / 1e6:.0f} MB) | {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
