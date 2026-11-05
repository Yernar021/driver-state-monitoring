"""Step 22. Ablations: what do calibration (10 s) and the window features (1 s) contribute?
Model EB (geometry + 1 s window + 52 blendshapes + PCA32 of the embeddings, random forest), LOPO, as in 09.
Variants (fixed in advance):
  EB             - full model (as in the paper);
  EB_nocal       - no calibration: features and embeddings without subtracting the median of the first 10 s;
  EB_nowin       - no 1 s window features (current frame only);
  EB_nocal_nowin - neither.
Frames without a face get the same fixed answer. Comparison with EB: Wilcoxon over people,
difference, number of people improved.

Run (from the project root), ~15-25 min for both tasks:
  python code\\22_ablation.py --task drowsy
  python code\\22_ablation.py --task attention
Output: results\\ablation_<task>.csv, results\\ablation_per_person_<task>.csv
"""
import argparse
import gc
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

BASE = ["ear", "mar", "yaw", "pitch", "roll"]
TASKS = {"drowsy": ("drowsy_label", ["alert", "eyes_closed", "yawning"]),
         "attention": ("attention_label", ["looking_road", "not_looking_road"])}
NOFACE_RULE = {"drowsy": "alert", "attention": "not_looking_road"}
BS = [f"bs_{i}" for i in range(52)]
WINDOW, CALIB_SEC, FPS, N_PCA, PCA_FIT_MAX = 5, 10, 5, 32, 20000
VARIANTS = {"EB": (True, True), "EB_nocal": (False, True), "EB_nowin": (True, False), "EB_nocal_nowin": (False, False)}


def add_window_features(df, k):
    df = df.sort_values(["video_id", "frame_idx"]).copy()
    g = df.groupby("video_id")
    feats = []
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
    df[feats] = df[feats].fillna(0)
    return df, feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), required=True)
    ap.add_argument("--labels", default=r"data_processed\dmd\labels.csv")
    ap.add_argument("--landmarks", default=r"data_processed\dmd\landmarks")
    ap.add_argument("--emb", default=r"results\emb_all_v2.npy")
    ap.add_argument("--out", default="results")
    ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args()
    t0 = time.time()
    label_col, classes = TASKS[a.task]

    lab = pd.read_csv(a.labels, keep_default_na=False, na_values=[""], low_memory=False)
    for c in BASE:
        lab[c] = pd.to_numeric(lab[c], errors="coerce")
    lab["face_found"] = pd.to_numeric(lab["face_found"], errors="coerce").fillna(0)
    face = lab[lab["face_found"] == 1].sort_values(["video_id", "frame_idx"]).copy()
    face["eidx"] = np.arange(len(face))
    if len(np.load(a.emb, mmap_mode="r")) != len(face):
        raise SystemExit("Embedding cache does not match the frames")

    Bm = np.full((len(face), 52), np.nan, np.float32)
    pos = {k: i for i, k in enumerate(zip(face["video_id"], face["frame_idx"]))}
    for vid in face["video_id"].unique():
        fn = os.path.join(a.landmarks, vid + ".npz")
        if not os.path.exists(fn):
            continue
        d = np.load(fn)
        rows = np.array([pos.get((vid, int(x)), -1) for x in d["frame_idx"]])
        ok = (rows >= 0) & (d["face"] == 1)
        Bm[rows[ok]] = d["blend"][ok][:, :52]
    face[BS] = Bm
    del Bm

    n_cal = int(round(CALIB_SEC * FPS))
    f = face
    f[BASE] = f.groupby("video_id")[BASE].ffill(limit=3)
    f = f.dropna(subset=BASE).sort_values(["video_id", "frame_idx"]).copy()
    f["is_cal"] = (f.groupby("video_id").cumcount() < n_cal).values
    f["bs_ok"] = f[BS].notna().all(axis=1).astype(float)
    vids = f["video_id"].values
    cal = f["is_cal"].values

    # two versions of the features: calibrated and raw
    raw = f.copy()
    raw[BS] = raw[BS].fillna(0.0)
    calf = f.copy()
    ref = calf[calf["is_cal"]].groupby("video_id")[BASE].median()
    calf[BASE] = calf[BASE].values - ref.reindex(calf["video_id"]).values
    ref_b = calf[calf["is_cal"]].groupby("video_id")[BS].median()
    calf[BS] = calf[BS].values - ref_b.reindex(calf["video_id"]).values
    calf[BS] = calf[BS].fillna(0.0)
    calf, wfeats = add_window_features(calf, WINDOW)
    raw, _ = add_window_features(raw, WINDOW)
    assert (calf.index == raw.index).all()
    feats_B = BS + ["bs_ok"]
    X = {True: calf, False: raw}

    E_all = np.load(a.emb, mmap_mode="r")
    E_raw = np.asarray(E_all[f["eidx"].values], dtype=np.float32)
    del E_all
    E_cal = E_raw.copy()
    for v in np.unique(vids):
        m = vids == v
        E_cal[m] -= np.median(E_cal[m & cal], axis=0)
    gc.collect()
    E = {True: E_cal, False: E_raw}

    lab_mask = f[label_col].isin(classes).values
    labeled = lab[lab[label_col].isin(classes)]
    noface = labeled[~labeled.index.isin(f.index[lab_mask])]
    person_f = f["person"].values
    y_f = f[label_col].values
    persons = sorted(set(person_f[lab_mask]) | set(noface["person"]))
    preds = {v: np.empty(len(f), dtype=object) for v in VARIANTS}

    for i, p in enumerate(persons, 1):
        tr = np.flatnonzero(lab_mask & (person_f != p))
        te = np.flatnonzero(lab_mask & (person_f == p))
        if not len(te):
            continue
        sub = tr[np.random.RandomState(0).permutation(len(tr))[:PCA_FIT_MAX]]
        Z = {}
        for c_ in (True, False):
            sc = StandardScaler().fit(E[c_][sub])
            pca = PCA(N_PCA, random_state=0).fit(sc.transform(E[c_][sub]))
            tf = lambda idx: np.vstack([pca.transform(sc.transform(E[c_][idx[j:j + 10000]]))
                                        for j in range(0, len(idx), 10000)])
            Z[c_] = (tf(tr), tf(te))
        for v, (use_cal, use_win) in VARIANTS.items():
            cols = BASE + (wfeats if use_win else []) + feats_B
            M = X[use_cal][cols].values.astype(np.float32)
            clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                                         random_state=0, n_jobs=a.jobs)
            clf.fit(np.hstack([M[tr], Z[use_cal][0]]), y_f[tr])
            preds[v][te] = clf.predict(np.hstack([M[te], Z[use_cal][1]]))
        print(f"[{i}/{len(persons)}] {p}: done ({(time.time() - t0) / 60:.1f} min)", flush=True)

    rule = NOFACE_RULE[a.task]
    pp = []
    for p in persons:
        m = lab_mask & (person_f == p)
        nf = noface[noface["person"] == p]
        yt = np.r_[y_f[m], nf[label_col].values]
        row = {"person": p}
        for v in VARIANTS:
            yp = np.r_[preds[v][m], np.array([rule] * len(nf), dtype=object)]
            row[v] = f1_score(yt, yp, labels=sorted(set(yt)), average="macro", zero_division=0)
        pp.append(row)
    pp = pd.DataFrame(pp).set_index("person")
    rows = []
    for v in VARIANTS:
        d = pp[v] - pp["EB"]
        pv = wilcoxon(pp[v], pp["EB"]).pvalue if v != "EB" and (d != 0).any() else np.nan
        rows.append({"variant": v, "mean": round(pp[v].mean(), 3), "sd": round(pp[v].std(ddof=1), 3),
                     "diff_vs_EB": round(d.mean(), 4), "better_than_EB": f"{int((d > 0).sum())}/{len(d)}",
                     "p_wilcoxon": round(pv, 4) if pv == pv else np.nan})
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\nAblations (" + a.task + "):")
    print(res.to_string(index=False))
    res.to_csv(os.path.join(a.out, f"ablation_{a.task}.csv"), index=False)
    pp.round(4).to_csv(os.path.join(a.out, f"ablation_per_person_{a.task}.csv"))
    print(f"\nSaved: results\\ablation_{a.task}.csv | {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
