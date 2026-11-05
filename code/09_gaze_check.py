"""Step 9. Random-forest models A, AB, E, EB (LOPO): do gaze features (blendshapes) help, or memory?
Plus temporal smoothing of probabilities.

Variants (all random forests without memory, apart from the 1 s feature window):
  A  - geometry + 1 s window
  AB - A + 52 blendshapes from 02b
  E  - hybrid: PCA32 of the embeddings + A
  EB - E + 52 blendshapes
  *_s - the same with smoothing: probabilities averaged over the last 5 frames (1 s) of the recording
Protocol: calibration on the first 10 s of each recording without labels; PCA and models are fitted
on training people only; frames without a face get a fixed answer
(attention -> not_looking_road, drowsy -> alert).

Needs: labels.csv, the landmarks folder (from 02b), results\\emb_all_v2.npy (from m0b_honest_colab.ipynb).

Run (from the project root):
  python code\\09_gaze_check.py --task attention
  python code\\09_gaze_check.py --task drowsy
  (A and AB only, faster and without embeddings:  --configs A,AB)
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
ALL_CONFIGS = {"A": (False, False), "AB": (False, True), "E": (True, False), "EB": (True, True)}

WINDOW, CALIB_SEC, FPS = 5, 10, 5   # fixed in advance, not tuned on test results
SMOOTH, N_PCA, PCA_FIT_MAX = 5, 32, 20000


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


def smooth(P, vid):
    """Moving average of probabilities over the last SMOOTH frames of the same recording (past only)."""
    return (pd.DataFrame(P).groupby(vid).rolling(SMOOTH, min_periods=1).mean()
            .reset_index(level=0, drop=True).sort_index().values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), required=True)
    ap.add_argument("--labels", default=r"data_processed\dmd\labels.csv")
    ap.add_argument("--landmarks", default=r"data_processed\dmd\landmarks")
    ap.add_argument("--emb", default=r"results\emb_all_v2.npy")
    ap.add_argument("--out", default="results")
    ap.add_argument("--configs", default="A,AB,E,EB")
    ap.add_argument("--jobs", type=int, default=-1, help="cores for RandomForest (-1 = all)")
    a = ap.parse_args()

    configs = {c: ALL_CONFIGS[c] for c in a.configs.split(",")}
    need_emb = any(e for e, _ in configs.values())
    label_col, classes = TASKS[a.task]
    os.makedirs(a.out, exist_ok=True)

    lab = pd.read_csv(a.labels, keep_default_na=False, na_values=[""], low_memory=False)
    for c in BASE:
        lab[c] = pd.to_numeric(lab[c], errors="coerce")
    lab["face_found"] = pd.to_numeric(lab["face_found"], errors="coerce").fillna(0)

    # frames with a face in the same order as the cached embeddings (as in m0b_honest)
    face = lab[lab["face_found"] == 1].sort_values(["video_id", "frame_idx"]).copy()
    face["eidx"] = np.arange(len(face))
    if need_emb:
        if not os.path.exists(a.emb):
            raise SystemExit(f"Embedding file not found: {a.emb} (create it with m0b_honest_colab.ipynb) "
                             f"or run with --configs A,AB")
        n_emb = len(np.load(a.emb, mmap_mode="r"))
        if n_emb != len(face):
            raise SystemExit(f"Embedding cache ({n_emb}) does not match the frames ({len(face)})")

    # blendshapes from 02b
    Bm = np.full((len(face), 52), np.nan, np.float32)
    pos = {k: i for i, k in enumerate(zip(face["video_id"], face["frame_idx"]))}
    missing = 0
    for vid in face["video_id"].unique():
        fn = os.path.join(a.landmarks, vid + ".npz")
        if not os.path.exists(fn):
            missing += 1
            continue
        d = np.load(fn)
        rows = np.array([pos.get((vid, int(x)), -1) for x in d["frame_idx"]])
        ok = (rows >= 0) & (d["face"] == 1)
        Bm[rows[ok]] = d["blend"][ok][:, :52]
    print("Recordings without a landmark file:", missing, flush=True)
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
    f[BS] = f[BS].values - ref_b.reindex(f["video_id"]).values
    f[BS] = f[BS].fillna(0.0)
    f, feats_A = add_window_features(f, WINDOW)
    feats_B = BS + ["bs_ok"]
    vids = f["video_id"].values

    Ec = None
    if need_emb:
        E_all = np.load(a.emb, mmap_mode="r")
        Ec = np.asarray(E_all[f["eidx"].values], dtype=np.float32)
        del E_all
        cal = f["is_cal"].values
        for v in np.unique(vids):
            m = vids == v
            Ec[m] -= np.median(Ec[m & cal], axis=0)
        gc.collect()

    lab_mask = f[label_col].isin(classes).values
    labeled = lab[lab[label_col].isin(classes)]
    noface = labeled[~labeled.index.isin(f.index[lab_mask])]
    person_f = f["person"].values
    y_f = f[label_col].values
    persons = sorted(set(person_f[lab_mask]) | set(noface["person"]))
    print(f"Task: {a.task} | variants: {list(configs)} | people: {len(persons)} | "
          f"labelled with a face: {int(lab_mask.sum())} | no face: {len(noface)}", flush=True)

    XA = f[feats_A].values.astype(np.float32)
    XB = f[feats_B].values.astype(np.float32)
    # only recordings of this task are evaluated: those with labelled frames
    task_vid = set(f.loc[lab_mask, "video_id"])
    in_task = np.isin(vids, list(task_vid))
    proba = {c: np.full((len(f), len(classes)), np.nan, np.float32) for c in configs}
    proba_s = {c: np.full((len(f), len(classes)), np.nan, np.float32) for c in configs}

    t0 = time.time()
    for i, p in enumerate(persons, 1):
        tr = np.flatnonzero(lab_mask & (person_f != p))
        te = np.flatnonzero((person_f == p) & in_task)
        if not len(te):
            continue
        Z = None
        if need_emb:
            sub = tr[np.random.RandomState(0).permutation(len(tr))[:PCA_FIT_MAX]]
            sc = StandardScaler().fit(Ec[sub])
            pca = PCA(N_PCA, random_state=0).fit(sc.transform(Ec[sub]))
            tf = lambda idx: np.vstack([pca.transform(sc.transform(Ec[idx[j:j + 10000]]))
                                        for j in range(0, len(idx), 10000)]).astype(np.float32)
            Z = {"tr": tf(tr), "te": tf(te)}
        for cfg, (use_e, use_b) in configs.items():
            ptr, pte = [XA[tr]], [XA[te]]
            if use_b:
                ptr.append(XB[tr]); pte.append(XB[te])
            if use_e:
                ptr.append(Z["tr"]); pte.append(Z["te"])
            clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                                         random_state=0, n_jobs=a.jobs)
            clf.fit(np.hstack(ptr), y_f[tr])
            P = np.zeros((len(te), len(classes)), np.float32)
            P[:, [classes.index(c) for c in clf.classes_]] = clf.predict_proba(np.hstack(pte))
            proba[cfg][te] = P
            proba_s[cfg][te] = smooth(P, vids[te])
        print(f"[{i}/{len(persons)}] {p}: done ({(time.time() - t0) / 60:.1f} min)", flush=True)

    # evaluation
    y_all = np.concatenate([y_f[lab_mask], noface[label_col].values])
    p_all = np.concatenate([person_f[lab_mask], noface["person"].values])
    yawn_all = (np.concatenate([f.loc[lab_mask, "yawning"].astype(str).values,
                                noface["yawning"].astype(str).values]) if "yawning" in f.columns else None)
    rule = np.array([NOFACE_RULE[a.task]] * len(noface), dtype=object)
    preds, rows, per_person = {}, [], {}
    for cfg in configs:
        for suffix, src in [("", proba), ("_s", proba_s)]:
            name = cfg + suffix
            pr = np.concatenate([np.array(classes, dtype=object)[src[cfg][lab_mask].argmax(1)], rule])
            preds[name] = pr
            row, f1s = {"config": name}, []
            for p in persons:
                m = p_all == p
                v = f1_score(y_all[m], pr[m], labels=sorted(set(y_all[m])), average="macro", zero_division=0)
                row[p] = round(v, 3); f1s.append(v)
            per_person[name] = np.array(f1s)
            row["mean"] = round(float(np.mean(f1s)), 3)
            row["std_people"] = round(float(np.std(f1s, ddof=1)), 3)
            if a.task == "drowsy" and yawn_all is not None:
                is_y = y_all == "yawning"
                for t, short in [("Yawning with hand", "yawn_with_hand_%"),
                                 ("Yawning without hand", "yawn_no_hand_%")]:
                    mt = is_y & (yawn_all == t)
                    row[short] = round(100 * float((pr[mt] == "yawning").mean()), 1) if mt.any() else np.nan
                row["false_yawn_frames"] = int(((pr == "yawning") & ~is_y).sum())
            rows.append(row)
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print("\n" + table.to_string(index=False))

    pairs = [("AB", "A"), ("EB", "E"), ("A_s", "A"), ("AB_s", "AB"), ("E_s", "E"), ("EB_s", "EB"), ("EB_s", "E")]
    sig = []
    for x, y in pairs:
        if x in per_person and y in per_person:
            d = per_person[x] - per_person[y]
            p = wilcoxon(per_person[x], per_person[y]).pvalue if np.any(d != 0) else np.nan
            sig.append({"comparison": f"{x} vs {y}", "diff": round(float(d.mean()), 4),
                        "better": f"{int((d > 0).sum())}/{len(d)}", "p": round(float(p), 4)})
    sig = pd.DataFrame(sig)
    print("\nSignificance (Wilcoxon over people; several comparisons, p~0.05 = trend):")
    print(sig.to_string(index=False))

    table.to_csv(os.path.join(a.out, f"summary_gazecheck_{a.task}.csv"), index=False)
    sig.to_csv(os.path.join(a.out, f"wilcoxon_gazecheck_{a.task}.csv"), index=False)
    out = pd.concat([f.loc[lab_mask, ["video_id", "frame_idx", "person", label_col]],
                     noface[["video_id", "frame_idx", "person", label_col]]], ignore_index=True)
    out["face"] = np.r_[np.ones(int(lab_mask.sum()), int), np.zeros(len(noface), int)]
    for name, pr in preds.items():
        out[f"pred_{name}"] = pr
    for cfg in configs:
        P = np.vstack([proba[cfg][lab_mask], np.full((len(noface), len(classes)), np.nan, np.float32)])
        for j, c in enumerate(classes):
            out[f"prob_{cfg}_{c}"] = P[:, j]
    out.to_csv(os.path.join(a.out, f"preds_gazecheck_{a.task}.csv"), index=False)
    print(f"\nSaved to {a.out}\\ (summary_, wilcoxon_, preds_gazecheck_{a.task}.csv) | "
          f"total {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
