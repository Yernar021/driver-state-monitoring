"""Step 12. Is talking read as yawning? Check and fix (drowsiness, model EB, LOPO).

Variants:
  EB   - as in 09/10: trained on labelled drowsiness frames only;
  EB+T - plus talking frames (column talking) from the distraction sessions, labelled alert
         (assumption: while talking the driver is neither yawning nor asleep).
For each test person:
  - macro-F1 on the usual drowsiness labels (frames without a face -> alert, as before);
  - talk_yawn_% - share of talking frames (with a face) that the model calls a yawn;
    the test person's own talking frames never enter training.
Same settings as 09 (1 s window, 10 s calibration, PCA32, random forest with 300 trees).

Run (from the project root):
  python code\\12_talking_check.py
"""
import argparse
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
CLASSES = ["alert", "eyes_closed", "yawning"]
LABEL = "drowsy_label"
BS = [f"bs_{i}" for i in range(52)]
WINDOW, CALIB_SEC, FPS, N_PCA, PCA_FIT_MAX = 5, 10, 5, 32, 20000
FALSY = {"", "0", "0.0", "false", "no", "none", "nan", "not talking", "no talking"}


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
    ap.add_argument("--labels", default=r"data_processed\dmd\labels.csv")
    ap.add_argument("--landmarks", default=r"data_processed\dmd\landmarks")
    ap.add_argument("--emb", default=r"results\emb_all_v2.npy")
    ap.add_argument("--out", default="results")
    ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args()
    t0 = time.time()

    lab = pd.read_csv(a.labels, keep_default_na=False, na_values=[""], low_memory=False)
    for c in BASE:
        lab[c] = pd.to_numeric(lab[c], errors="coerce")
    lab["face_found"] = pd.to_numeric(lab["face_found"], errors="coerce").fillna(0)
    # turn missing values into empty strings first (in recent pandas NaN does not become "nan")
    tk = lab["talking"].fillna("").astype(str).str.strip().str.lower()
    print("Values of the talking column (all frames):")
    print(tk.value_counts().head(10).to_string(), "\n")
    lab["is_talk"] = (tk.str.contains("talk") & ~tk.str.contains("not") & ~tk.isin(FALSY)).values

    face = lab[lab["face_found"] == 1].sort_values(["video_id", "frame_idx"]).copy()
    face["eidx"] = np.arange(len(face))
    n_emb = len(np.load(a.emb, mmap_mode="r"))
    if n_emb != len(face):
        raise SystemExit(f"Embedding cache ({n_emb}) does not match the frames ({len(face)})")

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
    f = f.dropna(subset=BASE).sort_values(["video_id", "frame_idx"])
    f["is_cal"] = (f.groupby("video_id").cumcount() < n_cal).values
    f["bs_ok"] = f[BS].notna().all(axis=1).astype(float)
    ref = f[f["is_cal"]].groupby("video_id")[BASE].median()
    f[BASE] = f[BASE].values - ref.reindex(f["video_id"]).values
    ref_b = f[f["is_cal"]].groupby("video_id")[BS].median()
    f[BS] = f[BS].values - ref_b.reindex(f["video_id"]).values
    f[BS] = f[BS].fillna(0.0)
    f, feats_A = add_window_features(f, WINDOW)
    feats = feats_A + BS + ["bs_ok"]

    vids = f["video_id"].values
    lab_mask = f[LABEL].isin(CLASSES).values
    talk_mask = f["is_talk"].values & ~lab_mask
    use = lab_mask | talk_mask
    idx_use = np.flatnonzero(use)
    E_all = np.load(a.emb, mmap_mode="r")
    Ec = np.asarray(E_all[f["eidx"].values[idx_use]], dtype=np.float32)
    cal_rows = np.flatnonzero(f["is_cal"].values)
    for v in np.unique(vids[idx_use]):
        cr = cal_rows[vids[cal_rows] == v]
        Ec[vids[idx_use] == v] -= np.median(np.asarray(E_all[f["eidx"].values[cr]], np.float32), axis=0)
    del E_all
    row_of = np.full(len(f), -1); row_of[idx_use] = np.arange(len(idx_use))

    X = f[feats].values.astype(np.float32)
    y = f[LABEL].values.astype(object)
    y_t = y.copy(); y_t[talk_mask] = "alert"
    person = f["person"].values
    print(f"Drowsiness frames with a face: {int(lab_mask.sum())} | talking frames with a face (outside drowsiness labels): "
          f"{int(talk_mask.sum())}")
    print("Talking frames per person:", pd.Series(person[talk_mask]).value_counts().sort_index().to_dict(), "\n")

    labeled = lab[lab[LABEL].isin(CLASSES)]
    noface = labeled[~labeled.index.isin(f.index[lab_mask])]
    persons = sorted(set(person[use]) | set(noface["person"]))

    def fit_predict(tr, te, labels):
        sub = tr[np.random.RandomState(0).permutation(len(tr))[:PCA_FIT_MAX]]
        sc = StandardScaler().fit(Ec[row_of[sub]])
        pca = PCA(N_PCA, random_state=0).fit(sc.transform(Ec[row_of[sub]]))
        z = lambda ii: pca.transform(sc.transform(Ec[row_of[ii]]))
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                                     random_state=0, n_jobs=a.jobs)
        clf.fit(np.hstack([X[tr], z(tr)]), labels[tr])
        return clf.predict(np.hstack([X[te], z(te)]))

    rows = []
    for i, p in enumerate(persons, 1):
        te = np.flatnonzero(use & (person == p))
        r = {"person": p}
        if not len(te):
            continue
        for name, trm, labels in [("EB", lab_mask, y), ("EB+T", lab_mask | talk_mask, y_t)]:
            tr = np.flatnonzero(trm & (person != p))
            pr = fit_predict(tr, te, labels)
            m_lab, m_talk = lab_mask[te], talk_mask[te]
            if m_lab.any():
                nf = noface[noface["person"] == p]
                yt = np.r_[y[te][m_lab], nf[LABEL].values]
                yp = np.r_[pr[m_lab], np.array(["alert"] * len(nf), dtype=object)]
                r[f"F1_{name}"] = f1_score(yt, yp, labels=sorted(set(yt)), average="macro", zero_division=0)
                r[f"false_yawn_{name}"] = int(((yp == "yawning") & (yt != "yawning")).sum())
                if name == "EB":
                    r["yawn_frames"] = int((yt == "yawning").sum())
                r[f"yawn_found%_{name}"] = (100 * ((yp == "yawning") & (yt == "yawning")).sum()
                                            / max((yt == "yawning").sum(), 1))
            if m_talk.any():
                r["talk_frames"] = int(m_talk.sum())
                r[f"talk_yawn%_{name}"] = 100 * float((pr[m_talk] == "yawning").mean())
        rows.append(r)
        print(f"[{i}/{len(persons)}] {p}: done ({(time.time() - t0) / 60:.1f} min)", flush=True)

    res = pd.DataFrame(rows).set_index("person")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print("\n" + res.round(3).to_string())
    print("\nMean over people:")
    print(res.mean(numeric_only=True).round(3).to_string())
    tot_t = res["talk_frames"].sum()
    for name in ["EB", "EB+T"]:
        k = (res[f"talk_yawn%_{name}"] * res["talk_frames"] / 100).sum()
        print(f"Talking frames in total: {int(tot_t)}; {name} called {int(round(k))} of them a yawn ({100 * k / tot_t:.1f}%)")

    print("\nSignificance (Wilcoxon over people):")
    for col, nm in [("F1", "drowsiness macro-F1"), ("talk_yawn%", "share of talking -> yawn"),
                    ("yawn_found%", "share of yawns found")]:
        d = res[[f"{col}_EB", f"{col}_EB+T"]].dropna()
        diff = d[f"{col}_EB+T"] - d[f"{col}_EB"]
        p = wilcoxon(d[f"{col}_EB+T"], d[f"{col}_EB"]).pvalue if (diff != 0).any() else np.nan
        print(f"  {nm}: EB {d[f'{col}_EB'].mean():.3f} -> EB+T {d[f'{col}_EB+T'].mean():.3f} "
              f"(people {len(d)}, p={p:.4f})")
    out = os.path.join(a.out, "talking_check_drowsy.csv")
    res.to_csv(out)
    print(f"\nSaved: {out} | {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
