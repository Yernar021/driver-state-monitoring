"""Step 13. Event-level metric (LOPO, model EB; for drowsiness also EB+T with talking frames).

Definitions (fixed BEFORE the run, not changed after seeing results):
  event           - an uninterrupted run of one label in a recording (5 frames/s, consecutive frames);
                    classes: yawning, eyes_closed (drowsiness), not_looking_road (distraction);
  alarm (K)       - the model outputs this class for K frames in a row; K = 1 (immediately) and K = 5 (1 s);
  event detected  - the alarm is active at any moment from the event start to 1 s after its end;
  delay           - from the event start to the first moment with an active alarm (s), median over detected;
  false alarm     - an alarm onset more than 1 s from every event of this class; per minute of recording;
  talking         - for yawns: onsets of a "yawn" alarm during talking (column talking), per minute of talking.
Model outputs are computed for ALL frames of a recording (unlabelled ones too); frames without a face get
the fixed answer (alert / not_looking_road), as everywhere.

Run (from the project root):
  python code\\13_event_metric.py --task drowsy
  python code\\13_event_metric.py --task attention
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

BASE = ["ear", "mar", "yaw", "pitch", "roll"]
TASKS = {"drowsy": ("drowsy_label", ["alert", "eyes_closed", "yawning"], ["yawning", "eyes_closed"]),
         "attention": ("attention_label", ["looking_road", "not_looking_road"], ["not_looking_road"])}
NOFACE_RULE = {"drowsy": "alert", "attention": "not_looking_road"}
BS = [f"bs_{i}" for i in range(52)]
WINDOW, CALIB_SEC, FPS, N_PCA, PCA_FIT_MAX = 5, 10, 5, 32, 20000
KS, TOL = [1, 5], 5   # alarm: K frames in a row; tolerance 5 frames = 1 s


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


def runs(mask):
    """Runs of True: list of (start, end), inclusive."""
    m = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(m == 1), np.flatnonzero(m == -1) - 1))


def active(pred_is, k):
    """The alarm is active at frame t if the class was predicted in k consecutive frames ending at t."""
    c = np.zeros(len(pred_is), int)
    run = 0
    for t, v in enumerate(pred_is):
        run = run + 1 if v else 0
        c[t] = run
    return c >= k


def video_events(y, pred, cls, k, talk=None):
    """Statistics of one recording for one class and one K."""
    ev = runs(y == cls)
    act = active(pred == cls, k)
    onsets = np.flatnonzero(act & ~np.r_[False, act[:-1]])
    near = np.zeros(len(y), bool)
    det, lat = 0, []
    for s, e in ev:
        near[max(s - TOL, 0):e + TOL + 1] = True
        w = np.flatnonzero(act[s:e + TOL + 1])
        if len(w):
            det += 1
            lat.append(w[0] / FPS)
    fa = int((~near[onsets]).sum()) if len(onsets) else 0
    talk_fa = int(talk[onsets].sum()) if (talk is not None and len(onsets)) else 0
    return len(ev), det, lat, fa, talk_fa


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
    label_col, classes, ev_classes = TASKS[a.task]
    rule = NOFACE_RULE[a.task]

    lab = pd.read_csv(a.labels, keep_default_na=False, na_values=[""], low_memory=False)
    for c in BASE:
        lab[c] = pd.to_numeric(lab[c], errors="coerce")
    lab["face_found"] = pd.to_numeric(lab["face_found"], errors="coerce").fillna(0)
    tk = lab["talking"].fillna("").astype(str).str.strip().str.lower()
    lab["is_talk"] = (tk.str.contains("talk") & ~tk.str.contains("not")).values
    lab = lab.sort_values(["video_id", "frame_idx"]).reset_index(drop=True)

    face = lab[lab["face_found"] == 1].copy()
    face["eidx"] = np.arange(len(face))
    n_emb = len(np.load(a.emb, mmap_mode="r"))
    if n_emb != len(face):
        raise SystemExit(f"Embedding cache ({n_emb}) does not match the frames ({len(face)})")

    # recordings to evaluate: those with task labels (+ for drowsiness, recordings with talking)
    has_lab = lab.groupby("video_id")[label_col].apply(lambda s: s.isin(classes).any())
    vids_eval = set(has_lab[has_lab].index)
    if a.task == "drowsy":
        vids_eval |= set(lab.loc[lab["is_talk"], "video_id"])
    face = face[face["video_id"].isin(vids_eval)]

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
    X = f[feats_A + BS + ["bs_ok"]].values.astype(np.float32)

    vids = f["video_id"].values
    E_all = np.load(a.emb, mmap_mode="r")
    Ec = np.asarray(E_all[f["eidx"].values], dtype=np.float32)
    del E_all
    cal = f["is_cal"].values
    for v in np.unique(vids):
        m = vids == v
        Ec[m] -= np.median(Ec[m & cal], axis=0)

    lab_mask = f[label_col].isin(classes).values
    talk_mask = f["is_talk"].values & ~lab_mask
    y = f[label_col].values.astype(object)
    y_t = y.copy(); y_t[talk_mask] = "alert"
    person = f["person"].values
    configs = {"EB": (lab_mask, y)}
    if a.task == "drowsy":
        configs["EB+T"] = (lab_mask | talk_mask, y_t)

    # outputs for all frames of the evaluated recordings: the fixed rule by default (frames without a face)
    lab_eval = lab[lab["video_id"].isin(vids_eval)].copy()
    rowpos = pd.Series(np.arange(len(lab_eval)), index=pd.MultiIndex.from_arrays(
        [lab_eval["video_id"], lab_eval["frame_idx"]]))
    f_to_lab = rowpos.reindex(pd.MultiIndex.from_arrays([f["video_id"], f["frame_idx"]])).values
    preds = {c: np.array([rule] * len(lab_eval), dtype=object) for c in configs}

    persons = sorted(set(lab_eval["person"]))
    for i, p in enumerate(persons, 1):
        te = np.flatnonzero(person == p)
        if not len(te):
            continue
        for cfg, (trm, labels) in configs.items():
            tr = np.flatnonzero(trm & (person != p))
            sub = tr[np.random.RandomState(0).permutation(len(tr))[:PCA_FIT_MAX]]
            sc = StandardScaler().fit(Ec[sub])
            pca = PCA(N_PCA, random_state=0).fit(sc.transform(Ec[sub]))
            z = lambda ii: np.vstack([pca.transform(sc.transform(Ec[ii[j:j + 10000]]))
                                      for j in range(0, len(ii), 10000)])
            clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                                         random_state=0, n_jobs=a.jobs)
            clf.fit(np.hstack([X[tr], z(tr)]), labels[tr])
            preds[cfg][f_to_lab[te]] = clf.predict(np.hstack([X[te], z(te)]))
        print(f"[{i}/{len(persons)}] {p}: done ({(time.time() - t0) / 60:.1f} min)", flush=True)

    # events
    out_rows, per_person = [], []
    yl = lab_eval[label_col].fillna("").astype(str).values
    talk_l = lab_eval["is_talk"].values
    groups = lab_eval.groupby("video_id").indices
    labeled_vids = set(has_lab[has_lab].index)
    for cfg in configs:
        pr = preds[cfg]
        for cls in ev_classes:
            for k in KS:
                tot = dict(events=0, found=0, fa=0, talk_fa=0, minutes=0.0, talk_minutes=0.0)
                lats, pp = [], {}
                for vid, ix in groups.items():
                    yv, pv, tv = yl[ix], pr[ix], talk_l[ix]
                    n_ev, det, lat, fa, tfa = video_events(yv, pv, cls, k, tv)
                    labeled_video = vid in labeled_vids
                    p = lab_eval["person"].values[ix[0]]
                    if labeled_video:
                        tot["events"] += n_ev; tot["found"] += det; lats += lat
                        tot["fa"] += fa; tot["minutes"] += len(ix) / FPS / 60
                        e = pp.setdefault(p, [0, 0, 0, 0.0])
                        e[0] += n_ev; e[1] += det; e[2] += fa; e[3] += len(ix) / FPS / 60
                    tot["talk_fa"] += tfa; tot["talk_minutes"] += tv.sum() / FPS / 60
                row = {"model": cfg, "class": cls, "K (frames in a row)": k, "events": tot["events"],
                       "detected %": round(100 * tot["found"] / max(tot["events"], 1), 1),
                       "delay, s (median)": round(float(np.median(lats)), 2) if lats else np.nan,
                       "false alarms/min": round(tot["fa"] / max(tot["minutes"], 1e-9), 2)}
                if a.task == "drowsy" and cls == "yawning":
                    row["yawn alarms/min of talking"] = round(tot["talk_fa"] / max(tot["talk_minutes"], 1e-9), 2)
                out_rows.append(row)
                for p, (n_ev, det, fa, mins) in pp.items():
                    per_person.append(dict(model=cfg, cls=cls, K=k, person=p,
                                           recall=det / n_ev if n_ev else np.nan, fa_per_min=fa / mins))

    res = pd.DataFrame(out_rows)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
    print("\nEvent-level metric (all people pooled):")
    print(res.to_string(index=False))

    pp = pd.DataFrame(per_person)
    if a.task == "drowsy":
        print("\nEB vs EB+T over people (Wilcoxon):")
        for cls in ev_classes:
            for k in KS:
                q = pp[(pp.cls == cls) & (pp.K == k)].pivot(index="person", columns="model")
                for metric, nm in [("recall", "share detected"), ("fa_per_min", "false alarms/min")]:
                    d = q[metric].dropna()
                    diff = d["EB+T"] - d["EB"]
                    pv = wilcoxon(d["EB+T"], d["EB"]).pvalue if (diff != 0).any() else np.nan
                    print(f"  {cls}, K={k}, {nm}: EB {d['EB'].mean():.3f} -> EB+T {d['EB+T'].mean():.3f} "
                          f"(people {len(d)}, p={pv:.4f})")

    res.to_csv(os.path.join(a.out, f"events_{a.task}.csv"), index=False)
    pp.to_csv(os.path.join(a.out, f"events_per_person_{a.task}.csv"), index=False)
    po = lab_eval[["video_id", "frame_idx", "person", label_col, "is_talk"]].copy()
    for cfg in configs:
        po[f"pred_{cfg}"] = preds[cfg]
    po.to_csv(os.path.join(a.out, f"preds_events_{a.task}.csv"), index=False)
    print(f"\nSaved to {a.out}\\ (events_, events_per_person_, preds_events_{a.task}.csv) | "
          f"{(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
