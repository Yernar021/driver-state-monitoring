"""Step 4b. Geometry baseline (model A in the paper) under the subject-independent protocol (leave-one-person-out).

Protocol:
  1) Frames without a face are NOT dropped. The model is not run on them; the answer is fixed:
     attention -> not_looking_road, drowsy -> alert. These frames count in every metric.
     (--noface drop restores the old behaviour, for comparison only.)
  2) Leak-free normalisation (--norm calib): from the base features of each recording we subtract
     the median over its first --calib seconds (frames with a face only, no labels used),
     as if the system calibrated at the start of a trip. Window statistics are computed
     on the normalised features.
  3) Settings fixed in advance: window 5 frames, calibration 10 s. Do NOT tune them on test results.

Run (from the project root):
  python code\\04b_baseline_m1_honest.py --labels data_processed\\dmd\\labels.csv --task drowsy
  python code\\04b_baseline_m1_honest.py --labels data_processed\\dmd\\labels.csv --task attention
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

BASE = ["ear", "mar", "yaw", "pitch", "roll"]
TASKS = {
    "drowsy": ("drowsy_label", ["alert", "eyes_closed", "yawning"]),
    "attention": ("attention_label", ["looking_road", "not_looking_road"]),
}
NOFACE_RULE = {"drowsy": "alert", "attention": "not_looking_road"}
FPS = 5  # frame rate after 02_extract_features.py


def add_window_features(df, k):
    df = df.sort_values(["video_id", "frame_idx"]).copy()
    g = df.groupby("video_id")
    feats = list(BASE)
    if k > 1:
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
        wcols = [f for f in feats if f not in BASE]
        df[wcols] = df[wcols].fillna(0)
    return df, feats


def calib_normalize(face, seconds):
    """Subtract from each recording's features the median over its first `seconds` seconds (frames with a face)."""
    face = face.sort_values(["video_id", "frame_idx"]).copy()
    n = max(1, int(round(seconds * FPS)))
    ref = face.groupby("video_id").head(n).groupby("video_id")[BASE].median()
    face[BASE] = face[BASE].values - ref.reindex(face["video_id"]).values
    return face


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--task", choices=list(TASKS), required=True)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--norm", choices=["none", "calib"], default="calib")
    ap.add_argument("--calib", type=float, default=10.0, help="calibration seconds at the start of each recording")
    ap.add_argument("--noface", choices=["rule", "drop"], default="rule")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    label_col, classes = TASKS[a.task]
    rule_label = NOFACE_RULE[a.task]
    df = pd.read_csv(a.labels, keep_default_na=False, na_values=[""], low_memory=False)
    for c in BASE:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["face_found"] = pd.to_numeric(df["face_found"], errors="coerce").fillna(0)

    # Frames with a face: features, calibration and windows use ALL such frames of a recording
    # (including frames without the task label), as the system would in a car.
    face = df[df["face_found"] == 1].copy()
    face[BASE] = face.groupby("video_id")[BASE].ffill(limit=3)
    face = face.dropna(subset=BASE)
    if a.norm == "calib":
        face = calib_normalize(face, a.calib)
    face, feats = add_window_features(face, a.window)

    labeled = df[df[label_col].isin(classes)]
    face = face[face[label_col].isin(classes)].copy()
    noface = labeled[~labeled.index.isin(face.index)].copy()  # no face, or features could not be computed
    if a.noface == "drop":
        noface = noface.iloc[0:0]

    persons = sorted(set(face["person"]) | set(noface["person"]))
    tag = f"window: {a.window} | norm: {a.norm}" + (f" ({a.calib:g} s)" if a.norm == "calib" else "")
    print(f"\nTask: {a.task} | features: {len(feats)} | {tag} | no face: {a.noface}")
    print(f"People: {persons}")
    cnt = pd.concat([face, noface]).groupby(["person", label_col]).size().unstack(fill_value=0)
    cnt["of which no face"] = noface.groupby("person").size().reindex(cnt.index, fill_value=0)
    print(cnt, "\n")

    rows, all_true, all_pred, per_person_cm, yawn_recall = [], [], [], {}, []
    for p in persons:
        tr = face[face["person"] != p]
        te_face = face[face["person"] == p]
        te_nf = noface[noface["person"] == p]
        te = pd.concat([te_face, te_nf])
        y = te[label_col].values
        rule = np.array([rule_label] * len(te_nf), dtype=object)
        preds, f1s = [], []
        present = sorted(set(y))
        for seed in range(a.seeds):
            clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                         class_weight="balanced", random_state=seed, n_jobs=-1)
            clf.fit(tr[feats], tr[label_col])
            pr_face = clf.predict(te_face[feats]) if len(te_face) else np.array([], dtype=object)
            pr = np.concatenate([pr_face.astype(object), rule])
            f1s.append(f1_score(y, pr, labels=present, average="macro", zero_division=0))
            preds.append(pr)
        pr = preds[0]
        all_true += list(y); all_pred += list(pr)
        if a.task == "drowsy" and "yawning" in te.columns:
            is_y = (y == "yawning")
            for t in sorted(te.loc[is_y, "yawning"].dropna().unique()):
                mt = is_y & (te["yawning"] == t).values
                yawn_recall.append(dict(test_person=p, type=t, frames=int(mt.sum()),
                                        found=int((pr[mt] == "yawning").sum())))
        per_person_cm[p] = pd.DataFrame(confusion_matrix(y, pr, labels=classes),
                                        index=classes, columns=classes)
        row = {"test_person": p, "n": len(te), "n_noface": len(te_nf),
               "accuracy": accuracy_score(y, pr),
               "macro_F1_mean": np.mean(f1s), "macro_F1_std": np.std(f1s)}
        for c in classes:
            if (y == c).any():
                row[f"F1_{c}"] = f1_score(y == c, pr == c, zero_division=0)
        rows.append(row)

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("Results per person (the test person was not in training):")
    print(res.round(3).to_string(index=False))
    print(f"\nMean macro-F1 over people: {res.macro_F1_mean.mean():.3f} "
          f"(SD between people: ±{res.macro_F1_mean.std():.3f})")
    if len(noface):
        ok = (noface[label_col] == rule_label).mean()
        print(f"Frames without a face in test: {len(noface)}; rule '{rule_label}' is correct on {100*ok:.1f}% of them")
    print("\nConfusion matrices per person (rows: true, columns: predicted):")
    for p, cmp_ in per_person_cm.items():
        print(f"\n{p}:")
        print(cmp_)
    if yawn_recall:
        yr = pd.DataFrame(yawn_recall)
        yr["found_pct"] = (100 * yr["found"] / yr["frames"]).round(1)
        print("\nShare of yawning frames found, by yawn type:")
        print(yr.to_string(index=False))
        tot = yr.groupby("type")[["frames", "found"]].sum()
        tot["found_pct"] = (100 * tot["found"] / tot["frames"]).round(1)
        print("\nTotal over all people:")
        print(tot)
    print("\nPooled confusion matrix (rows: true, columns: predicted):")
    cm = confusion_matrix(all_true, all_pred, labels=classes)
    print(pd.DataFrame(cm, index=classes, columns=classes))
    calib_tag = f"{a.calib:g}s" if a.norm == "calib" else ""
    out = a.out or f"results_m1h_{a.task}_w{a.window}_{a.norm}{calib_tag}_{a.noface}.csv"
    res.to_csv(out, index=False)
    print("\nSaved:", out)


if __name__ == "__main__":
    main()
