"""Step 21. Statistics for the revised paper (nothing is retrained, ~1-2 min).
Uses saved results:
  results\\preds_gazecheck_<task>.csv   (step 09: A, AB, E, EB)
  results\\preds_events_drowsy.csv       (step 13: EB and EB+T)
  data_processed\\dmd\\labels.csv        (data inclusion table)
  results\\spoof\\frames_*.npz            (step 16: per-recording liveness)
Computes:
  1) per person: mean +- SD, median [IQR], 95% bootstrap CI over people (10,000 resamples, seed 0);
  2) paired comparisons: difference, 95% CI of the difference, Cohen dz, rank-biserial r, Wilcoxon p;
  3) per-class precision / recall / F1 and confusion matrix: first per person,
     then averaged over people (long recordings do not dominate);
  4) data inclusion table (where the 80,545 frames come from);
  5) per-recording liveness (rule EAR < 0.8 x median over 10 s, T = 10 s): share of "live" decisions per
     recording, whether an attack was accepted even once, number of independent videos and presentations;
  6) figure: macro-F1 of each person (results\\fig_per_person.png).
Run (from the project root):
  python code\\21_revision_stats.py
Output: results\\revision\\*.csv and results\\fig_per_person.png
"""
import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix

TASKS = {"drowsy": ("drowsy_label", ["alert", "eyes_closed", "yawning"]),
         "attention": ("attention_label", ["looking_road", "not_looking_road"])}
OUT = os.path.join("results", "revision")
RNG = np.random.RandomState(0)
NB = 10000


def boot_ci(x):
    x = np.asarray(x, float)
    idx = RNG.randint(0, len(x), size=(NB, len(x)))
    m = x[idx].mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def summarize(v):
    lo, hi = boot_ci(v)
    q1, q3 = np.percentile(v, [25, 75])
    return {"mean": v.mean(), "sd": v.std(ddof=1), "median": np.median(v), "q1": q1, "q3": q3,
            "ci95_lo": lo, "ci95_hi": hi, "min": v.min(), "max": v.max(), "n_people": len(v)}


def paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    lo, hi = boot_ci(d)
    nz = d[d != 0]
    ranks = pd.Series(np.abs(nz)).rank().values
    wp, wm = ranks[nz > 0].sum(), ranks[nz < 0].sum()
    r = (wp - wm) / (wp + wm) if len(nz) else np.nan
    p = wilcoxon(a, b).pvalue if len(nz) else np.nan
    return {"diff_mean": d.mean(), "diff_ci95_lo": lo, "diff_ci95_hi": hi,
            "cohen_dz": d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.nan,
            "rank_biserial_r": r, "better": f"{int((d > 0).sum())}/{len(d)}", "p_wilcoxon": p}


def per_person_f1(df, label_col, pred_col):
    out = {}
    for p, g in df.groupby("person"):
        y, yp = g[label_col].astype(str).values, g[pred_col].astype(str).values
        out[p] = f1_score(y, yp, labels=sorted(set(y)), average="macro", zero_division=0)
    return pd.Series(out)


def class_report(df, label_col, pred_col, classes):
    per, cms = [], []
    for p, g in df.groupby("person"):
        y, yp = g[label_col].astype(str).values, g[pred_col].astype(str).values
        pr, rc, f1, sup = precision_recall_fscore_support(y, yp, labels=classes, zero_division=0)
        for c, a, b, f, s in zip(classes, pr, rc, f1, sup):
            if s > 0:
                per.append({"person": p, "class": c, "precision": a, "recall": b, "f1": f, "support": s})
        cm = confusion_matrix(y, yp, labels=classes).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            cms.append(cm / cm.sum(1, keepdims=True))
    per = pd.DataFrame(per)
    rep = per.groupby("class")[["precision", "recall", "f1"]].mean()
    rep["people_with_class"] = per.groupby("class")["person"].nunique()
    y, yp = df[label_col].astype(str).values, df[pred_col].astype(str).values
    pr, rc, f1, sup = precision_recall_fscore_support(y, yp, labels=classes, zero_division=0)
    rep["pooled_precision"], rep["pooled_recall"], rep["pooled_f1"], rep["frames"] = pr, rc, f1, sup
    cm = pd.DataFrame(100 * np.nanmean(np.stack(cms), 0), index=[f"true {c}" for c in classes],
                      columns=[f"pred {c}" for c in classes])
    return rep.loc[classes], cm


def main():
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    per_person_all = {}

    # ---------- 1-3: state recognition ----------
    for task, (label_col, classes) in TASKS.items():
        fn = os.path.join("results", f"preds_gazecheck_{task}.csv")
        if not os.path.exists(fn):
            print("missing", fn); continue
        d = pd.read_csv(fn, keep_default_na=False, low_memory=False)
        cfgs = [c for c in ["A", "AB", "E", "EB"] if f"pred_{c}" in d]
        P = pd.DataFrame({c: per_person_f1(d, label_col, f"pred_{c}") for c in cfgs})
        if task == "drowsy":
            fe = os.path.join("results", "preds_events_drowsy.csv")
            if os.path.exists(fe):
                e = pd.read_csv(fe, keep_default_na=False, low_memory=False)
                e = e[e[label_col].isin(classes)]
                P["EB+T"] = per_person_f1(e, label_col, "pred_EB+T")
                P["EB (run 13)"] = per_person_f1(e, label_col, "pred_EB")
        per_person_all[task] = P
        P.round(4).to_csv(os.path.join(OUT, f"per_person_{task}.csv"))
        S = pd.DataFrame({c: summarize(P[c].dropna().values) for c in P}).T
        print(f"\n===== {task}: macro-F1 per person =====")
        print(S.round(3).to_string())
        S.round(4).to_csv(os.path.join(OUT, f"summary_ci_{task}.csv"))

        pairs = [("AB", "A"), ("E", "A"), ("EB", "A"), ("EB", "E"), ("EB", "AB")]
        if "EB+T" in P:
            pairs.append(("EB+T", "EB (run 13)"))
        R = pd.DataFrame({f"{x} vs {y}": paired(P[x], P[y]) for x, y in pairs if x in P and y in P}).T
        print(f"\n----- {task}: paired comparisons over people -----")
        print(R.to_string(formatters={c: (lambda v: f"{v:.4f}") for c in R.columns if c != "better"}))
        R.to_csv(os.path.join(OUT, f"paired_{task}.csv"))

        rep, cm = class_report(d, label_col, "pred_EB", classes)
        print(f"\n----- {task}: EB, per class (mean over people; pooled = all frames together) -----")
        print(rep.round(3).to_string())
        print(f"\n----- {task}: EB, confusion matrix, row %, mean over people -----")
        print(cm.round(1).to_string())
        rep.round(4).to_csv(os.path.join(OUT, f"per_class_EB_{task}.csv"))
        cm.round(2).to_csv(os.path.join(OUT, f"confusion_EB_{task}.csv"))
        if task == "drowsy" and os.path.exists(os.path.join("results", "preds_events_drowsy.csv")):
            rep2, cm2 = class_report(e, label_col, "pred_EB+T", classes)
            print("\n----- drowsy: EB+T, per class -----")
            print(rep2.round(3).to_string())
            print("\n----- drowsy: EB+T, confusion matrix, row %, mean over people -----")
            print(cm2.round(1).to_string())
            rep2.round(4).to_csv(os.path.join(OUT, "per_class_EBT_drowsy.csv"))
            cm2.round(2).to_csv(os.path.join(OUT, "confusion_EBT_drowsy.csv"))

    # ---------- 4: data inclusion table ----------
    fl = os.path.join("data_processed", "dmd", "labels.csv")
    if os.path.exists(fl):
        L = pd.read_csv(fl, keep_default_na=False, low_memory=False,
                        usecols=lambda c: c in {"video_id", "person", "subset", "face_found", "drowsy_label",
                                                "attention_label", "talking", "transition"})
        L["face"] = pd.to_numeric(L["face_found"], errors="coerce").fillna(0).astype(int)
        print("\n===== Data inclusion table (5 frames/s) =====")
        print("Frames in total:", len(L), "| recordings:", L["video_id"].nunique(), "| people:", L["person"].nunique())
        flow = L.groupby("subset").agg(frames=("video_id", "size"), recordings=("video_id", "nunique"),
                                       people=("person", "nunique"), with_face=("face", "sum"))
        print(flow.to_string())
        rows = []
        for task, (label_col, classes) in TASKS.items():
            sub = "drowsiness" if task == "drowsy" else "distraction"
            g = L[L["subset"] == sub]
            lab = g[label_col].astype(str)
            rows.append({"task": task, "subset frames": len(g),
                         "labelled (in classes)": int(lab.isin(classes).sum()),
                         "  of which with face": int((lab.isin(classes) & (g["face"] == 1)).sum()),
                         "  of which no face": int((lab.isin(classes) & (g["face"] == 0)).sum()),
                         "not used (no label / transition / other)": int((~lab.isin(classes)).sum())})
            print(f"\n{task}: label values in the {sub} recordings:")
            print(lab.replace("", "(empty)").value_counts().to_string())
        if "transition" in L:
            print("\nFrames marked transition:", int((L["transition"].astype(str).str.strip() != "").sum()))
        tk = L["talking"].astype(str).str.strip().str.lower()
        talk = tk.str.contains("talk") & ~tk.str.contains("not")
        print("Talking frames (total / with a face):", int(talk.sum()), "/", int((talk & (L["face"] == 1)).sum()))
        R = pd.DataFrame(rows)
        print("\n" + R.to_string(index=False))
        R.to_csv(os.path.join(OUT, "inclusion_flow.csv"), index=False)
        flow.to_csv(os.path.join(OUT, "inclusion_by_subset.csv"))

    # ---------- 5: per-recording liveness ----------
    files = sorted(glob.glob(os.path.join("results", "spoof", "frames_*.npz")))
    if files:
        TRIM, MAX_S, T = 3.0, 60.0, 10
        rows = []
        for fn in files:
            name = os.path.basename(fn)[7:-4]
            m = re.match(r"^(p\d+)_(live_normal|live_still|photo_print|photo_screen|video_replay)(.*)$", name)
            if not m:
                continue
            d = dict(np.load(fn))
            fps = float(d["fps"]); n = len(d["t"]); cut = int(round(TRIM * fps))
            n = min(n, 2 * cut + int(round(MAX_S * fps)))
            ok = ((d["face"] == 1) & (d["two"] == 0))[:n]
            e = pd.Series(np.where(ok, d["ear"][:n], np.nan).astype(float))
            norm = e.rolling(int(round(10 * fps)), min_periods=int(round(2 * fps))).median().shift(1)
            hit = (e < 0.8 * norm).fillna(False).values & ok
            up = np.r_[False, hit[1:] & ~hit[:-1]]
            cs = np.cumsum(up); W = int(round(T * fps))
            dec = [(cs[t - 1] - (cs[t - W - 1] if t - W - 1 >= 0 else 0)) > 0
                   for t in range(cut + W, n - cut + 1, int(round(fps)))]
            dec = np.array(dec, bool)
            live = m.group(2).startswith("live")
            first = np.flatnonzero(dec)
            rows.append({"recording": name, "person": m.group(1),
                         "type": m.group(2) + (" (control)" if m.group(3) else ""),
                         "decisions": len(dec), "judged live %": round(100 * dec.mean(), 1),
                         "accepted at least once": bool(dec.any()),
                         "first live decision, s": (round(T + first[0], 0) if len(first) else np.nan)})
        V = pd.DataFrame(rows)
        print("\n===== Liveness per recording (EAR<0.8 x median, T=10 s) =====")
        print(V.to_string(index=False))
        atk = V[~V["type"].str.startswith("live")]
        lv = V[V["type"].str.startswith("live") & ~V["type"].str.contains("control")]
        print(f"\nIndependent recordings: {len(V)} ({V['person'].nunique()} people); "
              f"main live {len(lv)}, control {int(V['type'].str.contains('control').sum())}, attacks {len(atk)}")
        for t in sorted(atk["type"].unique()):
            a_ = atk[atk["type"] == t]
            print(f"  {t}: accepted at least once in {int(a_['accepted at least once'].sum())} of {len(a_)} presentations")
        print(f"  live: judged live by a majority of decisions in "
              f"{int((lv['judged live %'] >= 50).sum())} of {len(lv)} recordings; "
              f"median share of live decisions {lv['judged live %'].median():.1f}% "
              f"[min {lv['judged live %'].min():.1f}%]")
        V.to_csv(os.path.join(OUT, "liveness_per_recording.csv"), index=False)

    # ---------- 6: figure ----------
    if per_person_all:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(per_person_all), figsize=(9, 3.6), sharey=False)
        axes = np.atleast_1d(axes)
        names = {"drowsy": "Drowsiness", "attention": "Distraction"}
        for ax, (task, P) in zip(axes, per_person_all.items()):
            cols = [c for c in ["A", "AB", "E", "EB"] if c in P]
            for _, r in P[cols].iterrows():
                ax.plot(range(len(cols)), r.values, color="0.75", lw=0.8, zorder=1)
            for j, c in enumerate(cols):
                ax.scatter(np.full(len(P), j), P[c], s=14, color="0.35", zorder=2)
                lo, hi = boot_ci(P[c].values)
                ax.errorbar(j + 0.18, P[c].mean(), yerr=[[P[c].mean() - lo], [hi - P[c].mean()]],
                            fmt="o", color="black", ms=5, capsize=3, zorder=3)
            ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols)
            ax.set_title(names[task], fontsize=10); ax.set_ylabel("Macro-F1 per participant")
            ax.grid(axis="y", color="0.9")
        fig.tight_layout()
        fig.savefig(os.path.join("results", "fig_per_person.png"), dpi=300)
        fig.savefig(os.path.join("results", "fig_per_person.pdf"))
        print("\nFigure: results\\fig_per_person.png (grey dots - people, black dot - mean and 95% CI)")
    print("\nEverything saved to results\\revision\\")


if __name__ == "__main__":
    main()
