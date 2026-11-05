"""Step 14. Share of detected events as a function of event length.
Uses the predictions saved by step 13 (results\\preds_events_<task>.csv); nothing is retrained.
Definitions as in 13: alarm = K frames in a row (K = 1 and 5); an event is detected if the alarm
is active from the event start to 1 s after its end. Length bins are fixed in advance.

Run (from the project root):
  python code\\14_events_by_duration.py
"""
import os

import numpy as np
import pandas as pd

FPS, TOL, KS = 5, 5, [1, 5]
BINS = [0, 0.5, 1, 2, 3, np.inf]
NAMES = ["<0.5 s", "0.5-1 s", "1-2 s", "2-3 s", ">=3 s"]
TASKS = {"drowsy": ("drowsy_label", ["yawning", "eyes_closed"], ["EB", "EB+T"]),
         "attention": ("attention_label", ["not_looking_road"], ["EB"])}


def runs(mask):
    m = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(m == 1), np.flatnonzero(m == -1) - 1))


def active(pred_is, k):
    c, run = np.zeros(len(pred_is), int), 0
    for t, v in enumerate(pred_is):
        run = run + 1 if v else 0
        c[t] = run
    return c >= k


def main():
    rows = []
    for task, (label_col, ev_classes, models) in TASKS.items():
        fn = os.path.join("results", f"preds_events_{task}.csv")
        if not os.path.exists(fn):
            print("missing", fn, "- run step 13 first"); continue
        d = pd.read_csv(fn, keep_default_na=False, low_memory=False).sort_values(["video_id", "frame_idx"])
        for cls in ev_classes:
            for m in models:
                for k in KS:
                    found = []
                    for _, g in d.groupby("video_id"):
                        y, p = g[label_col].astype(str).values, g[f"pred_{m}"].astype(str).values
                        act = active(p == cls, k)
                        for s, e in runs(y == cls):
                            found.append(((e - s + 1) / FPS, act[s:e + TOL + 1].any()))
                    if not found:
                        continue
                    f = pd.DataFrame(found, columns=["dur", "found"])
                    f["bin"] = pd.cut(f["dur"], BINS, labels=NAMES, right=False)
                    agg = f.groupby("bin", observed=False)["found"].agg(["size", "mean"])
                    row = {"task": task, "class": cls, "model": m, "K": k}
                    for nm in NAMES:
                        n, r = agg.loc[nm, "size"], agg.loc[nm, "mean"]
                        row[nm] = f"{100 * r:.0f}% ({n})" if n else "-"
                    rows.append(row)
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
    print("Share of detected events by length (number of events in brackets):")
    print(res.to_string(index=False))
    res.to_csv(os.path.join("results", "events_by_duration.csv"), index=False)
    print("\nSaved: results\\events_by_duration.csv")


if __name__ == "__main__":
    main()
