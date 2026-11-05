"""Step 17. Blink-based liveness: a rule without training.
Rule (fixed in advance): at time t the recording is judged LIVE if at least one blink started in the last
T seconds. Pre-specified blink detector = the mean eyeBlinkLeft/Right blendshape rising above 0.5
(standard 0..1 scale, fixed in advance). Exploratory (after the pre-specified detector proved insensitive):
thresholds 0.4/0.3/0.2 and EAR dropping below 0.8/0.7 of its own median over the previous 10 s;
ALL variants are reported. Frames with two faces or no face produce no blinks.
The first/last 3 s of each recording are discarded (as in 16).
T = 3, 5, 10, 15, 20 s - ALL are reported (trade-off between live-face errors and decision time); no best one is picked.
Nothing is learned, so the rule cannot learn the place or lighting of the recordings.

The control recording (files with a suffix after the type, e.g. p1_live_normal_b) is reported separately.

Run (from the project root), needs the cache from 16_liveness.py:
  python code\\17_blink_rule.py
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

TYPES = ["live_normal", "live_still", "photo_print", "photo_screen", "video_replay"]
ATTACKS = ["photo_print", "photo_screen", "video_replay"]
TS = [3, 5, 10, 15, 20]
TRIM, MAX_S = 3.0, 60.0   # trim the edges; at most 60 s of each recording after trimming (equal weight per recording)


def parse(name):
    m = re.match(r"^(p\d+)_(" + "|".join(TYPES) + r")(.*)$", name, re.I)
    return (m.group(1).lower(), m.group(2).lower(), m.group(3)) if m else None


DETECTORS = [("bs>=0.5 (pre-specified)", "bs", 0.5), ("bs>=0.4", "bs", 0.4), ("bs>=0.3", "bs", 0.3),
             ("bs>=0.2", "bs", 0.2), ("EAR<0.8*own median", "ear", 0.8), ("EAR<0.7*own median", "ear", 0.7)]


def blink_onsets(d, kind="bs", thr=0.5):
    """Blink onsets. bs: the mean eyeBlinkL/R rises above thr. ear: EAR drops below thr * the median EAR
    over the last 10 s (the person's own baseline, past only)."""
    ok = (d["face"] == 1) & (d["two"] == 0)
    fps = float(d["fps"])
    if kind == "bs":
        names = [str(x) for x in d["blend_names"]]
        ib = [names.index(n) for n in ["eyeBlinkLeft", "eyeBlinkRight"] if n in names]
        B = d["blend"][:, ib] if ib else np.zeros((len(ok), 1))
        cnt = np.isfinite(B).sum(1)
        b = np.where(cnt > 0, np.where(np.isfinite(B), B, 0).sum(1) / np.maximum(cnt, 1), 0.0)
        b = np.where(ok, b, 0.0)
        hit = b >= thr
    else:
        e = pd.Series(np.where(ok, d["ear"], np.nan).astype(float))
        norm = e.rolling(int(round(10 * fps)), min_periods=int(round(2 * fps))).median().shift(1)
        hit = (e < thr * norm).fillna(False).values & ok
    up = np.r_[False, hit[1:] & ~hit[:-1]]
    return up, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=r"results\spoof")
    a = ap.parse_args()
    rows, vids = [], []
    for fn in sorted(glob.glob(os.path.join(a.cache, "frames_*.npz"))):
        name = os.path.basename(fn)[7:-4]
        pt = parse(name)
        if pt is None:
            continue
        person, typ, suffix = pt
        group = typ if not suffix else f"{typ} (control{suffix})"
        d = dict(np.load(fn))
        fps = float(d["fps"]); n = len(d["t"])
        cut = int(round(TRIM * fps))
        n = min(n, 2 * cut + int(round(MAX_S * fps)))   # end of the used segment = cut + 60 s (+ cut as margin)
        ok0 = (d["face"] == 1) & (d["two"] == 0)
        if ok0[cut:n - cut].mean() < 0.5:
            print("excluded (face < 50%):", name); continue
        dur_min = (n - 2 * cut) / fps / 60
        vrow = {"recording": name, "group": group}
        for det, kind, thr in DETECTORS:
            up, ok = blink_onsets(d, kind, thr)
            vrow[det] = round(int(up[cut:n - cut].sum()) / dur_min, 1)
            cs = np.cumsum(up)
            for T in TS:
                W = int(round(T * fps))
                for t in range(cut + W, n - cut + 1, int(round(fps))):   # one decision per second
                    live = (cs[t - 1] - cs[t - W - 1]) > 0 if t - W - 1 >= 0 else cs[t - 1] > 0
                    rows.append({"detector": det, "T": T, "recording": name, "person": person, "group": group,
                                 "judged_live": bool(live)})
        vids.append(vrow)
    if not rows:
        raise SystemExit("No cache: run python code\\16_liveness.py first")
    R = pd.DataFrame(rows)
    V = pd.DataFrame(vids)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print("Blinks per minute per recording for each detector (people usually 10-20/min; a photo should give 0):")
    print(V.to_string(index=False))

    out = []
    for det, _, _ in DETECTORS:
        for T in TS:
            r = R[(R["detector"] == det) & (R["T"] == T)]
            row = {"detector": det, "T, s": T}
            m = r["group"].isin(["live_normal", "live_still"])
            row["BPCER %"] = round(100 * (~r.loc[m, "judged_live"]).mean(), 1)
            for t_ in ["live_normal", "live_still"]:
                q = r[r["group"] == t_]
                row[f"BPCER {t_} %"] = round(100 * (~q["judged_live"]).mean(), 1) if len(q) else np.nan
            ctl = r[r["group"].str.contains("control")]
            row["control: live %"] = round(100 * ctl["judged_live"].mean(), 1) if len(ctl) else np.nan
            ap_ = []
            for t_ in ATTACKS:
                q = r[r["group"] == t_]
                v = round(100 * q["judged_live"].mean(), 1) if len(q) else np.nan
                row[f"APCER {t_} %"] = v
                if len(q):
                    ap_.append(v)
            row["ACER %"] = round((max(ap_) + row["BPCER %"]) / 2, 1) if ap_ else np.nan
            out.append(row)
    O = pd.DataFrame(out)
    O = O.dropna(axis=1, how="all")
    print("\nPRE-SPECIFIED result (threshold 0.5 fixed in advance):")
    print(O[O["detector"] == DETECTORS[0][0]].to_string(index=False))
    print("\nEXPLORATORY analysis: all blink detectors, T = 10 s (other T values are in the csv):")
    print(O[O["T, s"] == 10].to_string(index=False))
    O.to_csv(os.path.join(a.cache, "blink_rule_results.csv"), index=False)
    V.to_csv(os.path.join(a.cache, "blink_rule_per_video.csv"), index=False)
    print(f"\nSaved to {a.cache}\\ (blink_rule_results.csv, blink_rule_per_video.csv)")


if __name__ == "__main__":
    main()
