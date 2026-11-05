"""Step 15. Do presentation attacks fool the driver-state model?
Runs 11_demo_live.py (without a window) on every video in data_raw\\spoof and computes
the share of time the system sees an "alert driver looking at the road".
The first 3 s of each recording are skipped (start of recording), then 10 s of calibration, as in the demo.

File names: p<number>_<type>.<mp4|mov|...>, type: live_normal, live_still, photo_print, photo_screen, video_replay.
Run (from the project root):
  python code\\15_spoof_fool.py
Output: results\\spoof\\state_<file>.csv (per frame) and results\\spoof\\fool_summary.csv
"""
import argparse
import os
import re
import subprocess
import sys

import pandas as pd

TYPES = ["live_normal", "live_still", "photo_print", "photo_screen", "video_replay"]
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".3gp"}


def parse(name):
    m = re.match(r"^(p\d+)_(" + "|".join(TYPES) + r")", name, re.I)
    return (m.group(1).lower(), m.group(2).lower()) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=r"data_raw\spoof")
    ap.add_argument("--out", default=r"results\spoof")
    ap.add_argument("--redo", action="store_true", help="recompute even finished files")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    demo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "11_demo_live.py")

    files = sorted(f for f in os.listdir(a.input) if os.path.splitext(f)[1].lower() in VIDEO_EXT)
    rows = []
    for f in files:
        pt = parse(f)
        if pt is None:
            print("skipping (name does not match p<number>_<type>):", f)
            continue
        name = os.path.splitext(f)[0]
        log = os.path.join(a.out, f"state_{name}.csv")
        if a.redo or not os.path.exists(log):
            print("processing", f, "...", flush=True)
            r = subprocess.run([sys.executable, demo, "--video", os.path.join(a.input, f), "--no-window",
                                "--log", log, "--skip", "3"], capture_output=True, text=True, errors="replace")
            if r.returncode != 0:
                print("  ERROR:\n", r.stderr[-1500:])
                continue
        d = pd.read_csv(log)
        dc = d[d["calibrated"] == 1]
        ok = (dc["face"] == 1) & (dc["drowsy"] == "alert") & (dc["attention"] == "looking_road")
        rows.append({"file": f, "person": pt[0], "type": pt[1],
                     "sec": round(d["t"].iloc[-1] if len(d) else 0, 1) if "t" in d else None,
                     "face found %": round(100 * d["face"].mean(), 1),
                     "alert and looking at road %": round(100 * ok.mean(), 1) if len(dc) else None,
                     "eyes_closed %": round(100 * (dc["drowsy"] == "eyes_closed").mean(), 1) if len(dc) else None,
                     "yawning %": round(100 * (dc["drowsy"] == "yawning").mean(), 1) if len(dc) else None,
                     "not_looking_road %": round(100 * (dc["attention"] == "not_looking_road").mean(), 1)
                     if len(dc) else None})
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
    print("\nWhat the state model sees (after 10 s of calibration):")
    print(res.to_string(index=False))
    res.to_csv(os.path.join(a.out, "fool_summary.csv"), index=False)
    print("\nAn attack fooled the system if photo/video has a high 'alert and looking at road %': "
          "the system does not notice that there is no live person in front of it.")


if __name__ == "__main__":
    main()
