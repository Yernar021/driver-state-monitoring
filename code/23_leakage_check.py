"""Step 23. Check that there is no identity leakage in EB+T.
Repeats the training-frame selection rule of 12_talking_check.py / 13_event_metric.py:
    train = (labelled drowsiness frames | talking frames from the distraction sessions) & (person != test person)
and checks for each test person that none of their frames (including talking frames from the distraction
sessions) is in training. The person ID (gA_1, gB_6, ...) is the same in both DMD tasks.

Run (from the project root):
  python code\\23_leakage_check.py
"""
import os
import pandas as pd

lab = pd.read_csv(os.path.join("data_processed", "dmd", "labels.csv"), keep_default_na=False, low_memory=False,
                  usecols=["video_id", "person", "subset", "face_found", "drowsy_label", "talking"])
lab["face"] = pd.to_numeric(lab["face_found"], errors="coerce").fillna(0) == 1
drowsy = lab["drowsy_label"].isin(["alert", "eyes_closed", "yawning"])
tk = lab["talking"].astype(str).str.strip().str.lower()
talk = tk.str.contains("talk") & ~tk.str.contains("not") & ~drowsy
use = (drowsy | talk) & lab["face"]

both = sorted(set(lab.loc[drowsy, "person"]) & set(lab.loc[talk, "person"]))
print(f"People with drowsiness frames: {lab.loc[drowsy, 'person'].nunique()}; "
      f"with talking frames: {lab.loc[talk, 'person'].nunique()}; in both tasks: {len(both)}")
print("So leakage would be possible if the test person's talking frames entered training.\n")

rows, ok = [], True
for p in sorted(lab.loc[drowsy, "person"].unique()):
    train = use & (lab["person"] != p)
    own_talk = int((talk & lab["face"] & (lab["person"] == p)).sum())
    in_train = int((train & (lab["person"] == p)).sum())
    ok &= in_train == 0
    rows.append({"test person": p,
                 "own talking frames (excluded)": own_talk,
                 "talking frames in training (other people)": int((train & talk).sum()),
                 "frames of test person in training": in_train})
print(pd.DataFrame(rows).to_string(index=False))
print("\nRESULT:", "no leakage - 0 frames of the test person in training in every fold" if ok
      else "LEAKAGE FOUND - EB+T must be rerun")
