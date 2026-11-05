"""Step 3. Joins the DMD annotations (OpenLABEL .json) to features.csv.

Run (from the code folder):
  python 03_build_labels.py --ann <project>\\data_raw\\dmd\\extracted ^
         --features <project>\\data_processed\\dmd\\features.csv ^
         --out <project>\\data_processed\\dmd\\labels.csv

What it does:
  * finds all *_ann_*.json files and restores the labels of every frame of a session;
  * DMD labels come in independent "channels" (eyes_state, yawning, gaze_on_road,
    driver_actions, ...): one frame can carry several at once;
  * matches them to the frames in features.csv by (video, source frame number);
  * adds person (e.g. gA_5) and session (s1, s2, ...) so data can be split by person;
  * adds two derived labels: drowsy_label and attention_label.
"""
import os, re, json, argparse, collections
import pandas as pd


def find_jsons(roots):
    out = []
    for root in roots:
        for dp, _, fs in os.walk(root):
            for f in fs:
                if f.lower().endswith(".json") and "_ann_" in f:
                    out.append(os.path.join(dp, f))
    return sorted(out)


def session_key(openlabel, path):
    """Session key = face-video name without extension (the same as video_id in features.csv)."""
    try:
        uri = openlabel["streams"]["face_camera"]["uri"]
        return os.path.splitext(os.path.basename(uri))[0]
    except Exception:
        m = re.split(r"_rgb_ann_", os.path.basename(path))[0]
        return m + "_rgb_face"


def load_annotations(paths):
    """-> {session_key: {"subsets": set, "n": number of frames, "cats": {category: {frame: value}}}}"""
    sessions = {}
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            o = json.load(fh)["openlabel"]
        key = session_key(o, p)
        subset = os.path.basename(p).split("_ann_")[1].rsplit(".", 1)[0]
        s = sessions.setdefault(key, {"subsets": set(), "n": 0, "cats": collections.defaultdict(dict)})
        s["subsets"].add(subset)
        s["n"] = max(s["n"], o["frame_intervals"][0]["frame_end"] + 1)
        for a in o.get("actions", {}).values():
            if "/" not in a["type"]:
                continue
            cat, val = a["type"].split("/", 1)
            for iv in a["frame_intervals"]:
                for i in range(iv["frame_start"], iv["frame_end"] + 1):
                    s["cats"][cat][i] = val
    return sessions


def drowsy_label(eyes, yawn):
    if yawn:
        return "yawning"
    if eyes == "close":
        return "eyes_closed"
    if eyes == "open":
        return "alert"
    if eyes in ("closing", "opening"):
        return "transition"
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", nargs="+", required=True, help="folder(s) with the .json files")
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", default="labels.csv")
    a = ap.parse_args()

    paths = find_jsons(a.ann)
    print(f"json files found: {len(paths)}")
    sessions = load_annotations(paths)

    feats = pd.read_csv(a.features)
    feats["_key"] = feats["video_id"].astype(str).str.split("__").str[-1]
    all_cats = sorted({c for s in sessions.values() for c in s["cats"]})

    recs, missing = [], set()
    for r in feats.to_dict("records"):
        key = r["_key"]
        s = sessions.get(key)
        if s is None:
            missing.add(key)
        t = key.split("_")
        r["person"] = f"{t[0]}_{t[1]}" if len(t) > 2 else ""
        r["session"] = t[2] if len(t) > 2 else ""
        r["subset"] = "+".join(sorted(s["subsets"])) if s else ""
        fi = int(r["frame_idx"])
        for c in all_cats:
            r[c] = s["cats"].get(c, {}).get(fi, "") if s else ""
        recs.append(r)
    if missing:
        print("WARNING: no annotations for videos:", sorted(missing))
    df = pd.DataFrame(recs).drop(columns=["_key"])

    if "eyes_state" in df.columns:
        yawn = df["yawning"].astype(bool) if "yawning" in df.columns else False
        df["drowsy_label"] = [drowsy_label(e, bool(y)) if e != "" else ""
                              for e, y in zip(df["eyes_state"], yawn)]
    if "gaze_on_road" in df.columns:
        df["attention_label"] = df["gaze_on_road"]
    df.to_csv(a.out, index=False, encoding="utf-8")
    print("Saved:", a.out, "rows:", len(df))

    for col in ("drowsy_label", "attention_label", "driver_actions", "hands_using_wheel"):
        if col in df.columns:
            sub = df[df[col] != ""]
            if len(sub):
                print(f"\n{col}: by person")
                print(pd.crosstab(sub["person"], sub[col]))


if __name__ == "__main__":
    main()
