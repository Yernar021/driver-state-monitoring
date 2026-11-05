# Lightweight Driver State Monitoring Under Subject-Independent Evaluation and Static-Photo Presentation Attacks

Code for the paper by Bolysbek Yernar and Yelaman Yergakiev (Astana IT University).

A CPU-only driver monitor (drowsiness and distraction) evaluated with leave-one-person-out (LOPO) cross-validation on the public Driver Monitoring Dataset (DMD), plus an exploratory study of static-photo presentation attacks and a training-free blink rule.

## What is (and is not) in this repository

- **Included:** all scripts and notebooks needed to reproduce the results in the paper.
- **Not included:** DMD videos and anything derived from them (features, landmarks, embeddings, predictions, trained models). The DMD licence (CC BY-NC-ND 4.0) does not allow redistribution of derived data. Request the dataset from the authors: https://dmd.vicomtech.org
- **Not included:** the spoofing recordings (5 volunteers). They show identifiable faces and are not released. Scripts 15–17 are provided so the procedure can be inspected and applied to new recordings.
- **Not included:** `face_landmarker.task`. Download the MediaPipe Face Landmarker model from https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker and place it in `code/`.

## Setup

```
pip install -r requirements.txt
```

Scripts are run from the project root (the folder that contains `code/`, `data_raw/`, `data_processed/`, `results/`). Expected layout:

```
data_raw/dmd/extracted/    DMD face-camera videos (.mp4) and annotation files (.json)
data_raw/spoof/            own spoofing recordings (not released)
data_processed/dmd/        created by steps 02, 02b, 03
results/                   created by the other steps
models/                    created by step 10
```

## Pipeline (run in this order)

| Step | File | What it does | Paper |
| --- | --- | --- | --- |
| 1 | `code/02_extract_features.py` | Samples DMD face video at 5 fps, MediaPipe face detection, EAR, mouth ratio, head pose; saves face crops | 3.1, 4.1 |
| 2 | `code/03_build_labels.py` | Joins frames with DMD annotations, builds `labels.csv` (drowsiness, distraction, talking, yawn type, person ID) | 3.1, Table 1 |
| 3 | `code/02b_extract_landmarks.py` | 478 landmarks and 52 blendshapes per frame (`landmarks/*.npz`) | 4.1 |
| 4 | `notebooks/m0b_honest_colab.ipynb` | MobileNetV2 embeddings of all face frames (`results/emb_all_v2.npy`); first LOPO comparison A / E (Colab, T4) | 4.1 |
| 5 | `code/04b_baseline_m1_honest.py` | Geometry baseline A with calibration and 1 s window | 6.1 |
| 6 | `code/09_gaze_check.py` | Random forests A, AB, E, EB; probability smoothing | 6.1, 6.2 |
| 7 | `notebooks/gru_memory_colab.ipynb` | GRU with 2 s memory, inputs G, GB, GBE, P; seeds 0–2 (Colab, T4) | 6.2, Table 6 |
| 8 | `code/12_talking_check.py` | EB vs EB+T (talking frames as extra negatives) | 6.3, Table 7 |
| 9 | `code/13_event_metric.py`, `code/14_events_by_duration.py` | Event-level detection, delay, false alarms per minute | 6.4, Tables 8–9 |
| 10 | `code/10_train_final.py` | Trains the final models on all people (EB+T drowsiness, EB distraction) | 4.2 |
| 11 | `code/11_demo_live.py` | Live webcam demo and per-stage timing | 6.5 |
| 12 | `code/15_spoof_fool.py` | Final state model on spoofing videos | 6.6 |
| 13 | `code/16_liveness.py` | Learned attack detector (motion / image features), LOPO over 5 people | 6.6, Table 10 |
| 14 | `code/17_blink_rule.py` | Training-free blink rule, blink detectors, decision- and recording-level results | 6.6, Table 11 |
| 15 | `code/21_revision_stats.py` | Bootstrap CIs, paired comparisons, per-class results, confusion matrices, per-person figure, inclusion table | 6.1, Tables 1–4, Fig. 2 |
| 16 | `code/22_ablation.py` | Ablation of calibration and 1 s window features | Table 5 |
| 17 | `code/23_leakage_check.py` | Checks that no LOPO training set contains frames of the held-out person | 4.2 |

The notebooks were run in Google Colab and read their inputs from `MyDrive/driver_project/` (paths are set in the first cells).

## Evaluation protocol and randomness

- **Folds:** leave-one-person-out by DMD participant ID (for example `gA_1`). The same ID is used in the drowsiness and distraction sessions, so every recording of the held-out person, including talking frames used in EB+T, is excluded from training. Drowsiness: 13 people; distraction: 14 people.
- **Calibration:** each recording minus the median of its first 10 s (50 frames with a face); no labels used.
- **Fitted only on training people:** scaling, PCA (32 components, fitted on a 20,000-frame subsample), classifiers.
- **Frames without a face** stay in the test set: drowsiness → `alert`, distraction → `not_looking_road`.
- **Seeds:** 0 for all random forests (300 trees, `min_samples_leaf=3`, balanced class weights), PCA and subsampling; 0, 1, 2 for the GRU (results averaged); bootstrap uses 10,000 resamples of people.
- **Metric:** macro-F1 per person, then mean ± SD over people.

Because PCA is fitted on a random subsample, repeated runs of the embedding variants can differ by about 0.01 macro-F1.

## Citation

If you use this code, please cite the paper (citation will be added after publication).

## Licence

Code: MIT (see `LICENSE`). DMD is © Vicomtech and subject to its own licence.
