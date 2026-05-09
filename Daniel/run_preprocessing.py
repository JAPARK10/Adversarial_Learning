"""
run_preprocessing.py
-------------------
Processes all participants from DataSet3m/ and saves .npy tensors to:
    SavedTensor/<participant_id>/<gesture_label>/sample_XXXXX.npy

Each participant is processed as one independent batch.
No data leaks between participants.

Usage:
    cd c:/Users/PC/Desktop/AL
    python run_preprocessing.py
"""

import os
import sys

# ─────────────────────────────────────────────
#  PATHS — edit these if needed
# ─────────────────────────────────────────────
DATASET_ROOT   = r'c:\Users\PC\Desktop\AL\DataSet3m'
SAVEDTENSOR_ROOT = r'c:\Users\PC\Desktop\AL\SaveAsTensors\SavedTensor'
SAVETENSORS_DIR  = r'c:\Users\PC\Desktop\AL\SaveAsTensors'   # must be on sys.path
# ─────────────────────────────────────────────

# ── make SaveAsTensors importable ──────────────────────────────────────────────
if SAVETENSORS_DIR not in sys.path:
    sys.path.insert(0, SAVETENSORS_DIR)

import formatData
from fileManage import get_csv_all

# ── discover participants ──────────────────────────────────────────────────────
all_items = sorted(os.listdir(DATASET_ROOT))
participants = [
    d for d in all_items
    if os.path.isdir(os.path.join(DATASET_ROOT, d))
]

print(f"Found {len(participants)} participant folders:")
for p in participants:
    print(f"  {p}")
print()

# ── process each participant independently ────────────────────────────────────
for participant_folder in participants:

    participant_dir = os.path.join(DATASET_ROOT, participant_folder)

    # Derive a clean ID: "DE 3m" → "DE"
    participant_id = participant_folder.replace('3m', '').strip()

    print(f"\n{'='*65}")
    print(f"  Participant: {participant_id}  (folder: '{participant_folder}')")
    print(f"{'='*65}")

    # ── collect all CSVs for this participant ─────────────────────────────
    # get_csv_all() expects:  root/<gesture_N_folder>/*.csv
    # participant_dir IS that root — correct depth.
    csv_paths, labels = get_csv_all(participant_dir)

    n_gestures = len(set(labels))
    print(f"  CSVs found  : {len(csv_paths)}")
    print(f"  Unique labels: {n_gestures}  -> {sorted(set(labels), key=lambda x: int(x.replace('gesture','')) if x else 0)}")

    if len(csv_paths) == 0:
        print(f"  [!] No CSVs found - skipping.")
        continue

    # ── set per-participant output root ───────────────────────────────────
    participant_output = os.path.join(SAVEDTENSOR_ROOT, participant_id)
    os.makedirs(participant_output, exist_ok=True)

    # Inject the output path into formatData before calling format().
    # formatData.py L142 reads this variable instead of its hardcoded string.
    formatData.OUTPUT_ROOT = participant_output

    # ── call format() with all of this participant's data ─────────────────
    # norm_flag=1  -> compute normalization stats from this participant's batch
    # aug_flag=1   -> augmentation is commented out in formatData, so no effect
    h5_name   = f'participant_{participant_id}'
    RSSI_val  = []          # fresh per participant
    phase_val = []          # fresh per participant
    flags     = [1, 1, 1]  # h5_flag, norm_flag, aug_flag
    lengths   = [30, 10]   # interpolation_length=30, threshold=10

    try:
        formatData.format(
            csv_paths,
            h5_name,
            labels,
            RSSI_val,
            phase_val,
            flags,
            lengths
        )
        print(f"  [OK] {participant_id} done.")
    except Exception as e:
        print(f"  [ERR] ERROR processing {participant_id}: {e}")
        import traceback
        traceback.print_exc()
        print("  Continuing with next participant...")
        continue

# ── final report ──────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("  Preprocessing complete. Verifying output...")
print(f"{'='*65}")

total_npy = 0
for pid in os.listdir(SAVEDTENSOR_ROOT):
    pid_path = os.path.join(SAVEDTENSOR_ROOT, pid)
    if not os.path.isdir(pid_path):
        continue
    pid_count = 0
    gestures = sorted(os.listdir(pid_path))
    for g in gestures:
        g_path = os.path.join(pid_path, g)
        if os.path.isdir(g_path):
            n = len([f for f in os.listdir(g_path) if f.endswith('.npy')])
            pid_count += n
    print(f"  {pid:6s}: {pid_count:4d} .npy files  ({len(gestures)} gesture folders)")
    total_npy += pid_count

print(f"\n  Total .npy files saved: {total_npy}")
print(f"  Expected (approx):      {16 * 21 * 18} (16p × 21g × ~18 trials avg)")
print()

# ── quick shape check ─────────────────────────────────────────────────────────
import numpy as np
import glob

sample_files = glob.glob(os.path.join(SAVEDTENSOR_ROOT, '*', '*', '*.npy'))
if sample_files:
    sample = np.load(sample_files[0])
    print(f"  Shape check: {os.path.relpath(sample_files[0], SAVEDTENSOR_ROOT)}")
    print(f"  Array shape : {sample.shape}   <- expected (30, 8, 2)")
    assert sample.shape == (30, 8, 2), f"Unexpected shape: {sample.shape}"
    print("  [OK] Shape correct.")
else:
    print(" [WARNING] No .npy files found to verify shape.")
