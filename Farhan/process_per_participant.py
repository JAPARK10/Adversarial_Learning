"""
process_per_participant.py

Processes each participant separately through the SaveAsTensors pipeline
and saves output .npy files named:
    p01_sample_00000.npy  ...  p16_sample_XXXXX.npy

Output goes directly into RFIDDataSet/raw/gesture1/ ... gesture21/

HOW TO RUN:
    conda activate GNNPlus
    cd /Users/farhan/Downloads/projectcourse1
    python process_per_participant.py

THEN delete old processed cache and retrain:
    rm -rf /Users/farhan/Downloads/projectcourse1/ICML/GNNPlus-main/RFIDDataSet/processed
    cd /Users/farhan/Downloads/projectcourse1/ICML/GNNPlus-main
    python main.py
"""

import os
import sys
import shutil
import numpy as np

# ── CONFIGURE THESE PATHS ─────────────────────────────────────────────────────
DATASET_DIR   = '/Users/farhan/Downloads/projectcourse1/DataSet3m'
OUTPUT_DIR    = '/Users/farhan/Downloads/projectcourse1/ICML/GNNPlus-main/RFIDDataSet/raw'
SAVEASTENSORS = '/Users/farhan/Downloads/projectcourse1/SaveAsTensors'
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, SAVEASTENSORS)

from formatData import format
from fileManage import get_csv_all


def get_participants(dataset_dir):
    entries = sorted(os.listdir(dataset_dir))
    return [
        e for e in entries
        if os.path.isdir(os.path.join(dataset_dir, e))
        and e.lower() != 'combined'
    ]


def process_participant(participant_folder, participant_idx):
    prefix = f'p{participant_idx:02d}'
    participant_path = os.path.join(DATASET_DIR, participant_folder)

    print(f'\n{"="*60}')
    print(f'Processing participant {participant_idx:02d}: {participant_folder}  [{prefix}]')
    print(f'{"="*60}')

    # ── run SaveAsTensors pipeline (exactly like main.py does it) ────────
    try:
        csv_paths, labels = get_csv_all(participant_path)
    except Exception as e:
        print(f'  ERROR getting CSV files: {e}')
        return 0

    h5_name   = 'ply_data_train'
    RSSI_val  = []
    phase_val = []
    flags     = [1, 1, 0]   # h5_flag=1, norm_flag=1, aug_flag=0
    lengths   = [30, 10]    # interpolation_length=30, threshold=10

    # format() saves .npy files into cwd/SavedTensor/gesture<N>/
    # We temporarily cd into a per-participant temp dir so outputs
    # don't collide between participants
    temp_dir = os.path.join(OUTPUT_DIR, f'_temp_{prefix}')
    os.makedirs(temp_dir, exist_ok=True)
    original_cwd = os.getcwd()
    os.chdir(temp_dir)

    try:
        format(csv_paths, h5_name, labels, RSSI_val, phase_val, flags, lengths)
    except Exception as e:
        print(f'  ERROR during format(): {e}')
        os.chdir(original_cwd)
        shutil.rmtree(temp_dir, ignore_errors=True)
        return 0

    os.chdir(original_cwd)

    # ── move tensors from temp_dir/SavedTensor/ to OUTPUT_DIR ───────────
    saved_tensor_dir = '/Users/farhan/Downloads/projectcourse1/SaveAsTensors/SavedTensor'
    if not os.path.exists(saved_tensor_dir):
        print(f'  WARNING: SavedTensor folder not found in {temp_dir}')
        shutil.rmtree(temp_dir, ignore_errors=True)
        return 0

    total_saved = 0
    for gesture_folder in sorted(os.listdir(saved_tensor_dir)):
        src_gesture = os.path.join(saved_tensor_dir, gesture_folder)
        if not os.path.isdir(src_gesture):
            continue

        dst_gesture = os.path.join(OUTPUT_DIR, gesture_folder)
        os.makedirs(dst_gesture, exist_ok=True)

        npy_files = sorted([f for f in os.listdir(src_gesture)
                            if f.endswith('.npy')])
        for i, fname in enumerate(npy_files):
            src = os.path.join(src_gesture, fname)
            dst = os.path.join(dst_gesture, f'{prefix}_sample_{i:05d}.npy')
            shutil.move(src, dst)
            total_saved += 1

    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f'  Saved {total_saved} tensors for {participant_folder}')
    return total_saved


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    participants = get_participants(DATASET_DIR)
    print(f'Found {len(participants)} participants:')
    for i, p in enumerate(participants, 1):
        print(f'  p{i:02d}: {p}')

    grand_total = 0
    for idx, participant in enumerate(participants, 1):
        count = process_participant(participant, idx)
        grand_total += count

    print(f'\n{"="*60}')
    print(f'DONE. Total tensors saved: {grand_total}')
    print(f'Output: {OUTPUT_DIR}')
    print(f'\nNext steps:')
    print(f'  rm -rf {OUTPUT_DIR.replace("/raw", "/processed")}')
    print(f'  cd /Users/farhan/Downloads/projectcourse1/ICML/GNNPlus-main')
    print(f'  python main.py')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
