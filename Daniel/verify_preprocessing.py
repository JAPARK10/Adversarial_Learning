import os
import numpy as np

OUTPUT_ROOT = r'c:\Users\PC\Desktop\AL\SaveAsTensors\SavedTensor'

def verify_output():
    if not os.path.exists(OUTPUT_ROOT):
        print(f"Error: {OUTPUT_ROOT} does not exist.")
        return

    participants = sorted([d for d in os.listdir(OUTPUT_ROOT) if os.path.isdir(os.path.join(OUTPUT_ROOT, d))])
    print(f"Number of participant folders: {len(participants)}")
    print(f"Participants: {participants}")

    total_npy = 0
    all_shapes_correct = True
    participant_gesture_counts = {}

    for p in participants:
        p_path = os.path.join(OUTPUT_ROOT, p)
        gestures = sorted([d for d in os.listdir(p_path) if os.path.isdir(os.path.join(p_path, d))])
        participant_gesture_counts[p] = len(gestures)
        
        for g in gestures:
            g_path = os.path.join(p_path, g)
            files = [f for f in os.listdir(g_path) if f.endswith('.npy')]
            total_npy += len(files)
            
            for f in files:
                f_path = os.path.join(g_path, f)
                try:
                    data = np.load(f_path)
                    if data.shape != (30, 8, 2):
                        print(f"Error: Incorrect shape {data.shape} in {f_path}")
                        all_shapes_correct = False
                except Exception as e:
                    print(f"Error loading {f_path}: {e}")
                    all_shapes_correct = False

    print(f"Gesture folders per participant: {set(participant_gesture_counts.values())}")
    print(f"Total .npy files generated: {total_npy}")
    if all_shapes_correct:
        print("All .npy files have correct shape (30, 8, 2).")
    else:
        print("Some .npy files have INCORRECT shape.")

if __name__ == "__main__":
    verify_output()
