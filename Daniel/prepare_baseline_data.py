import os
import shutil
from tqdm import tqdm

SOURCE_ROOT = r'c:\Users\PC\Desktop\AL\SaveAsTensors\SavedTensor'
TARGET_ROOT = r'c:\Users\PC\Desktop\AL\ICML\GNNPlus-main\RFIDDataSet'
TARGET_RAW = os.path.join(TARGET_ROOT, 'raw')

def prepare_data():
    if not os.path.exists(SOURCE_ROOT):
        print(f"Error: Source {SOURCE_ROOT} not found.")
        return

    # Create target directories
    os.makedirs(TARGET_RAW, exist_ok=True)

    participants = [d for d in os.listdir(SOURCE_ROOT) if os.path.isdir(os.path.join(SOURCE_ROOT, d))]
    print(f"Found {len(participants)} participants to merge.")

    total_copied = 0

    for p_id in participants:
        p_path = os.path.join(SOURCE_ROOT, p_id)
        gestures = [d for d in os.listdir(p_path) if os.path.isdir(os.path.join(p_path, d))]
        
        for g_label in gestures:
            g_source_path = os.path.join(p_path, g_label)
            g_target_path = os.path.join(TARGET_RAW, g_label)
            os.makedirs(g_target_path, exist_ok=True)
            
            files = [f for f in os.listdir(g_source_path) if f.endswith('.npy')]
            
            for f in files:
                source_file = os.path.join(g_source_path, f)
                # target filename includes participant ID to avoid collisions and keep track
                target_file = os.path.join(g_target_path, f"{p_id}_{f}")
                
                # Using copy2 to preserve metadata, or copy if you prefer
                shutil.copy2(source_file, target_file)
                total_copied += 1

    print(f"Successfully merged {total_copied} files into {TARGET_RAW}")

if __name__ == "__main__":
    prepare_data()
