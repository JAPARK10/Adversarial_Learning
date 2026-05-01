import os
import sys
import torch

# Add current directory to path to find GNNPlus
sys.path.append(os.getcwd())

from GNNPlus.loader.dataset.rfid_dataset import RFIDDataset

def main():
    # Paths for the new Pod structure
    root_dir = "/root/Adversarial_Learning/Jeremias/codebase/AdversarialLearningProject/SavedTensor"
    
    print(f"Starting dataset processing in: {root_dir}")
    if not os.path.exists(root_dir):
        print(f"ERROR: Directory {root_dir} not found. Please ensure the SavedTensor folder is copied to the Pod.")
        return

    # Initializing the dataset triggers the 'process' method automatically
    # This will scan SavedTensor/raw and create SavedTensor/processed/geometric_data_processed.pt
    dataset = RFIDDataset(root=root_dir, name="rfid")
    
    print("\nSUCCESS!")
    print(f"Total graphs processed: {len(dataset)}")
    processed_file = os.path.join(root_dir, "processed", "geometric_data_processed.pt")
    if os.path.exists(processed_file):
        size_mb = os.path.getsize(processed_file) / (1024 * 1024)
        print(f"Processed file created: {processed_file} ({size_mb:.2f} MB)")
    else:
        print("ERROR: Processed file was not found after execution.")

if __name__ == "__main__":
    main()
