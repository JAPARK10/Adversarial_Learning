import torch
import os

def check_dead_air(dataset_path):
    if not os.path.exists(dataset_path):
        print("Dataset not found.")
        return
    
    data_obj, slices = torch.load(dataset_path, weights_only=False)
    # Check sample 0
    x = data_obj.x[0:8] # First sample (8 nodes)
    # x is [8, 120]
    # Features: [RSSI(30), Phase(30), RSSI_Delta(30), Phase_Delta(30)]
    
    rssi = x[:, 0:30]
    phase = x[:, 30:60]
    
    # Calculate variance over sensors for each timestep
    rssi_var = rssi.std(dim=0) # [30]
    phase_var = phase.std(dim=0) # [30]
    
    print("RSSI Std over sensors per frame:")
    print(rssi_var)
    print("\nPhase Std over sensors per frame:")
    print(phase_var)

if __name__ == "__main__":
    check_dead_air("RFIDDataSet/processed/super_geometric_data.pt")
