import os
import pandas as pd
import numpy as np
from tqdm import tqdm

def process_csv_to_npy(csv_path, npy_path):
    pd.set_option('display.float_format', lambda x: '%.6f' % x)
    # Read the CSV file, skipping the first 3 comment lines
    try:
        df = pd.read_csv(csv_path, skiprows=3, header=None)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return False
        
    # Standard 8 EPCs we expect based on RFID setups 
    # Let's extract them from the dataset dynamically or use the most frequent 8
    epcs = df[1].unique()
    if len(epcs) < 8:
        print(f"Warning: Less than 8 EPCs found in {csv_path} (Found {len(epcs)})")
        
    # We only take the top 8 most frequent EPCs to match the 8 expected nodes
    top_epcs = df[1].value_counts().head(8).index.tolist()
    
    # We need to construct an array of shape (30, 8, 2)
    # 30 time steps, 8 tags (EPCs), 2 features per tag (PhaseAngle, RSSI)
    
    # Let's sort the DataFrame by Timestamp (column 0)
    df = df.sort_values(by=0)
    
    # Initialize our target array with zeros
    # Shape: (30 time steps, 8 tags, 2 features)
    result_array = np.zeros((30, 8, 2))
    
    # Clean up the numerical columns that use comma instead of dot for decimals
    def clean_number(x):
        if isinstance(x, str):
            return float(x.replace('"', '').replace(',', '.'))
        return float(x)
        
    df[4] = df[4].apply(clean_number) # RSSI
    df[7] = df[7].apply(clean_number) # PhaseAngle
    
    # Group by EPC to get time series for each tag
    for tag_idx, epc in enumerate(top_epcs):
        # Filter data for this tag
        tag_data = df[df[1] == epc]
        
        # Take up to 30 earliest readings (or sample them if needed)
        # If there are fewer than 30, it will pad with the last value or zeros
        readings = min(30, len(tag_data))
        
        # Feature 1: PhaseAngle (col 7)
        # Feature 2: RSSI (col 4)
        if readings > 0:
            result_array[:readings, tag_idx, 0] = tag_data[7].values[:readings]
            result_array[:readings, tag_idx, 1] = tag_data[4].values[:readings]
            
            # Pad the rest with the last known value if we have less than 30 readings
            if readings < 30:
                for r in range(readings, 30):
                    result_array[r, tag_idx, 0] = result_array[readings-1, tag_idx, 0]
                    result_array[r, tag_idx, 1] = result_array[readings-1, tag_idx, 1]
                    
    # Save to npy
    np.save(npy_path, result_array)
    return True

def main():
    source_dir = r"c:\Users\PC\Desktop\AdversarialLearningProject 1\DataSet3m"
    target_dir = r"C:\Users\PC\Desktop\AdversarialLearningProject 1\ICML\GNNPlus-main\RFIDDataSet\DataSetLOPO\raw"
    
    # Create target dir if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    subjects = [s for s in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, s))]
    print(f"Found {len(subjects)} subjects in DataSet3m")
    
    # Dict to keep track of gesture counts across all subjects
    gesture_counts = {}
    
    for subject in subjects:
        subject_path = os.path.join(source_dir, subject)
        gestures = [g for g in os.listdir(subject_path) if os.path.isdir(os.path.join(subject_path, g))]
        
        for gesture in tqdm(gestures, desc=f"Processing {subject}"):
            gesture_path = os.path.join(subject_path, gesture)
            
            # Determine the generic gesture category name (e.g., '12 Two-hand push' -> 'gesture12')
            gesture_id = gesture.split(' ')[0]
            target_gesture_dir_name = f"gesture{gesture_id}"
            
            target_gesture_path = os.path.join(target_dir, target_gesture_dir_name)
            os.makedirs(target_gesture_path, exist_ok=True)
            
            # Initialize counter for this gesture if not exists
            if target_gesture_dir_name not in gesture_counts:
                gesture_counts[target_gesture_dir_name] = 0
                
            # Process all csv files for this gesture & subject combinations
            csv_files = [f for f in os.listdir(gesture_path) if f.endswith('.csv')]
            
            for csv_file in csv_files:
                csv_path = os.path.join(gesture_path, csv_file)
                
                # Naming convention: subj_{subject}_gest_{target_gesture_dir_name}_idx_{sample_idx:05d}.npy
                # We use the subject and gesture info to ensure unique filenames and preserve metadata
                sample_idx = gesture_counts[target_gesture_dir_name]
                npy_filename = f"subj_{subject}_gest_{target_gesture_dir_name}_idx_{sample_idx:05d}.npy"
                npy_path = os.path.join(target_gesture_path, npy_filename)
                
                # Only process if it doesn't already exist from the sample data
                if not os.path.exists(npy_path):
                    success = process_csv_to_npy(csv_path, npy_path)
                    if success:
                        gesture_counts[target_gesture_dir_name] += 1
                else:
                    # File exists, just increment the counter to avoid overwriting original sample data
                    gesture_counts[target_gesture_dir_name] += 1

    print("\nExtraction Complete! Target dataset stats:")
    for gesture, count in sorted(gesture_counts.items(), key=lambda x: int(x[0].replace('gesture', ''))):
        print(f"  {gesture}: {count} samples")

if __name__ == "__main__":
    main()
