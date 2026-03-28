# import functions
from formatData import format
from fileManage import get_csv_all, clear_folder, clear_directory, get_csv_filenames
import sys
import numpy as np
np.random.seed(42)

# this function is to reformat and save a .h5 file for set of gestures
def main(combined_directory):
    print(f"[*] Starting preprocessing for: {combined_directory}")

    ############################ SET FLAGS ##########################
    h5_flag = 1
    norm_flag = 1
    aug_flag = 1
    flags = [h5_flag, norm_flag, aug_flag]

    interpolation_length = 30
    threshold = 10
    lengths = [interpolation_length, threshold]

    
    csv_path_combined, comb_labels, participant_ids = get_csv_all(combined_directory)
    print(f"[*] Found {len(csv_path_combined)} CSV files.")

    # define h5 file name and data normalization values
    h5_name = 'ply_data_train'
    RSSI_val = []
    phase_val = []
    phase_val_tx = []

    # format data
    format(csv_path_combined, h5_name, comb_labels, RSSI_val, phase_val, flags, lengths, participant_ids=participant_ids)

    

if __name__ == '__main__':
    
    combined_directory = r'c:\Users\jerem\Desktop\Workspace_VSCode\CoDaS\Adversarial_Learning\codebase\AdversarialLearningProject (1)\DataSet3m'

    main(combined_directory)

