
import pandas as pd
import copy
import numpy as np
from smoothData import gaussian, savgol, detrend
from fileSaveHDF5 import  saveto_h5_4Dmatrix_with_split
from interpData import (RDP_interpolate, interpolate_data, segment_interpolation, remove_zero_padding, 
                        split_and_pad_dataframe, pad_dataframe, pad_dataframe_new, pad_data_zeros, split_by_discontinuous_index, 
                        split_into_k_parts)
from normalizeData import (normalize, normalize_EPC, phase_standardize, phase_scale,
                           standardize_robust)
from transformData import (unwrap, phase_log_transform, phase_quantile_transform, phase_robust_scaler,
                           phase_power_transform, phase_log_transform_shift)
from augmentData import add_constant_offset, sub_constant_offset, add_gaussian_noise, add_offset_and_noise
from scipy.spatial.distance import euclidean
import os

# this gets rid of some random warning when separating data into EPC1 & EPC2
pd.options.mode.chained_assignment = None  # default='warn'

# Set display options
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_colwidth', None)  # Show full column content

# this function formats the original data exported from the ItemTest program
# made by IMPINJ. The output of this function returns a dataframe list
# which contains RSSI & phase data based on EPC and iteration for 1 gesture
def format(inputs, h5_name, labels, RSSI_val, phase_val, flags, lengths, participant_ids=None):


    ############################### FUNCTION VARIABLES ###############################
    # init empty lists
    data = []
    EPC_sep = []
    EPC_timestamps = []
    EPC_padded = []
    EPC_interp = []

    # store list variables into readable names
    h5_flag = flags[0]
    norm_flag = flags[1]
    aug_flag = flags[2]

    interp_length = lengths[0]
    threshold = lengths[1]

    #init dictionary to store EPC values
    mapping = {'A10000000000000000000000': 1,
               'A20000000000000000000000': 2,
               'A30000000000000000000000': 3,
               'A40000000000000000000000': 4,
               'A60000000000000000000000': 5,
               'A70000000000000000000000': 6,
               'A80000000000000000000000': 7,
               'A90000000000000000000000': 8}
    
    EPC_count = len(mapping)
    
    ############################### DATA PREFORMATTING ###############################
    # load csv file into dataframe and change header row (deletes everything before)
    total_files = len(inputs)
    for i, input in enumerate(inputs):
        if i % 100 == 0:
            print(f'[*] Reading file {i}/{total_files}: {input}')
        try:
            data.append(pd.read_csv(input, header=2))
        except Exception as e:
            print(f"Error reading file: {input}")
            raise e

      
    # format column names and datatypes for easier preprocessing
    data = [format_datatypes_columns(df, mapping, inputs[i]) for i, df in enumerate(data)]

    # iterate through the dataframe list to split by EPC and get timestamp column
    # must split by EPC first so we can unwrap
    for df in data:
        for j in range(1, EPC_count + 1):
            EPC_timestamps.append(np.sort(df['TimeValue'].values))
            EPC_sep.append(df[df['EPC'] == j])

    if(norm_flag == 1):   
        print(f'---------------------- TRAINING DATA --------------------')
        print(f'Preprocessing {len(data)} gestures')                       
        print(f'Tags            : {EPC_count}')
    else:
        print(f'---------------------- TESTING DATA ---------------------')
        print(f'Preprocessing {len(data)} gestures')                            
        print(f'Tags            : {EPC_count}')

    
    
    #################################### NORMALIZE DATA ####################################
    # normalize RSSI based on preferences
    if(norm_flag == 1):
        # RSSI normalization options
        EPC_sep, RSSI_val = normalize(EPC_sep, 'RSSI', norm_flag, RSSI_val)
        #EPC_sep, RSSI_val = normalize_EPC(EPC_sep, EPC_count, 'RSSI', norm_flag, RSSI_val)

        EPC_sep = unwrap(EPC_sep)

       

        # phase normalization options
        EPC_sep, phase_val = standardize_robust(EPC_sep, norm_flag, phase_val, 'PhaseAngle')

        

    else:
        # RSSI normalization options
        EPC_sep, RSSI_val = normalize(EPC_sep, 'RSSI', norm_flag, RSSI_val)
        

        EPC_sep = unwrap(EPC_sep)


        # phase normalization options
        EPC_sep, phase_val = standardize_robust(EPC_sep, norm_flag, phase_val, 'PhaseAngle')
    
    ##################################### SMOOTH DATA ######################################
    window_length = 5
    polyorder = 2
    EPC_sep = savgol(EPC_sep, 'PhaseAngle', window_length, polyorder)

    stdv = 1
    EPC_sep = gaussian(EPC_sep, 'PhaseAngle', stdv)

   

    ############################### PAD AND INTERPOLATE DATA ################################
    # pad all EPC separated dataframes with zeros during time reads of entire gesture
    for i, df in enumerate(EPC_sep):
        EPC_padded.append(pad_dataframe_new(df, EPC_timestamps[i], i, EPC_count))

    EPC_interpolated = [interpolate_dataframe(df) for df in EPC_padded]

    
    euclidean_results, closest_counts, filled_data = calculate_euclidean_distance_and_fill_main(EPC_interpolated, labels)

    
    

    output_root = r'c:\Users\jerem\Desktop\Workspace_VSCode\CoDaS\Adversarial_Learning\codebase\AdversarialLearningProject (1)\SaveAsTensors\SavedTensor'
    num_epcs = 8
    target_length = 30

    label_counters = {label: 0 for label in filled_data.keys()}

    for i in range(0, len(EPC_interpolated), num_epcs):

        label = labels[i // num_epcs]
        data_index = label_counters[label]
        label_counters[label] += 1
        
        # Get participant ID if available
        p_id = participant_ids[i // num_epcs] if participant_ids is not None else 0

        epc_group = EPC_interpolated[i:i + num_epcs]
        updated_group = []

        # ---------- APPLY IMPUTATION ----------
        for df in epc_group:
            updated_df = df.copy()

            for epc in df["EPC"].unique():
                if epc in filled_data[label][data_index]:
                    filled_values = filled_data[label][data_index][epc]
                    if filled_values is not None:
                        mask = (
                            (updated_df["EPC"] == epc) &
                            (updated_df[["RSSI", "PhaseAngle"]].sum(axis=1) == 0)
                        )
                        updated_df.loc[mask, ["RSSI", "PhaseAngle"]] = filled_values

            updated_group.append(updated_df)

        # ---------- SAVE THIS CSV AS NPY ----------
        tensor = epc_group_to_tensor(updated_group, target_length, num_epcs)

        class_dir = os.path.join(output_root, str(label))
        os.makedirs(class_dir, exist_ok=True)

        fname = f"sample_{data_index:05d}_p{p_id:02d}.npy"
        np.save(os.path.join(class_dir, fname), tensor)

        
    
    # #################################### AUGMENT EPC DATA ###################################
    # if(norm_flag == 1 & aug_flag == 1):
    #     # create augmented list and add in augmented dataframes
    #     aug_EPC = copy.deepcopy(EPC_filled)
    #     print(len(aug_EPC))
    #     for i, df in enumerate(aug_EPC):
    #         aug_EPC[i] = add_gaussian_noise(df, RSSI_std=0.015, phase_std=0.15)

    #         if (i + 1) % 10000 == 0:
    #             print(f'Augmentation on dataframe {i + 1}/{len(EPC_padded)}...')

    #     # concatenate all dataframes and labels togther
    #     EPC_filled = EPC_filled + aug_EPC
    #     labels = labels + labels

    # # use data to save as csv which will return and save unchanged data to h5
    # if h5_flag == 1:
    #     saveto_h5_4Dmatrix_with_split(EPC_filled, h5_name, labels, EPC_count, norm_flag) # 4D matrix
        

    # if (norm_flag == 1):
    #     return RSSI_val, phase_val
    
def format_datatypes_columns(df, mapping, file_name):
        print(f'Processing file: {file_name}')
    # change which columns are needed and remove spaces
        df = df.iloc[0:, [0,1,4,7]]
        df.columns = df.columns.str.strip()

        # remove A5 tag from all gestures
        df = df[df['EPC'] != 'A50000000000000000000000']

        # rename timestamp column
        df.rename(columns={'// Timestamp' : 'Timestamp'}, inplace = True)

        # parse timestamp data and create new column to have comparable time values
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        try:
            df['Timestamp'] = (df['Timestamp'] - df['Timestamp'].iloc[0]).dt.total_seconds()
        except Exception as e:
            print(f"Error processing file: {file_name}")
            raise e
        df.rename(columns={'Timestamp': 'TimeValue'}, inplace=True)

        # correctly change RSSI column to numbers
        if(df['RSSI'].dtype != 'int64'):
            # replace commas, negative signs, then change to float64s
            df['RSSI'] = df['RSSI'].str.replace(',' , '.', regex = False)
            df['RSSI'] = df['RSSI'].str.replace(r'[^\d.-]', '-', regex = True)
            df['RSSI'] = pd.to_numeric(df['RSSI'], errors='coerce')

        # replace commas and change to float64s
        df['PhaseAngle'] = df['PhaseAngle'].str.replace(',' , '.', regex = False)
        df['PhaseAngle'] = pd.to_numeric(df['PhaseAngle'])

        # replace long EPC values with integers from mapping dictionary
        df['EPC'] = df['EPC'].replace(mapping)

        # call cleaning function to remove duplicate reads
        clean(df)

        return df


def clean(df):
    # Drop duplicate rows based on the 'TimeValue' column, keeping the first occurrence
    df_cleaned = df.drop_duplicates(subset='TimeValue', keep='first')
    
    # Optionally, reset the index if needed
    df_cleaned.reset_index(drop=True, inplace=True)
    
    return df_cleaned



def calculate_euclidean_distance_and_fill_main(EPC_interp, labels, k=5):
    missing_epcs = find_missing_epcs(EPC_interp, labels)
    print("Missing EPCs per class:", missing_epcs)

    label_data = {}
    num_epcs = 8
    label_counters = {}
    closest_mapping = {1: 2, 2: 1, 3: 4, 4: 3, 5: 6, 6: 5, 7: 8, 8: 7}

    for i in range(0, len(EPC_interp), num_epcs):
        label = labels[i // num_epcs]
        if label not in label_data:
            label_data[label] = {}
            label_counters[label] = 0

        data_index = label_counters[label]
        label_counters[label] += 1
        label_data[label][data_index] = {}
        group = EPC_interp[i:i + num_epcs]
    
        for df in group:
            for epc in df['EPC'].unique():
                if epc not in label_data[label][data_index]:
                    label_data[label][data_index][epc] = []
                label_data[label][data_index][epc].append(df[df['EPC'] == epc][['RSSI', 'PhaseAngle']].values)

    euclidean_results = {}
    closest_counts = {}
    filled_data = {}
    
    for label, data_groups in label_data.items():
        euclidean_results[label] = {}
        closest_counts[label] = {}
        filled_data[label] = {}
        num_data = len(data_groups)
        
        for data_idx in range(num_data):
            missing = missing_epcs[label][data_idx]
            available_epcs = set(range(1, num_epcs + 1)) - missing
            min_distances = {}
            closest_data_points = {}
            
            for epc in available_epcs:
                distances = []
                for other_idx in range(num_data):
                    if other_idx == data_idx:
                        continue
                    if epc in data_groups[data_idx] and epc in data_groups[other_idx]:
                        data_points1 = data_groups[data_idx][epc]
                        data_points2 = data_groups[other_idx][epc]
                        if len(data_points1) > 0 and len(data_points2) > 0:
                            pairwise_distances = [euclidean(p1, p2) for p1, p2 in zip(data_points1[0], data_points2[0])]
                            mean_distance = np.mean(pairwise_distances)
                            distances.append((mean_distance, other_idx))
                
                if distances:
                    distances.sort()
                    min_distances[epc] = distances[0][0]
                    closest_data_points[epc] = distances[0][1]
            
            euclidean_results[label][data_idx] = min_distances
            closest_counts[label][data_idx] = {}
            all_closest_points = [closest_data_points[epc] for epc in closest_data_points]
            
            for point in all_closest_points:
                if point not in closest_counts[label][data_idx]:
                    closest_counts[label][data_idx][point] = 0
                closest_counts[label][data_idx][point] += 1
            
            sorted_closest = sorted(closest_counts[label][data_idx].items(), key=lambda x: x[1], reverse=True)
            closest_counts[label][data_idx] = sorted_closest[:k]
            filled_data[label][data_idx] = {}
            
            if len(missing) == 1:
                epc = list(missing)[0]
                sum_values = []
                count_values = 0
                for (closest_idx, _) in closest_counts[label][data_idx]:
                    if epc in data_groups[closest_idx]:
                        values = data_groups[closest_idx][epc][0]
                        if not np.all(values == 0):
                            sum_values.append(values)
                            count_values += 1
                if count_values > 0:
                    filled_data[label][data_idx][epc] = np.mean(sum_values, axis=0)
                elif epc in closest_mapping and closest_mapping[epc] in data_groups[data_idx]:
                    filled_data[label][data_idx][epc] = data_groups[data_idx][closest_mapping[epc]][0]
                else:
                    filled_data[label][data_idx][epc] = None
            
            elif len(missing) == 2:
                epc1, epc2 = list(missing)
                if abs(epc1 - epc2) == 1 and epc1 % 2 == 1:
                    for epc in [epc1, epc2]:
                        sum_values = []
                        count_values = 0
                        for (closest_idx, _) in closest_counts[label][data_idx]:
                            if epc in data_groups[closest_idx]:
                                values = data_groups[closest_idx][epc][0]
                                if not np.all(values == 0):
                                    sum_values.append(values)
                                    count_values += 1
                        if count_values > 0:
                            filled_data[label][data_idx][epc] = np.mean(sum_values, axis=0)
                        elif epc in closest_mapping and closest_mapping[epc] in filled_data[label][data_idx]:
                            filled_data[label][data_idx][epc] = filled_data[label][data_idx][closest_mapping[epc]]
                        elif any(epc in data_groups[i] for i in data_groups):
                            available_values = [
                                data_groups[i][epc][0] for i in data_groups 
                                if epc in data_groups[i] and not np.all(data_groups[i][epc][0] == 0)
                            ]
                            if available_values:
                                filled_data[label][data_idx][epc] = np.mean(available_values, axis=0)
                            else:
                                filled_data[label][data_idx][epc] = None
                else:
                    for epc in [epc1, epc2]:
                        sum_values = []
                        count_values = 0
                        for (closest_idx, _) in closest_counts[label][data_idx]:
                            if epc in data_groups[closest_idx]:
                                values = data_groups[closest_idx][epc][0]
                                if not np.all(values == 0):
                                    sum_values.append(values)
                                    count_values += 1
                        if count_values > 0:
                            filled_data[label][data_idx][epc] = np.mean(sum_values, axis=0)
                        elif epc in closest_mapping and closest_mapping[epc] in data_groups[data_idx]:
                            filled_data[label][data_idx][epc] = data_groups[data_idx][closest_mapping[epc]][0]
                        else:
                            filled_data[label][data_idx][epc] = None
            else:
                for epc in missing:
                    available_values = [
                        data_groups[i][epc][0] for i in data_groups 
                        if epc in data_groups[i] and not np.all(data_groups[i][epc][0] == 0)
                    ]
                    if available_values:
                        filled_data[label][data_idx][epc] = np.mean(available_values, axis=0)
                    else:
                        filled_data[label][data_idx][epc] = None
    
    return euclidean_results, closest_counts, filled_data


def find_missing_epcs(EPC_interp, labels):
    missing_epcs = {label: [] for label in set(labels)}
    num_epcs = 8
    
    for i in range(0, len(EPC_interp), num_epcs):
        label = labels[i // num_epcs]
        group = EPC_interp[i:i + num_epcs]
        missing = []
        
        for df in group:
            for epc in df['EPC'].unique():
                data = df[df['EPC'] == epc][['RSSI', 'PhaseAngle']].values
                if np.all(data == 0):  # Check if all values for this EPC are zero
                    missing.append(epc)
        missing_epcs[label].append(set(missing))
    
    return missing_epcs



def interpolate_dataframe(df):
    """Interpolates RSSI and PhaseAngle separately for each EPC, properly handling edge extrapolation."""
    df = df.copy()  # Avoid modifying the original DataFrame
    
    # Identify unique EPC values
    unique_epcs = df['EPC'].unique()
    resampled_dfs = []  # Store resampled dataframes

    for epc in unique_epcs:
        mask = df['EPC'] == epc  # Filter data for current EPC
        df_epc = df.loc[mask].copy()  # Work with a copy

        for col in ['RSSI', 'PhaseAngle']:
            if df_epc[col].eq(0).all():
                continue  # Skip if all values are zero
            
            # Replace zeros with NaN for interpolation
            df_epc[col] = df_epc[col].replace(0, np.nan)
            
            # Handle extrapolation at the beginning
            first_valid_idx = df_epc[col].first_valid_index()
            second_valid_idx = df_epc[col].index[df_epc[col].notna()][1]  # Second valid index
            
            if first_valid_idx is not None and second_valid_idx is not None:
                x1, y1 = df_epc.loc[first_valid_idx, 'TimeValue'], df_epc.loc[first_valid_idx, col]
                x2, y2 = df_epc.loc[second_valid_idx, 'TimeValue'], df_epc.loc[second_valid_idx, col]
                
                # Slope for extrapolation
                slope = (y2 - y1) / (x2 - x1)

                if col == 'RSSI':
                    # Exponential extrapolation for RSSI
                    df_epc.loc[:first_valid_idx, col] = y1 * np.exp(slope * (df_epc.loc[:first_valid_idx, 'TimeValue'] - x1))
                else:
                    # Linear extrapolation for PhaseAngle
                    df_epc.loc[:first_valid_idx, col] = y1 + slope * (df_epc.loc[:first_valid_idx, 'TimeValue'] - x1)
            
            # Handle extrapolation at the end
            last_valid_idx = df_epc[col].last_valid_index()
            second_last_valid_idx = df_epc.index[df_epc[col].notna()][-2]  # One before last valid
            
            if last_valid_idx is not None and second_last_valid_idx is not None:
                x1, y1 = df_epc.loc[second_last_valid_idx, 'TimeValue'], df_epc.loc[second_last_valid_idx, col]
                x2, y2 = df_epc.loc[last_valid_idx, 'TimeValue'], df_epc.loc[last_valid_idx, col]
                
                # Slope for extrapolation
                slope = (y2 - y1) / (x2 - x1)

                if col == 'RSSI':
                    # Exponential extrapolation for RSSI
                    df_epc.loc[last_valid_idx:, col] = y2 * np.exp(slope * (df_epc.loc[last_valid_idx:, 'TimeValue'] - x2))
                else:
                    # Linear extrapolation for PhaseAngle
                    df_epc.loc[last_valid_idx:, col] = y2 + slope * (df_epc.loc[last_valid_idx:, 'TimeValue'] - x2)

            # Apply interpolation
            if col == 'RSSI':
                df_epc[col] = exponential_interpolation(df_epc[col])  # Exponential for RSSI
            else:
                df_epc[col] = df_epc[col].interpolate(method='linear', limit_direction='both')  # Linear for PhaseAngle

            # Clip RSSI values to stay within [0,1]
            if col == 'RSSI':
                df_epc[col] = np.clip(df_epc[col], 0, 1)

        # Apply resampling to get exactly 40 samples
        df_epc_resampled = resample_dataframe(df_epc)
        resampled_dfs.append(df_epc_resampled)

    # Concatenate all resampled dataframes
    df_final = pd.concat(resampled_dfs, ignore_index=True)

    return df_final




def resample_dataframe(df, target_length=30):
    """Resamples the dataframe to have exactly 'target_length' samples per EPC, evenly distributing changes."""
    
    df = df.reset_index(drop=True)  # Reset index to ensure proper interpolation

    current_length = len(df)

    if current_length == target_length:
        return df  # No resampling needed
    
    elif current_length > target_length:
        # Downsampling: Select evenly spaced points
        indices = np.linspace(0, current_length - 1, num=target_length, dtype=int)
        df_resampled = df.iloc[indices].reset_index(drop=True)

    else:
        # Upsampling: Interpolate new points
        new_indices = np.linspace(0, current_length - 1, num=target_length)
        df_resampled = df.reindex(new_indices).reset_index(drop=True)
        
        # Interpolate all columns
        df_resampled.interpolate(method='linear', inplace=True, limit_direction='both')

    return df_resampled




def exponential_interpolation(series):
    """Performs exponential interpolation on a given pandas series (for RSSI)."""
    valid_indices = series[series.notna()].index
    if len(valid_indices) < 2:
        return series  # Not enough data for interpolation

    first_valid_idx, last_valid_idx = valid_indices[0], valid_indices[-1]
    
    # Take the log of valid values (to apply exponential interpolation)
    log_series = np.log(series)
    log_series = log_series.interpolate(method='linear', limit_direction='both')
    
    # Convert back from log space
    series_interpolated = np.exp(log_series)

    # Clip values to ensure they remain within [0,1]
    series_interpolated = np.clip(series_interpolated, 0, 1)

    return series_interpolated



def epc_group_to_tensor(epc_group, target_length=30, num_epcs=8):
    """
    epc_group: list of 8 EPC dataframes (one CSV)
    returns: np.ndarray of shape (30, 8, 2)
    """
    tensor = np.zeros((target_length, num_epcs, 2), dtype=np.float32)

    for epc_idx, df in enumerate(epc_group):
        df = df.sort_values("TimeValue")

        tensor[:, epc_idx, 0] = df["RSSI"].values[:target_length]
        tensor[:, epc_idx, 1] = df["PhaseAngle"].values[:target_length]

    return tensor


