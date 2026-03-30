import numpy as np
import os
import h5py
from sklearn.model_selection import train_test_split

def one_hot_encode(epc, num_classes):
    one_hot = np.zeros(num_classes)
    one_hot[epc - 1] = 1  # Assuming EPC is 1-indexed (1 to num_classes)
    return one_hot

# this function is to take a list of all the dataframes and create an HDF5 file. The output is:
# 2 datasets - Data and Label
# Data is a 4D matrix with (gestures, datapoints, EPC, and the 2 columns of RSSI and Phase data)
def saveto_h5_4Dmatrix(EPC_sep, h5_name, labels, EPC_count, norm_flag):
    # number of gestures, datapoints, & EPC count
    num_gestures = int(len(EPC_sep) / EPC_count)
    num_datapoints = EPC_sep[0].shape[0]
    num_EPC = EPC_count

    # placeholder for the combined data array
    data = np.zeros((num_gestures, num_datapoints, num_EPC, 2))

    # iterate over each pair of DataFrames and populate the data array
    for i in range(num_gestures):
        for j in range(num_EPC):
            # get the DataFrame for the current tag
            df = EPC_sep[i * num_EPC + j]

            # store dataframe data in 4D matrix
            data[i, :, j, 0] = df['RSSI']
            data[i, :, j, 1] = df['PhaseAngle']
            
    # Create the HDF5 file and save the datasets
    with h5py.File(f'HDF5_formatted/{h5_name}.h5', 'w') as f:
        f.create_dataset('data', data = data)

        # define string data type and create dataset
        dt = h5py.string_dtype(encoding = 'utf-8')  
        f.create_dataset('label', data = np.array(labels, dtype = dt))

    if(norm_flag == 1):
        print('---------------- TRAINING HDF5 FILE SAVED ---------------\n')
    else:
        print('---------------- TESTING HDF5 FILE SAVED ----------------\n')



####Train-Test-Split######

def saveto_h5_4Dmatrix_with_split(EPC_sep, h5_name, labels, EPC_count, norm_flag, test_size=0.02, random_state=42):
    # Ensure the output directory exists
    output_dir = r'\\home.org.aalto.fi\golipos1\data\Desktop\P\RFIDSkeletonGestureRecognition\TheStateofArt\TeslaWithinUserVersion2DataImputation\HDF5_formatted'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # number of gestures, datapoints, & EPC count
    num_gestures = int(len(EPC_sep) / EPC_count)
    num_datapoints = EPC_sep[0].shape[0]
    num_EPC = EPC_count

    # placeholder for the combined data array
    data = np.zeros((num_gestures, num_datapoints, num_EPC, 2))

    # iterate over each pair of DataFrames and populate the data array
    for i in range(num_gestures):
        for j in range(num_EPC):
            # get the DataFrame for the current tag
            df = EPC_sep[i * num_EPC + j]

            # store dataframe data in 4D matrix
            data[i, :, j, 0] = df['RSSI']
            data[i, :, j, 1] = df['PhaseAngle']

    # Split data and labels into training and testing sets
    data_train, data_test, labels_train, labels_test = train_test_split(
        data, labels, test_size=test_size, random_state=random_state
    )

    # Save the training data to an HDF5 file
    with h5py.File(f'{output_dir}/{h5_name}_train.h5', 'w') as f_train:
        f_train.create_dataset('data', data=data_train)
        dt = h5py.string_dtype(encoding='utf-8')
        f_train.create_dataset('label', data=np.array(labels_train, dtype=dt))

    # Save the testing data to a separate HDF5 file
    with h5py.File(f'{output_dir}/{h5_name}_test.h5', 'w') as f_test:
        f_test.create_dataset('data', data=data_test)
        dt = h5py.string_dtype(encoding='utf-8')
        f_test.create_dataset('label', data=np.array(labels_test, dtype=dt))

    if norm_flag == 1:
        print('---------------- TRAINING HDF5 FILE SAVED ---------------\n')
        print('---------------- TESTING HDF5 FILE SAVED ----------------\n')
    else:
        print('---------------- TRAINING HDF5 FILE SAVED ---------------\n')
        print('---------------- TESTING HDF5 FILE SAVED ----------------\n')






def saveto_h5_3Dmatrix(EPC_sep, h5_name, labels, EPC_count, RSSI_flag):
    # number of gestures, datapoints, & EPC count
    num_gestures = int(len(EPC_sep) / EPC_count)
    num_datapoints = EPC_sep[0].shape[0]  # Should be 20 after interpolation
    num_EPC = EPC_count
    num_features = 10  # 8 for one-hot encoded EPC + 2 for RSSI and Phase

    # Placeholder for the combined data array
    data = np.zeros((num_gestures, num_datapoints, num_EPC * num_features))

    for i in range(num_gestures):
        for j in range(num_EPC):
            df = EPC_sep[i * num_EPC + j]

            # Extract the one-hot vector for the current EPC
            one_hot = one_hot_encode(j + 1, num_EPC)
            
            for k in range(num_datapoints):
                # Extract RSSI and Phase for the current datapoint
                rssi = df['RSSI'].iloc[k]
                phase = df['PhaseAngle'].iloc[k]
                
                # Combine one-hot vector, RSSI, and Phase into a single vector
                combined_data = np.concatenate([one_hot, [rssi, phase]])
                
                # Place the combined vector into the data array
                data[i, k, j * num_features: (j+1) * num_features] = combined_data

# Now save the 'data' array to an HDF5 file as before

    # Create the HDF5 file and save the datasets
    with h5py.File(f'HDF5_formatted/{h5_name}.h5', 'w') as f:
        f.create_dataset('data', data=data)

        # Define string data type and create dataset for labels
        dt = h5py.string_dtype(encoding='utf-8')
        f.create_dataset('label', data=np.array(labels, dtype=dt))

    if RSSI_flag == 1:
        print('---------------- TRAINING HDF5 FILE SAVED ---------------\n')
    else:
        print('---------------- TESTING HDF5 FILE SAVED ----------------\n')