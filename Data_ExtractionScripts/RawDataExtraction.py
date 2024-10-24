import os
import pandas as pd
import numpy as np


def read_eeg_data_flattened(patient_folder, condition, max_timestamps=1024):
    all_data = []
    electrode_names = []

    for file in sorted(os.listdir(patient_folder)):
        if file.endswith('.txt'):
            electrode_name = file.split('.')[0]
            electrode_names.append(electrode_name)
            file_path = os.path.join(patient_folder, file)

            # Check if the file is empty
            if os.path.getsize(file_path) == 0:
                print(f"Skipping empty file: {file_path}")
                continue

            try:
                with open(file_path, 'r') as f:
                    data = f.read().splitlines()

                # Convert data to floats
                data = [float(value) for value in data]

                # Pad data with NaNs if needed
                padded_data = data + [np.nan] * (max_timestamps - len(data))
                all_data.append(padded_data[:max_timestamps])
            except ValueError:
                print(f"Skipping file due to invalid data: {file_path}")
                continue

    # Flatten the list of data
    flattened_data = [item for sublist in all_data for item in sublist]
    flattened_data.append(condition)

    return flattened_data, electrode_names


def aggregate_patient_data_flattened(root_dir, max_timestamps=1024):
    aggregated_data = []
    column_names_set = False
    column_names = []

    for condition in ['AD', 'Healthy']:
        condition_folder = os.path.join(root_dir, condition)

        for state in ['Eyes_closed', 'Eyes_open']:
            state_folder = os.path.join(condition_folder, state)

            if not os.path.exists(state_folder):
                continue

            for patient in os.listdir(state_folder):
                print(f"Processing patient: {patient}")
                patient_folder = os.path.join(state_folder, patient)

                if os.path.isdir(patient_folder):
                    patient_data, electrode_names = read_eeg_data_flattened(patient_folder, condition, max_timestamps)
                    aggregated_data.append(patient_data)

                    if not column_names_set:
                        for electrode in electrode_names:
                            column_names += [f'{electrode}_T{t}' for t in range(1, max_timestamps + 1)]
                        column_names.append('Condition')
                        column_names_set = True

    eeg_df = pd.DataFrame(aggregated_data, columns=column_names)

    return eeg_df


# Set your root directory for EEG data
root_dir = '../EEG_data'
eeg_df = aggregate_patient_data_flattened(root_dir)
print(eeg_df)

# Save to CSV
output_file = '../DATA/RAW_eeg_data.csv'
eeg_df.to_csv(output_file, index=False)
print(f"Data exported to {output_file}")
