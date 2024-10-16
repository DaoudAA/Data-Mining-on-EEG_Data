import os
import pandas as pd


# Function to read electrode data for a patient and convert it to a DataFrame
def read_eeg_data(patient_folder, condition):
    all_data = []

    # Loop through all electrode files for the patient
    for file in os.listdir(patient_folder):
        if file.endswith('.txt'):
            electrode_name = file.split('.')[0]
            file_path = os.path.join(patient_folder, file)

            # Read the entire EEG data from the file as a list of timestamp values
            with open(file_path, 'r') as f:
                data = f.read().splitlines()  # All records as a single list of strings

            # Skip if the file is empty
            if not data:
                print(f"Skipping empty file: {file_path}")
                continue

            # Convert all data into floats (or handle any conversion issues as needed)
            try:
                data = [float(value) for value in data]
            except ValueError:
                print(f"Skipping file due to invalid data: {file_path}")
                continue

            # Append the data in the desired format: [Electrode, Timestamp values..., Condition]
            row = [electrode_name] + data + [condition]
            all_data.append(row)

    return all_data  # List of rows


# Traverse the directory structure and aggregate data
def aggregate_patient_data(root_dir):
    aggregated_data = []
    max_timestamps = 0  # Track the maximum number of timestamps

    # Loop through AD and Healthy folders
    for condition in ['AD', 'Healthy']:
        condition_folder = os.path.join(root_dir, condition)

        for state in ['Eyes_closed', 'Eyes_open']:
            state_folder = os.path.join(condition_folder, state)

            if not os.path.exists(state_folder):
                continue

            for patient in os.listdir(state_folder):
                print(patient)
                patient_folder = os.path.join(state_folder, patient)

                if os.path.isdir(patient_folder):
                    patient_data = read_eeg_data(patient_folder, condition)
                    aggregated_data.extend(patient_data)

                    # Check if this patient's data has more timestamps than previously recorded
                    for row in patient_data:
                        max_timestamps = max(max_timestamps, len(row) - 2)  # Exclude 'Electrode' and 'Condition'

    eeg_df = pd.DataFrame(aggregated_data)

    eeg_df.columns = ['Electrode'] + [f'T{i}' for i in range(1, len(eeg_df.columns) - 1)] + ['Condition']
    return eeg_df


# Example usage
root_dir = 'EEG_data'
eeg_df = aggregate_patient_data(root_dir)

output_file = 'RAW_eeg_data.csv'
eeg_df.to_csv(output_file, index=False)

print(f"Data exported to {output_file}")
