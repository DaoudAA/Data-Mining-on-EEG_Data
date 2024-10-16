import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

root_dir = "EEG_data"

def extract_features(file_path):
    data = np.loadtxt(file_path)

    # Skip if the file is empty
    if data.size == 0:
        print(f"Skipping empty file: {file_path}")
        return None

    # Statistical features
    mean_val = np.mean(data)
    std_val = np.std(data)
    skewness = skew(data)
    kurt = kurtosis(data)

    # Additional statistical features
    mean_first_diff = np.mean(np.abs(np.diff(data)))
    mean_second_diff = np.mean(np.abs(np.diff(data, n=2)))
    rms_val = np.sqrt(np.mean(data ** 2))

    return [mean_val, std_val, skewness, kurt, mean_first_diff, mean_second_diff, rms_val]

all_features = []

for condition in ['AD', 'Healthy']:
    for state in ['Eyes_closed', 'Eyes_open']:
        condition_path = os.path.join(root_dir, condition, state)

        for patient in os.listdir(condition_path):
            patient_path = os.path.join(condition_path, patient)

            for file_name in os.listdir(patient_path):
                file_path = os.path.join(patient_path, file_name)

                #print(f"Processing file: {file_path}")

                features = extract_features(file_path)
                if features is None:
                    continue

                electrode_name = file_name.split('.')[0]
                all_features.append([electrode_name, condition] + features)

# Define the columns for the DataFrame
columns = ['Electrode', 'Condition', 'Mean', 'Std', 'Skew', 'Kurt', 'Mean_First_Diff', 'Mean_Second_Diff', 'RMS']

# Create the DataFrame with the required structure
df = pd.DataFrame(all_features, columns=columns)
df.to_csv('FS_eeg_data.csv', index=False)

print("Feature extraction completed and saved to 'FS_eeg_data.csv'.")
