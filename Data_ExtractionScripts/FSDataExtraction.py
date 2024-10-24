import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

root_dir = "../EEG_data"

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
            patient_features = [condition, patient]  # Metadata columns
            for file_name in os.listdir(patient_path):
                file_path = os.path.join(patient_path, file_name)
                features = extract_features(file_path)
                patient_features.extend(features)
            all_features.append(patient_features)
columns = ['Condition', 'Patient'] + [f'Electrode_{i}_Feature_{j}' for i in range(1, 22) for j in
                                               ['Mean', 'Std', 'Skew', 'Kurt', 'Mean_First_Diff', 'Mean_Second_Diff', 'RMS']]
df = pd.DataFrame(all_features, columns=columns)
df.to_csv('../DATA/FS_eeg_data.csv', index=False)

print("Feature extraction completed and saved to 'FS_eeg_data.csv'.")
