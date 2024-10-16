# This is a sample Python script.

# Press Ctrl+F5 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

root_dir = "EEG_data"


def extract_features(file_path):
    data = np.loadtxt(file_path)

    mean_val = np.mean(data)
    std_val = np.std(data)
    skewness = skew(data)
    kurt = kurtosis(data)

    return [mean_val, std_val, skewness, kurt]


all_features = []
for condition in ['AD', 'Healthy']:
    for state in ['Eyes_closed', 'Eyes_open']:
        condition_path = os.path.join(root_dir, condition, state)

        for patient in os.listdir(condition_path):
            patient_path = os.path.join(condition_path, patient)

            patient_features = [condition, state, patient]  # Metadata columns

            for file_name in os.listdir(patient_path):
                file_path = os.path.join(patient_path, file_name)
                features = extract_features(file_path)
                patient_features.extend(features)

            all_features.append(patient_features)

columns = ['Condition', 'State', 'Patient'] + [f'Electrode_{i}_Feature_{j}' for i in range(1, 22) for j in
                                               ['Mean', 'Std', 'Skew', 'Kurt']]
df = pd.DataFrame(all_features, columns=columns)

df.to_csv('eeg_preprocessed_data.csv', index=False)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
