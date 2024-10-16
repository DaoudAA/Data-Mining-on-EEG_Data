import os
import pandas as pd


# Function to read electrode data for a patient
def read_eeg_data(patient_folder):
    electrode_data = {}
    for file in os.listdir(patient_folder):
        if file.endswith('.txt'):
            electrode_name = file.split('.')[0]
            file_path = os.path.join(patient_folder, file)
            with open(file_path, 'r') as f:
                data = f.read().splitlines()  # Assuming the file contains lines of EEG data
                electrode_data[electrode_name] = [float(d) for d in data]
    return pd.DataFrame(electrode_data)


# Traverse the directory structure and aggregate data
def aggregate_patient_data(root_dir):
    patient_data = []
    condition_data = []

    # Loop through AD and Healthy folders
    for condition in ['AD', 'Healthy']:
        condition_folder = os.path.join(root_dir, condition)

        for state in ['Eyes_closed', 'Eyes_open']:
            state_folder = os.path.join(condition_folder, state)

            if not os.path.exists(state_folder):
                continue  # Skip if the folder does not exist

            for patient in os.listdir(state_folder):
                patient_folder = os.path.join(state_folder, patient)
                if os.path.isdir(patient_folder):
                    patient_df = read_eeg_data(patient_folder)

                    # Save patient data with state included in the ID
                    patient_id = f"{patient}_{state}"
                    patient_data.append((patient_id, patient_df))

                    # Create summary info (ID and condition)
                    condition_data.append({
                        'Patient_ID': patient_id,
                        'Condition': condition
                    })

    # Create the summary DataFrame (ID and condition)
    condition_df = pd.DataFrame(condition_data)

    return patient_data, condition_df


# Example usage
root_dir = 'EEG_data'
patient_data, condition_df = aggregate_patient_data(root_dir)

# Now you can access each patient's DataFrame and the summary DataFrame
for patient, df in patient_data:
    print(f"Patient: {patient}")
    print(df.head())  # Show the first few rows of the DataFrame for each patient

print(condition_df.head())


def export_patient_data(patient_data, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Export each patient's DataFrame to a CSV file
    for patient, df in patient_data:
        output_path = os.path.join(output_folder, f"{patient}_EEG.csv")
        df.to_csv(output_path, index=False)  # Export each patient's EEG data


# Function to export the condition summary to a CSV file
def export_condition_summary(condition_df, output_folder):
    output_path = os.path.join(output_folder, "patient_conditions.csv")
    condition_df.to_csv(output_path, index=False)  # Export the condition DataFrame


# Example usage
output_folder = 'exported_eeg_data'
export_patient_data(patient_data, output_folder)  # Export each patient's EEG data
export_condition_summary(condition_df, output_folder)  # Export condition summary
