import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

# Load the EEG data
eeg_features = pd.read_csv('../DATA/RAW_eeg_data.csv')

# Define constants
num_patients = 182
num_electrodes = 21
num_time_steps = 1024

# Define the Convolutional Autoencoder
def build_conv_autoencoder():
    input_shape = (1024, 1)  # 1024 time steps and 1 channel

    # Encoder
    encoder_input = layers.Input(shape=input_shape)
    x = layers.Conv1D(32, 3, activation='relu', padding='same')(encoder_input)
    x = layers.MaxPooling1D(2, padding='same')(x)
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2, padding='same')(x)
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    encoded = layers.MaxPooling1D(2, padding='same')(x)

    # Decoder
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(encoded)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(32, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(2)(x)
    decoded = layers.Conv1D(1, 3, activation='sigmoid', padding='same')(x)

    # Build and compile the model
    autoencoder = models.Model(encoder_input, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

# Build the autoencoder
autoencoder = build_conv_autoencoder()
autoencoder.summary()

# List to store the encoded data for each electrode
encoded_eeg_data = []

# Loop over each electrode, apply CAE, and store the encoded data
for electrode in range(num_electrodes):
    # Extract the block of data corresponding to this electrode (1024 columns per electrode)
    start_col = electrode * num_time_steps
    end_col = start_col + num_time_steps
    electrode_data = eeg_features.iloc[:, start_col:end_col]  # Shape: (182, 1024)

    # Reshape to prepare for the convolutional autoencoder (patients, timesteps, 1 channel)
    electrode_data_reshaped = electrode_data.values.reshape(num_patients, num_time_steps, 1)

    # Train the autoencoder on this electrode's data
    autoencoder.fit(electrode_data_reshaped, electrode_data_reshaped, epochs=50, batch_size=16, shuffle=True)

    # Encode the data
    encoded_electrode_data = autoencoder.predict(electrode_data_reshaped)

    # Store the encoded data (flatten the time steps)
    encoded_eeg_data.append(encoded_electrode_data.reshape(num_patients, -1))

# Combine all encoded electrode data into one DataFrame
encoded_eeg_data_array = np.concatenate(encoded_eeg_data, axis=1)  # Shape: (182, <encoded_features>)
encoded_df = pd.DataFrame(encoded_eeg_data_array)

# Save the encoded data
encoded_df.to_csv('./DATA/CAE2_eeg_data.csv', index=False)

print("Convolutional Autoencoder applied to all electrodes, data saved.")

# Visualize the original vs reconstructed data for the first electrode of the first patient
decoded_electrode_data = autoencoder.predict(electrode_data_reshaped)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(electrode_data_reshaped[0].reshape(1024))
plt.title('Original EEG Data - Electrode 1')

plt.subplot(1, 2, 2)
plt.plot(decoded_electrode_data[0].reshape(1024))
plt.title('Reconstructed EEG Data - Electrode 1')

plt.show()
