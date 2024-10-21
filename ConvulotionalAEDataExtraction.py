import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras import layers, models

# Load the EEG data into a DataFrame
eeg_data = pd.read_csv('RAW_eeg_data.csv')

# Separate the features (electrode data) and labels (Condition)
eeg_features = eeg_data.drop('Condition', axis=1)  # Dropping the 'Condition' column for reshaping
eeg_labels = eeg_data['Condition']  # Storing the condition separately for later use

# Convert the features DataFrame to a NumPy array
eeg_data_array = eeg_features.values

# Reshape the EEG data to (num_patients, 21, 1024, 1) to treat it as a 2D signal
eeg_data_reshaped = eeg_data_array.reshape(-1, 21, 1024, 1)

# Print shape to verify
print(eeg_data_reshaped.shape)

# Define the Convolutional Autoencoder architecture
def build_conv_autoencoder():
    input_shape = (21, 1024, 1)  # EEG input shape (21 electrodes, 1024 timestamps, 1 channel)

    # Encoder
    #This defines the input layer of the network, with a shape corresponding to the EEG data dimensions.
    encoder_input = layers.Input(shape=input_shape)
    # This is a 2D convolutional layer with 32 filters of size 3x3. It scans across the input, learning spatial
    #       features (patterns across electrodes) and temporal features (patterns across timestamps).
        #Activation: relu is the activation function, which adds non-linearity to the model.
        #Padding='same': This ensures that the output has the same height and width as the input
    #               by padding the borders appropriately.
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_input)
        #This pooling layer reduces the spatial size by taking the maximum value in every 2x2 block,
    #           effectively downsampling the input by a factor of 2 in both dimensions.
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Cropping2D(((1, 2), (0, 0)))(x)  # Crop to get back to (21, 1024, 32)
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    # Build the autoencoder model
    autoencoder = models.Model(encoder_input, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder

# Build the autoencoder
autoencoder = build_conv_autoencoder()
autoencoder.summary()

# Train the autoencoder
autoencoder.fit(eeg_data_reshaped, eeg_data_reshaped, epochs=50, batch_size=16, shuffle=True)

# Save the encoder part of the autoencoder for feature extraction
encoder = models.Model(autoencoder.input, autoencoder.layers[6].output)

# Example: Encode the EEG data
encoded_eeg_data = encoder.predict(eeg_data_reshaped)

# Save the encoded features (useful for later classification tasks)
encoded_df = pd.DataFrame(encoded_eeg_data.reshape(encoded_eeg_data.shape[0], -1))
encoded_df.to_csv('ConvolutionalAE_eeg_data.csv', index=False)

# Visualize one example of original vs reconstructed data
decoded_eeg_data = autoencoder.predict(eeg_data_reshaped)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(eeg_data_reshaped[0].reshape(21, 1024), cmap='viridis')
plt.title('Original EEG Data')

plt.subplot(1, 2, 2)
plt.imshow(decoded_eeg_data[0].reshape(21, 1024), cmap='viridis')
plt.title('Reconstructed EEG Data')

plt.show()
