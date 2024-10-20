import numpy as np
import pandas as pd
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

df = pd.read_csv('RAW_eeg_data.csv')

X = df.drop(['Condition'], axis=1).values
y = df['Condition'].values

y_binary = (y == 'AD').astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.3, random_state=42)

# Define the autoencoder architecture
input_dim = X_train.shape[1]  # Number of features (flattened raw EEG data)
encoding_dim = 128  # Dimension of the compressed representation

# Build the autoencoder model
input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

# Compile the model
autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer=Adam(), loss='mse')

# Train the autoencoder
autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, shuffle=True, validation_data=(X_test, X_test))
encoder = Model(inputs=input_layer, outputs=encoded)
X_train_encoded = encoder.predict(X_train)
X_test_encoded = encoder.predict(X_test)

# Save the encoded features
np.save('X_train_encoded_basic.npy', X_train_encoded)
np.save('X_test_encoded_basic.npy', X_test_encoded)

print("Basic Autoencoder: Features extracted and saved.")