# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Step 1: Load the data
file_path = r'C:\Users\LENOVO\Documents\3AINFO\data_mining\Data-Mining-on-EEG_Data\FS_eeg_data.csv'
data = pd.read_csv(file_path)

# Step 2: Data Preprocessing
# Encoding categorical data (Condition: AD = 0, Healthy = 1)
data['Condition'] = data['Condition'].map({'AD': 0, 'Healthy': 1})

# Selecting the relevant features (X) and target variable (y)
features = ['Mean', 'Std', 'Skew', 'Kurt', 'Mean_First_Diff', 'Mean_Second_Diff', 'RMS']
X = data[features]
y = data['Condition']

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = dt_model.predict(X_test)

# Step 6: Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
