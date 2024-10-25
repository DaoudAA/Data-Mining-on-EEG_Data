# Alzheimer's Disease Detection from EEG Data

This project focuses on detecting Alzheimer's disease (AD) by analyzing EEG (Electroencephalogram) signals. Using machine learning techniques, we preprocess the EEG data, extract key features, and apply various models to classify patients as either healthy or affected by Alzheimer's. The workflow involves data organization, feature extraction, and model implementation within a well-defined directory structure.

---

## Project Structure

The project directory is organized as follows:

    -   DATA # Contains preprocessed CSV files
    -   Data_ExtractionScripts:  # Python scripts for data extraction and preprocessing
    -       BasicAutoEncodedDataExtraction.py # Script for extracting Basic Autoencoded data.
    -       CAE_DATAExtraction.py             # Script for extracting Convolutional Autoencoded data.
    -       FSDataExtraction.py               # Script for extracting data with statistical features.
    -       RawDataExtraction.py              # Script for extracting raw data.
    -   Data_Analysis_Notebooks
    -       BasicAEData_DM.ipynb              # Data mining with Basic Autoencoder data.
    -       ConvolutionalAEDataWork_DM.ipynb  # Data mining with Convolutional Autoencoder data.
    -       FFNNAEData_DM.ipynb               # Data mining with Feedforward NN Autoencoder data.
    -       RawData_DM.ipynb                  # Data mining with Raw EEG data.
    -       StatisticsFeaturesDM.ipynb        # Data mining with statistical features.
    -   EEG_data # Raw EEG data in a structured format

### Folder Descriptions

- **DATA**: This folder contains CSV files with preprocessed EEG data, organized for direct use in machine learning tasks.
  
- **Data_ExtractionScripts**: This folder contains Python scripts that extract and preprocess the EEG data from raw files. This module performs initial cleaning and transformation of the data, readying it for feature extraction and analysis.

- **EEG_data**: The main repository for raw EEG data, organized by patient condition (AD or Healthy) and state (Eyes_closed or Eyes_open). Each patient’s data is stored in a separate subfolder for streamlined processing.

- **Notebooks**: Jupyter notebooks that contain the data mining and analysis workflow, including feature extraction, training, and tuning of four different models (Decision Tree, Random Forest, SVM, and KNN). Each notebook is structured to enable reproducibility of the preprocessing steps, model training, and parameter tuning.

---

## Machine Learning Models

The following models are trained on the extracted features to predict whether a patient is healthy or affected by Alzheimer's:

1. **Decision Tree (DT)**
2. **Random Forest (RF)**
3. **Support Vector Machine (SVM)**
4. **K-Nearest Neighbors (KNN)**

Each model undergoes parameter tuning to optimize performance, and results are evaluated using metrics like accuracy, precision, and recall.

---

## Dependencies

To ensure smooth execution of the project, please install the following dependencies:

- **tensorflow**: Deep learning library for autoencoder implementations
- **pandas**: Data manipulation and analysis
- **matplotlib**: Data visualization
- **scikit-learn**: Machine learning models and evaluation
- **numpy**: Numerical computations
- **scipy**: Scientific computations
- **jupyter**: Interactive notebooks for project workflow

To install all dependencies, use:

```bash
pip install tensorflow pandas matplotlib scikit-learn numpy scipy jupyter
````
Getting Started :

    -   Data Preprocessing: Run the Python scripts in Data_ExtractionScripts to preprocess and structure the EEG data for analysis.
    -   Model Training and Evaluation: Use the notebooks in the main directory to train and tune the models on the preprocessed data.
    -   Results Visualization: Evaluate model performance and visualize results using provided functions.