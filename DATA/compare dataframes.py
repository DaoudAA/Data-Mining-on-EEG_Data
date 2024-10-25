import pandas as pd
import os

# Define the CSV file names
csv_files = ['ConvolutionalAE_eeg_data.csv','FFNNAE_eeg_data.csv','FS_eeg_data.csv', 'RAW_eeg_data.csv']

# Initialize a dictionary to hold the file statistics
file_stats = {}

# Iterate through the CSV files and gather statistics
for csv_file in csv_files:
    if os.path.exists(csv_file):  # Check if the file exists
        # Load the CSV file
        df = pd.read_csv(csv_file)

        # Gather statistics
        file_stats[csv_file] = {
            'Size (bytes)': os.path.getsize(csv_file),
            'Number of Rows': df.shape[0],
            'Number of Columns': df.shape[1]
        }
    else:
        print(f"File not found: {csv_file}")

# Print the statistics for comparison
for file, stats in file_stats.items():
    print(f"\nStatistics for {file}:")
    print(f"Size (bytes): {stats['Size (bytes)']}")
    print(f"Number of Rows: {stats['Number of Rows']}")
    print(f"Number of Columns: {stats['Number of Columns']}")
