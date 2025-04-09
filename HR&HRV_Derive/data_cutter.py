import os
import pandas as pd

def cut_csv_file(file_path, output_path, n_rows):
    """
    Reads a CSV file, trims it to n_rows (if possible), and writes the trimmed data to output_path.
    
    Parameters:
        file_path (str): Path to the input CSV file.
        output_path (str): Path to the output CSV file.
        n_rows (int): Number of rows to keep.
    """
    try:
        df = pd.read_csv(file_path)
        if len(df) >= n_rows:
            df_cut = df.iloc[:n_rows, :]
        else:
            print(f"Warning: {file_path} has only {len(df)} rows (less than {n_rows}). Saving original data.")
            df_cut = df
        df_cut.to_csv(output_path, index=False)
        print(f"Saved trimmed file to: {output_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def process_folder(input_folder, output_folder, n_rows):
    """
    Process all CSV files in a folder, trimming each file to n_rows rows.
    
    Parameters:
        input_folder (str): Path to the folder containing CSV files.
        output_folder (str): Path to the folder where trimmed files will be saved.
        n_rows (int): Number of rows to keep in each file.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Process each file in the folder
    for file in os.listdir(input_folder):
        if file.lower().endswith('.csv'):
            input_file_path = os.path.join(input_folder, file)
            output_file_path = os.path.join(output_folder, file)
            cut_csv_file(input_file_path, output_file_path, n_rows)

def check_folder_csv_length(folder_path, expected_rows):
    """
    Check the length (number of rows) of the CSV file in the given folder.
    Assumes that there is only one CSV file in the folder.
    
    Parameters:
        folder_path (str): Path to the folder containing one CSV file.
        expected_rows (int): Expected number of rows after cutting.
    """
    csv_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.csv')]
    if not csv_files:
        print(f"No CSV files found in {folder_path}.")
        return
    file_path = os.path.join(folder_path, csv_files[0])
    try:
        df = pd.read_csv(file_path)
        actual_rows = len(df)
        if actual_rows == expected_rows:
            print(f"{file_path}: OK (Rows: {actual_rows})")
        else:
            print(f"{file_path}: Mismatch! Expected {expected_rows} rows but got {actual_rows}.")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
# Example usage:
# Suppose you have six folders (e.g., "subject1", "subject2", ... "subject6")
# stored under an "input_data" directory and you want to save the trimmed files
# in a corresponding "output_data" directory.
header_path = 'collected_data/PPG'
input_base = os.path.join(header_path)
output_base = os.path.join(header_path)
folder_names = ["yuze", "002", "003", "005", "004", "006"]
n_rows = 225000               # Change this to your desired fixed length

header_path_baseline = 'collected_data/baseline_PPG'
input_base_baseline = os.path.join(header_path_baseline)
output_base_baseline = os.path.join(header_path_baseline)
n_rows_baseline = 15000               # Change this to your desired fixed length

# Baseline cut
# for folder in folder_names:
#     input_folder = os.path.join(input_base_baseline, folder)
#     output_folder = os.path.join(output_base_baseline, folder)
#     print(f"Processing folder: {input_folder}")
#     process_folder(input_folder, output_folder, n_rows_baseline)
#     check_folder_csv_length(input_folder, n_rows_baseline)

# Exp data cut
for folder in folder_names:
    input_folder = os.path.join(input_base, folder)
    output_folder = os.path.join(output_base, folder)
    print(f"Processing folder: {input_folder}")
    process_folder(input_folder, output_folder, n_rows)
    check_folder_csv_length(input_folder, n_rows)