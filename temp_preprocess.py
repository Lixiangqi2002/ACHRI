import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler

# Set data paths
data_dir = "data/raw_temp"  # Replace with your path
output_dir = "data/temp_preprocessed"  # Path to save normalized data
os.makedirs(output_dir, exist_ok=True)

# Get all files
all_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".csv")])

# Classify files into baseline and exp
baseline_files = {t: [] for t in ["min", "max", "avg"]}
exp_files = []

for f in all_files:
    if "baseline" in f:
        if "min" in f:
            baseline_files["min"].append(f)
        elif "max" in f:
            baseline_files["max"].append(f)
        elif "avg" in f:
            baseline_files["avg"].append(f)
    elif "exp" in f:
        exp_files.append(f)

# Calculate the mean of baseline min/max/avg
baseline_means = {}
for temp_type, files in baseline_files.items():
    for file in files:
        file_path = os.path.join(data_dir, file)
        df = pd.read_csv(file_path, header=None)  # Read data
        baseline_means[file] = df.mean().values[0]  # Calculate baseline mean (single column data)

# Process data for each person
people = ["001", "002", "003", "005", "004", "006"]  # Identifiers for 6 people

for person in people:
    plt.figure(figsize=(12, 6))

    for temp_type in ["min", "max", "avg"]:
        exp_file = f"nose_temp_{temp_type}_{person}_exp.csv"
        baseline_file = f"nose_temp_{temp_type}_{person}_baseline.csv"

        # Check if files exist
        if exp_file not in exp_files or baseline_file not in baseline_means:
            print(f"{person}'s {temp_type} file is missing, skipping")
            continue

        # Read data
        df_exp = pd.read_csv(os.path.join(data_dir, exp_file), header=None)

        # Baseline correction
        baseline_mean = baseline_means[baseline_file]
        df_exp_adjusted = df_exp - baseline_mean

        # Resample to 5.5Hz
        target_length = 5400  # 15 min * 6Hz = 5400 points
        original_length = len(df_exp)

        x_original = np.linspace(0, 1, original_length)
        x_target = np.linspace(0, 1, target_length)

        df_exp_resampled = pd.DataFrame(interp1d(x_original, df_exp_adjusted.iloc[:, 0], kind="linear")(x_target))

        # Normalize (Min-Max normalization)
        scaler = MinMaxScaler()
        df_exp_normalized = pd.DataFrame(scaler.fit_transform(df_exp_resampled), columns=df_exp_resampled.columns)

        # Save normalized data
        output_file = os.path.join(output_dir, f"normalized_{exp_file}")
        df_exp_normalized[1:].to_csv(output_file, index=False)

        # Plot
        plt.plot(df_exp.iloc[:, 0], label=f"{temp_type} - original", linestyle="dashed", alpha=0.5)
        plt.plot(df_exp_adjusted.iloc[:, 0], label=f"{temp_type} - adjusted", linestyle="dotted", alpha=0.7)
        # plt.plot(df_exp_normalized.iloc[:, 0], label=f"{temp_type} - normalized", linewidth=2)

    plt.xlabel("Time step")
    plt.ylabel("Temperature")
    plt.legend()
    plt.title(f"Temperature Data Comparison: {person}")
    plt.show()
