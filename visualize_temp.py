import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_dir = "data/raw_temp"

samples = ["001", "002", "003", "005", "004", "006"]
categories = ["min", "max", "avg"]
colors = {"min": "red", "max": "blue", "avg": "green"}  

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Nose Temperature Comparison (Baseline vs Exp, Normalized)", fontsize=16)

for idx, sample in enumerate(samples):
    row, col = divmod(idx, 3) 
    ax = axes[row, col]
    ax.set_title(f"Sample: {sample.upper()}")
    
    for category in categories:
        # baseline_file = os.path.join(data_dir, f"nose_temp_{category}_{sample}_baseline.csv")
        baseline_file = os.path.join(data_dir, f"nose_temp_avg_{sample}_baseline.csv")

        exp_file = os.path.join(data_dir, f"nose_temp_{category}_{sample}_exp.csv")

        if os.path.exists(baseline_file) and os.path.exists(exp_file):

            baseline_df = pd.read_csv(baseline_file)
            exp_df = pd.read_csv(exp_file)

            baseline_mean = baseline_df.iloc[:, 0].mean()
            
            # ax.axhline(y=baseline_mean, color=colors[category], linestyle="dashed", label=f"{category} - Baseline Mean")

            exp_values = exp_df.iloc[:, 0] / baseline_mean
  
            ax.plot(exp_values, label=f"{category} - Exp", color=colors[category])

    ax.set_xlabel("Time")
    ax.set_ylabel("Normalized Temperature")
    ax.legend()
    ax.grid()
    # break

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
