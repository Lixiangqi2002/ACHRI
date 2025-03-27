import glob
import os
import pandas as pd
import numpy as np
import time
import importlib.util
import sys

# 全局变量 缓存最新数据
last_thermal_data = None
last_thermal_time = 0
last_hr_data = None
last_hr_time = 0

ppg_temp_path = r"D:/programming/COMP0053/PhysioKit"
last_seen_ppg_row = None

def get_thermal_data(user_name="111"):
    global last_thermal_data, last_thermal_time
    
    return np.random.rand(6, 6).astype(np.float32)

def get_latest_temp_csv(path=ppg_temp_path):
    csv_files = glob.glob(os.path.join(path, "*_temp.csv"))
    if not csv_files:
        raise FileNotFoundError("No temp CSV file found.")
    return max(csv_files, key=os.path.getmtime)

# set PPG_PROJECT_PATH=D:\programming\COMP0053\PhysioKit
def get_ppg_data():
    global last_seen_ppg_row

    # try:
    file_path = get_latest_temp_csv()
    df = pd.read_csv(file_path)
    ppg_df = df[["PPG A0", "PPG A1"]].dropna()
    ppg_data = ppg_df.iloc[-250:].values.astype(np.float32)

    if ppg_data.shape[0] < 250:
        pad = np.zeros((250 - ppg_data.shape[0], 2), dtype=np.float32)
        ppg_data = np.vstack((pad, ppg_data))

    last_row = ppg_data[-1]

    # # check 更新频率
    # if last_seen_ppg_row is not None and np.array_equal(last_row, last_seen_ppg_row):
    #     print("PPG data not updated, skipping...")
    #     return None
    # else:
    #     last_seen_ppg_row = last_row

    print("PPG Data Shape:", ppg_data.shape)
    print("PPG Data Sample:", ppg_data[:5])

    return ppg_data
        

    # except Exception as e: # 不行就先随机
    #     print(f"Error reading PPG data: {str(e)}")
    #     return np.random.rand(250, 2).astype(np.float32)

def get_hr_data():
    global last_hr_data, last_hr_time

    return np.random.rand(50, 2).astype(np.float32)