# Global variables to cache the latest data
last_thermal_data = None
last_thermal_time = 0

ppg_signal_buffer = []  # Cache for real-time input [N, 2] PPG
fs = 250  # Sampling rate in Hz
hr_cache_length = 10 * fs  # Cache 10 seconds of data [2500, 2]
hr_window_size = 4 * fs  # Window size of 4 seconds
hr_step = 1 * fs  # Window step size of 1 second
hr_output_len = 50  # HR output [50, 2]

ppg_temp_path = r"D:/programming/COMP0053/PhysioKit"  # Remember to replace with the absolute path of the PhysioKit root directory
last_seen_ppg_row = None

# -- Thermal --
def get_thermal_data(user_name="01_Ben"):
    global last_thermal_data, last_thermal_time

    dir_path = 'PyThermalCamera-Segmentation/'
    csv_avg = glob.glob(os.path.join(dir_path, f"nose_temp_avg_{user_name}.csv"))
    csv_min = glob.glob(os.path.join(dir_path, f"nose_temp_min_{user_name}.csv"))
    csv_max = glob.glob(os.path.join(dir_path, f"nose_temp_max_{user_name}.csv"))

    if not (csv_avg and csv_min and csv_max):
        raise FileNotFoundError("Missing one or more required CSV files.")

    csv_files = [csv_min[0], csv_max[0], csv_avg[0]]
    
    # For three CSV files, read all data into one 3*n numpy array
    thermal_data_array = np.zeros((6, 3), dtype=np.float32)
    
    for idx, file in enumerate(csv_files):
        thermal_df = pd.read_csv(file)
        thermal_data = thermal_df.values.astype(np.float32)
        
        # If there are fewer than 6 rows, do not read the CSV
        if thermal_data.shape[0] < 6:
            return

        # If there are more than 6 rows, read the most recent 6 rows
        thermal_data = thermal_df.iloc[-6:, 0].values.astype(np.float32)
        min_val, max_val = thermal_data.min(), thermal_data.max()
        denom = max_val - min_val
        if denom < 1e-6:
            thermal_data = np.zeros_like(thermal_data) 
        else:
            thermal_data = (thermal_data - min_val) / denom
        thermal_data_array[:, idx] = thermal_data

    if last_thermal_data is None:
        temp_diff = np.zeros_like(thermal_data_array)
    else:
        temp_diff = thermal_data_array - last_thermal_data

    # Update cache
    last_thermal_data = thermal_data_array.copy()

    # Combine into (6, 6)
    thermal_full = np.hstack((thermal_data_array, temp_diff))  # (6, 6)
    thermal_full = np.nan_to_num(thermal_full) 
    print(thermal_full)

    return thermal_full


# -- PPG --
def get_latest_temp_csv(path=ppg_temp_path):
    csv_files = glob.glob(os.path.join(path, "*_temp.csv"))  # Ensure there is a temp.csv in the PhysioKit root directory for data storage
    if not csv_files:
        raise FileNotFoundError("No temp CSV file found.")
    return max(csv_files, key=os.path.getmtime)

# set PPG_PROJECT_PATH=D:\programming\COMP0053\PhysioKit
def get_ppg_data():
    global last_seen_ppg_row

    file_path = get_latest_temp_csv()
    df = pd.read_csv(file_path)
    ppg_df = df[["PPG A0", "PPG A1"]].dropna()
    ppg_data = ppg_df.iloc[-250:].values.astype(np.float32)

    if ppg_data.shape[0] < 250:
        pad = np.zeros((250 - ppg_data.shape[0], 2), dtype=np.float32)
        ppg_data = np.vstack((pad, ppg_data))

    last_row = ppg_data[-1]

    print("PPG Data Shape:", ppg_data.shape)
    print("PPG Data Sample:", ppg_data[:5])

    return ppg_data


# -- HR --

def extract_hr(signal, fs):
    try:
        signals, _ = nk.ppg_process(signal, sampling_rate=fs)
        return np.nanmean(signals["PPG_Rate"])
    except Exception as e:
        print(f"[extract_hr] Error: {e}")
        return np.nan

def extract_hrv_rmssd(signal, fs):
    try:
        signals, info = nk.ppg_process(signal, sampling_rate=fs)
        if "PPG_Peaks" not in info or len(info["PPG_Peaks"]) < 2:
            return np.nan
        peaks = info["PPG_Peaks"]
        ibi = np.diff(peaks) / fs
        if len(ibi) < 2:
            return np.nan
        rmssd = np.sqrt(np.mean(np.diff(ibi)**2))
        return rmssd
    except Exception as e:
        print(f"[extract_hrv_rmssd] Error: {e}")
        return np.nan

def update_ppg_buffer(new_ppg_data):
    global ppg_signal_buffer
    if new_ppg_data is None:
        return
    if len(ppg_signal_buffer) == 0:
        ppg_signal_buffer = new_ppg_data
    else:
        ppg_signal_buffer = np.vstack([ppg_signal_buffer, new_ppg_data])
    
    # Save the last 10 seconds of data
    if len(ppg_signal_buffer) > hr_cache_length:
        ppg_signal_buffer = ppg_signal_buffer[-hr_cache_length:]

def get_hr_data():
    global ppg_signal_buffer

    if len(ppg_signal_buffer) < hr_cache_length:
        print(f"No enough PPG data yet ({len(ppg_signal_buffer)} / {hr_cache_length}), using default [HR, HRV] = [70, 0]")
        return np.full((hr_output_len, 2), [70.0, 0.0], dtype=np.float32)

    ppg_left = ppg_signal_buffer[:, 0]
    ppg_right = ppg_signal_buffer[:, 1]

    hr_list, hrv_list = [], []

    for start in range(0, len(ppg_left) - hr_window_size + 1, hr_step):
        left_win = ppg_left[start:start + hr_window_size]
        right_win = ppg_right[start:start + hr_window_size]

        hr_left = extract_hr(left_win, fs)
        hr_right = extract_hr(right_win, fs)
        hr = np.nanmean([hr_left, hr_right])

        hrv_left = extract_hrv_rmssd(left_win, fs)
        hrv_right = extract_hrv_rmssd(right_win, fs)
        hrv = np.nanmean([hrv_left, hrv_right])

        hr_list.append(hr)
        hrv_list.append(hrv)

    hr_array = np.stack([hr_list, hrv_list], axis=1).astype(np.float32)

    # Reshape to [50, 2]
    if len(hr_array) < hr_output_len:
        pad = np.full((hr_output_len - len(hr_array), 2), [70.0, 0.0], dtype=np.float32)
        hr_array = np.vstack([pad, hr_array])
    else:
        hr_array = hr_array[-hr_output_len:]

    print("HR Data Shape:", hr_array.shape)
    print("HR Data Sample:", hr_array)
    return hr_array
