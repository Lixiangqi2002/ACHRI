import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, periodogram, find_peaks
from scipy.interpolate import interp1d
import neurokit2 as nk
import matplotlib.pyplot as plt

#############################
# Utility Functions
#############################
def load_ppg_data(path, fs=250):
    """
    Load PPG data from a CSV file.
    If 'path' is a directory, the function selects the first CSV file found.

    Returns:
        time (array): Time vector (s).
        ppg_left (array): Data from 'PPG A0' (left).
        ppg_right (array): Data from 'PPG A1' (right).
    """
    if os.path.isdir(path):
        files = [f for f in os.listdir(path) if f.lower().endswith('.csv')]
        if len(files) == 0:
            raise FileNotFoundError(f"No CSV files found in folder: {path}")
        if len(files) > 1:
            print(f"Warning: More than one CSV file found in {path}. Using {files[0]}")
        filepath = os.path.join(path, files[0])
    else:
        filepath = path
    df = pd.read_csv(filepath)
    ppg_left = df["PPG A0"].values
    ppg_right = df["PPG A1"].values
    time = np.arange(len(ppg_left)) / fs
    return time, ppg_left, ppg_right

def bandpass_filter(data, fs, lowcut=0.1, highcut=0.5, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def extract_respiratory_rate(signal, fs):
    filtered_signal = bandpass_filter(signal, fs)
    freqs, psd = periodogram(filtered_signal, fs)
    valid = np.logical_and(freqs >= 0.1, freqs <= 0.5)
    if not np.any(valid):
        return np.nan
    dominant_freq = freqs[valid][np.argmax(psd[valid])]
    return dominant_freq * 60

# --这里获取HRV--
def extract_hrv_rmssd(signal_window, fs):
    signals, info = nk.ppg_process(signal_window, sampling_rate=fs)
    if "PPG_Peaks" not in info or len(info["PPG_Peaks"]) < 2:
        return np.nan
    peaks = info["PPG_Peaks"]
    ibi = np.diff(peaks) / fs
    if len(ibi) < 2:
        return np.nan
    rmssd = np.sqrt(np.mean(np.diff(ibi)**2))
    return rmssd

def sliding_window_analysis(signal, fs, window_size_sec, step_size_sec, estimator_func, **kwargs):
    window_size = int(window_size_sec * fs)
    step_size = int(step_size_sec * fs)
    estimates = []
    times = []
    for start in range(0, len(signal) - window_size + 1, step_size):
        window = signal[start:start + window_size]
        est = estimator_func(window, fs, **kwargs)
        time_center = (start + window_size / 2) / fs
        estimates.append(est)
        times.append(time_center)
    return np.array(times), np.array(estimates)

def compute_instantaneous_hr(ppg, fs, min_distance=50):
    peaks, _ = find_peaks(ppg, distance=min_distance)
    beat_times = peaks / fs
    ibi = np.diff(beat_times)
    hr_inst = 60 / ibi
    hr_times = beat_times[:-1] + ibi/2
    return hr_times, hr_inst

def interpolate_signal(times, values, new_sampling_rate):
    new_dt = 1 / new_sampling_rate
    new_times = np.arange(times[0], times[-1], new_dt)
    interpolator = interp1d(times, values, kind='cubic', fill_value="extrapolate")
    new_values = interpolator(new_times)
    return new_times, new_values

def minmax_normalize(values):
    """
    Normalize an array using min–max scaling to the range [0, 1].
    """
    v_min = np.nanmin(values)
    v_max = np.nanmax(values)
    if v_max - v_min == 0:
        return values
    normalized = (values - v_min) / (v_max - v_min)
    return normalized

# --这里获取HR--
def extract_hr_from_neurokit(window, fs):
    signals, _ = nk.ppg_process(window, sampling_rate=fs)
    return np.nanmean(signals["PPG_Rate"])

#############################
# Subject Processing Function
#############################
def process_subject(baseline_folder, experiment_folder, output_csv_path, fs=250,
                    window_size_sec_short=5, step_size_sec_short=5,
                    window_size_sec_long=30, step_size_sec_long=5, new_rate=50):
    """
    Process one subject:
      - Load the CSV file from each folder.
      - Compute the mean baseline for each channel using the baseline folder.
      - Subtract the baseline from experiment data.
      - Perform sliding window analysis to compute HR and HRV (short window, without time stretching).
      - Perform a "stretched" analysis (long window + interpolation) for smoother trends.
      - Plot the results.
      - Save the short-window sliding-window HR and HRV results (raw and normalized) as a CSV.
    """
    # --- Baseline Correction ---
    _, baseline_left, baseline_right = load_ppg_data(baseline_folder, fs)
    const_baseline_left = np.mean(baseline_left)
    const_baseline_right = np.mean(baseline_right)
    print("Computed Baseline (Left):", const_baseline_left)
    print("Computed Baseline (Right):", const_baseline_right)
    
    # Load experiment data.
    time_exp, exp_left, exp_right = load_ppg_data(experiment_folder, fs)
    
    # Baseline correction.
    exp_left_corrected = exp_left - const_baseline_left
    exp_right_corrected = exp_right - const_baseline_right

    # --- Plot Selected Range of Raw PPG Data ---
    start_idx = 0
    end_idx = 1000  # adjust as needed
    plt.figure(figsize=(10, 4))
    plt.plot(time_exp[start_idx:end_idx], exp_left[start_idx:end_idx], label='Left Ear (Raw)')
    plt.plot(time_exp[start_idx:end_idx], exp_right[start_idx:end_idx], label='Right Ear (Raw)', alpha=0.7)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Raw PPG Data (Selected Range)")
    plt.legend()
    plt.grid(True)
    # plt.show()
    
    # --- Sliding Window Analysis Without Time Stretching ---
    ws_short = window_size_sec_short
    ss_short = step_size_sec_short
    
    hr_times_left, hr_estimates_left = sliding_window_analysis(exp_left_corrected, fs, ws_short, ss_short, extract_hr_from_neurokit)
    hr_times_right, hr_estimates_right = sliding_window_analysis(exp_right_corrected, fs, ws_short, ss_short, extract_hr_from_neurokit)
    hr_times = np.nanmean([hr_times_left, hr_times_right], axis=0)
    hr_estimates = np.nanmean([hr_estimates_left, hr_estimates_right], axis=0)
    
    hrv_times_left, hrv_estimates_left = sliding_window_analysis(exp_left_corrected, fs, ws_short, ss_short, extract_hrv_rmssd)
    hrv_times_right, hrv_estimates_right = sliding_window_analysis(exp_right_corrected, fs, ws_short, ss_short, extract_hrv_rmssd)
    hrv_times = np.nanmean([hrv_times_left, hrv_times_right], axis=0)
    hrv_estimates = np.nanmean([hrv_estimates_left, hrv_estimates_right], axis=0)
    
    print('Before time stretch (HR):', len(hr_estimates))
    print('Before time stretch (HRV):', len(hrv_estimates))
    
    # Figure 1: Plot HR and HRV without time stretching on twin axes.
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(hrv_times, hrv_estimates, 'o-', color='green', label='HRV (RMSSD)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('HRV (RMSSD, s)', color='green')
    ax1.tick_params(axis='y', labelcolor='green')
    ax2 = ax1.twinx()
    ax2.plot(hr_times, hr_estimates, 'o-', color='magenta', label='HR')
    ax2.set_ylabel('Heart Rate (bpm)', color='magenta')
    ax2.tick_params(axis='y', labelcolor='magenta')
    plt.title("HR and HRV Over Time (Without Time Stretching)")
    fig1.tight_layout()
    # plt.show()
    
    # --- Sliding Window Analysis With Time Stretching ---
    ws_long = window_size_sec_long
    ss_long = step_size_sec_long
    
    hr_times_left_long, hr_estimates_left_long = sliding_window_analysis(exp_left_corrected, fs, ws_long, ss_long, extract_hr_from_neurokit)
    hr_times_right_long, hr_estimates_right_long = sliding_window_analysis(exp_right_corrected, fs, ws_long, ss_long, extract_hr_from_neurokit)
    hr_times_dense = np.nanmean([hr_times_left_long, hr_times_right_long], axis=0)
    hr_estimates_dense = np.nanmean([hr_estimates_left_long, hr_estimates_right_long], axis=0)
    
    hrv_times_left_long, hrv_estimates_left_long = sliding_window_analysis(exp_left_corrected, fs, ws_long, ss_long, extract_hrv_rmssd)
    hrv_times_right_long, hrv_estimates_right_long = sliding_window_analysis(exp_right_corrected, fs, ws_long, ss_long, extract_hrv_rmssd)
    hrv_times_dense = np.nanmean([hrv_times_left_long, hrv_times_right_long], axis=0)
    hrv_estimates_dense = np.nanmean([hrv_estimates_left_long, hrv_estimates_right_long], axis=0)
    
    common_start = max(hr_times_dense[0], hrv_times_dense[0])
    common_end = min(hr_times_dense[-1], hrv_times_dense[-1])
    # To guarantee exactly 50 samples per second over the original duration:
    # Suppose the original recording (after baseline correction) lasts T seconds.
    # You can compute T from your original time vector (time_exp).
    T = 900  # total duration in seconds
    num_samples = int(T * 50)  # 45001 samples for 50 Hz over 900 seconds
    new_time = np.linspace(0, T, num_samples, endpoint=True)



    # For interpolation, we first need to restrict our dense sliding window data to the overlapping range.
    # (Alternatively, you can choose common_start = time_exp[0] and common_end = time_exp[-1] if your padded analysis covers the whole range.)
    common_start = new_time[0]
    common_end = new_time[-1]

    # Interpolate HR and HRV onto this common dense time grid.
    interp_hr = interp1d(hr_times_dense, hr_estimates_dense, kind='cubic', fill_value="extrapolate")
    interp_hrv = interp1d(hrv_times_dense, hrv_estimates_dense, kind='cubic', fill_value="extrapolate")
    hr_interp = interp_hr(new_time)
    hrv_interp = interp_hrv(new_time)

    # Normalize HR and HRV to the range [0, 1]
    hr_normalized = minmax_normalize(hr_interp)
    hrv_normalized = minmax_normalize(hrv_interp)
    
    print("After time stretch (HR):", len(hr_interp))
    print("After time stretch (HRV):", len(hrv_interp))
    print("New time range from", new_time[0], "to", new_time[-1])
    print('new time', new_time)
    
    # Figure 2: With time stretching, plot HR and HRV in separate subplots.
    fig2, (ax_hr, ax_hrv) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax_hr.plot(new_time, hr_interp, '-', color='magenta', label='Interpolated HR')
    ax_hr.set_ylabel("Heart Rate (bpm)")
    ax_hr.set_title("Interpolated Heart Rate Over Time")
    ax_hr.legend()
    ax_hr.grid(True)
    
    ax_hrv.plot(new_time, hrv_interp, '-', color='green', label='Interpolated HRV (RMSSD)')
    ax_hrv.set_xlabel("Time (s)")
    ax_hrv.set_ylabel("HRV (RMSSD, s)")
    ax_hrv.set_title("Interpolated HRV Over Time")
    ax_hrv.legend()
    ax_hrv.grid(True)
    
    plt.tight_layout()
    # plt.show()

    # Figure 3: With time stretching, plot Normalized HR and HRV in separate subplots.
    fig3, (ax_hr, ax_hrv) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax_hr.plot(new_time, hr_normalized, '-', color='magenta', label='Normalized Interpolated HR')
    ax_hr.set_ylabel("Normalized Heart Rate (bpm)")
    ax_hr.set_title("Interpolated Heart Rate Over Time")
    ax_hr.legend()
    ax_hr.grid(True)
    
    ax_hrv.plot(new_time, hrv_normalized, '-', color='green', label='Normalized Interpolated HRV (RMSSD)')
    ax_hrv.set_xlabel("Time (s)")
    ax_hrv.set_ylabel("Normalized HRV (RMSSD, s)")
    ax_hrv.set_title("Interpolated HRV Over Time")
    ax_hrv.legend()
    ax_hrv.grid(True)
    
    plt.tight_layout()
    # plt.show()
    
    # --- Output CSV ---
    # Save the short-window sliding window results (without time stretching)
    # along with the normalized values.
    results_df = pd.DataFrame({
        "Time (s)": new_time,  # HR and HRV time stamps (short window) should match.
        "HR_norm": hr_normalized,
        "HRV_norm": hrv_normalized
    })
    results_df.to_csv(output_csv_path, index=False)
    print("Processed HR and HRV data saved to:", output_csv_path)

#############################
# Main Loop: Process Multiple Subjects
#############################
baseline_base = "collected_data/baseline_PPG"         # Folder containing subfolders for baseline data
experiment_base = "collected_data/PPG_trimed_225000"    # Folder containing subfolders for experiment data
output_base = "Processed_HR_HRV"               # Output base folder

subject_folders = ["yuze", "002", "003", "005", "004", "006"]

os.makedirs(output_base, exist_ok=True)

for subject in subject_folders:
    baseline_folder = os.path.join(baseline_base, subject)
    experiment_folder = os.path.join(experiment_base, subject)
    output_csv_path = os.path.join(output_base, f"{subject}_HR_HRV.csv")
    print(f"Processing {subject} ...")
    print("Baseline folder:", baseline_folder)
    print("Experiment folder:", experiment_folder)
    print("Output CSV path:", output_csv_path)
    print(f"Processing {subject} ...")
    process_subject(baseline_folder, experiment_folder, output_csv_path, fs=250,
                    window_size_sec_short=5, step_size_sec_short=5,
                    window_size_sec_long=30, step_size_sec_long=5, new_rate=50)