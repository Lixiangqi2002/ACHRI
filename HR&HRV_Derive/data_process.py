import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, periodogram, find_peaks
from scipy.interpolate import interp1d
import neurokit2 as nk
import matplotlib.pyplot as plt

#############################
# Utility Functions
#############################
def load_ppg_data(filepath, fs=250):
    """
    Load PPG data from a CSV file with two channels: 'PPG A0' (left) and 'PPG A1' (right).

    Returns:
        time (array): Time vector (s).
        ppg_left (array): Left ear data.
        ppg_right (array): Right ear data.
    """
    df = pd.read_csv(filepath)
    ppg_left = df["PPG A0"].values
    ppg_right = df["PPG A1"].values
    time = np.arange(len(ppg_left)) / fs
    return time, ppg_left, ppg_right

def bandpass_filter(data, fs, lowcut=0.1, highcut=0.5, order=4):
    """
    Apply a Butterworth bandpass filter to isolate respiratory frequencies.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def extract_respiratory_rate(signal, fs):
    """
    Estimate respiratory rate (bpm) from a baseline-corrected PPG signal.
    """
    filtered_signal = bandpass_filter(signal, fs)
    freqs, psd = periodogram(filtered_signal, fs)
    valid = np.logical_and(freqs >= 0.1, freqs <= 0.5)
    if not np.any(valid):
        return np.nan
    dominant_freq = freqs[valid][np.argmax(psd[valid])]
    return dominant_freq * 60

def extract_hrv_rmssd(signal_window, fs):
    """
    Extract a time-domain HRV metric (RMSSD) from a window of PPG data using NeuroKit2.
    """
    signals, info = nk.ppg_process(signal_window, sampling_rate=fs)
    if "PPG_Peaks" not in info or len(info["PPG_Peaks"]) < 2:
        return np.nan
    peaks = info["PPG_Peaks"]
    ibi = np.diff(peaks) / fs  # inter-beat intervals in seconds
    if len(ibi) < 2:
        return np.nan
    rmssd = np.sqrt(np.mean(np.diff(ibi)**2))
    return rmssd

def sliding_window_analysis(signal, fs, window_size_sec, step_size_sec, estimator_func, **kwargs):
    """
    Compute an estimate (HR, HRV, RR, etc.) over time using a sliding window.
    
    Returns:
        times (np.array): Center time of each window.
        estimates (np.array): Estimated values per window.
    """
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
    """
    Detect peaks in PPG signal and compute instantaneous HR (bpm) from beat-to-beat intervals.
    
    Returns:
        hr_times (array): Time points (s) for each HR estimate.
        hr_inst (array): Instantaneous HR (bpm).
    """
    peaks, _ = find_peaks(ppg, distance=min_distance)
    beat_times = peaks / fs
    ibi = np.diff(beat_times)
    hr_inst = 60 / ibi
    hr_times = beat_times[:-1] + ibi/2
    return hr_times, hr_inst

def interpolate_signal(times, values, new_sampling_rate):
    """
    Interpolate a time series to a new sampling rate.
    
    Returns:
        new_times (array): New time vector.
        new_values (array): Interpolated values.
    """
    new_dt = 1 / new_sampling_rate
    new_times = np.arange(times[0], times[-1], new_dt)
    interpolator = interp1d(times, values, kind='cubic', fill_value="extrapolate")
    new_values = interpolator(new_times)
    return new_times, new_values

#############################
# Main Processing
#############################
fs = 250  # Sampling frequency

# File paths for baseline and experiment data
baseline_file_path = "collected_data/baseline_PPG/002/002_Game_baseline_1741371475_622094_747569.csv"
experiment_file_path = "collected_data/PPG_trimed_225000/002/002_Game_expCond1_1741372191_419938_430846.csv"

# -----------------------------
# Baseline Correction Using Recorded Baseline Data
# -----------------------------
# Load baseline data and compute a constant baseline for each channel.
time_baseline, baseline_left, baseline_right = load_ppg_data(baseline_file_path, fs)
const_baseline_left = np.mean(baseline_left)
const_baseline_right = np.mean(baseline_right)
print("Computed Baseline (Left Ear):", const_baseline_left)
print("Computed Baseline (Right Ear):", const_baseline_right)

# Load experiment data.
time_exp, exp_left, exp_right = load_ppg_data(experiment_file_path, fs)

# Subtract the computed baseline (from baseline file) from the experiment data.
exp_left_corrected = exp_left - const_baseline_left
exp_right_corrected = exp_right - const_baseline_right


# Plot a selected range (e.g., samples 0 to 1000) of the raw PPG data for both channels.
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

# -----------------------------
# Estimate HRV and HR Without Stretching (Sliding Window Analysis)
# -----------------------------
# Use identical sliding window parameters (here 5-second window, 5-second step) for both HR and HRV.
window_size_sec = 5  # seconds
step_size_sec = 5    # seconds

# HR extraction using NeuroKit2 processing on baseline-corrected signals.
def extract_hr_from_neurokit(window, fs):
    signals, _ = nk.ppg_process(window, sampling_rate=fs)
    return np.nanmean(signals["PPG_Rate"])

hr_times_left, hr_estimates_left = sliding_window_analysis(exp_left_corrected, fs, window_size_sec, step_size_sec, extract_hr_from_neurokit)
hr_times_right, hr_estimates_right = sliding_window_analysis(exp_right_corrected, fs, window_size_sec, step_size_sec, extract_hr_from_neurokit)
hr_times = np.nanmean([hr_times_left, hr_times_right], axis=0)
hr_estimates = np.nanmean([hr_estimates_left, hr_estimates_right], axis=0)

# HRV extraction (RMSSD) using sliding window analysis.
hrv_times_left, hrv_estimates_left = sliding_window_analysis(exp_left_corrected, fs, window_size_sec, step_size_sec, extract_hrv_rmssd)
hrv_times_right, hrv_estimates_right = sliding_window_analysis(exp_right_corrected, fs, window_size_sec, step_size_sec, extract_hrv_rmssd)
hrv_times = np.nanmean([hrv_times_left, hrv_times_right], axis=0)
hrv_estimates = np.nanmean([hrv_estimates_left, hrv_estimates_right], axis=0)

print('Before time stretch (HR):', len(hr_estimates))
print('Before time stretch (HRV):', len(hrv_estimates))

# Figure 1: Plot HR and HRV together (without time stretching) on twin axes.
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(hr_times, hrv_estimates, 'o-', color='green', label='HRV (RMSSD)')
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

def pad_signal(signal, pad_seconds, fs):
    """
    Pad the signal at the beginning and end with edge values.
    
    Parameters:
        signal (array): Input signal.
        pad_seconds (float): Number of seconds to pad at each end.
        fs (int): Sampling frequency.
    
    Returns:
        padded_signal (array): Padded signal.
        pad_samples (int): Number of samples padded at each end.
    """
    pad_samples = int(pad_seconds * fs)
    padded_signal = np.pad(signal, pad_width=(pad_samples, pad_samples), mode='edge')
    return padded_signal, pad_samples

# For the time stretching (long window) part, assume:
ws_long = 30  # window size in seconds
ss_long = 5   # step size in seconds
pad_seconds = ws_long / 2  # pad with half window length, e.g., 15 s

# Pad the baseline-corrected signals.
padded_left, pad_samples = pad_signal(exp_left_corrected, pad_seconds, fs)
padded_right, _ = pad_signal(exp_right_corrected, pad_seconds, fs)

# Run sliding window analysis on the padded signals.
hr_times_left_long, hr_estimates_left_long = sliding_window_analysis(padded_left, fs, ws_long, ss_long, extract_hr_from_neurokit)
hr_times_right_long, hr_estimates_right_long = sliding_window_analysis(padded_right, fs, ws_long, ss_long, extract_hr_from_neurokit)
# The returned time stamps are relative to the padded signal.
# Adjust by subtracting pad_seconds to bring them back to the original timeline.
hr_times_left_long_adjusted = hr_times_left_long - pad_seconds
hr_times_right_long_adjusted = hr_times_right_long - pad_seconds
hr_times_dense = np.nanmean([hr_times_left_long_adjusted, hr_times_right_long_adjusted], axis=0)
hr_estimates_dense = np.nanmean([hr_estimates_left_long, hr_estimates_right_long], axis=0)

hrv_times_left_long, hrv_estimates_left_long = sliding_window_analysis(padded_left, fs, ws_long, ss_long, extract_hrv_rmssd)
hrv_times_right_long, hrv_estimates_right_long = sliding_window_analysis(padded_right, fs, ws_long, ss_long, extract_hrv_rmssd)
hrv_times_left_long_adjusted = hrv_times_left_long - pad_seconds
hrv_times_right_long_adjusted = hrv_times_right_long - pad_seconds
hrv_times_dense = np.nanmean([hrv_times_left_long_adjusted, hrv_times_right_long_adjusted], axis=0)
hrv_estimates_dense = np.nanmean([hrv_estimates_left_long, hrv_estimates_right_long], axis=0)

# Now define a common time range based on the adjusted time stamps.
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
