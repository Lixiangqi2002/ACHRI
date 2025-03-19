from turtle import st
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.stats import skew, kurtosis
from scipy.fftpack import fft
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR


class MultiModalDatasetWithFeatureEngineeringNoPCA(Dataset):
    def __init__(self, temp_data_dir, ppg_data_dir, hr_data_dir, label_path):
        self.temp_data_dir = temp_data_dir
        self.ppg_data_dir = ppg_data_dir
        self.hr_data_dir = hr_data_dir
        self.label_data_dir = label_path

        self.temp_files = sorted([f for f in os.listdir(temp_data_dir) if f.endswith(".csv")])
        self.ppg_files = sorted([f for f in os.listdir(ppg_data_dir) if f.endswith(".csv")])
        self.hr_files = sorted([f for f in os.listdir(hr_data_dir) if f.endswith(".csv")])
        self.label_files = sorted([f for f in os.listdir(label_path) if f.endswith(".csv")])

        self.person_temp_data = {}
        self.person_ppg_data = {}
        self.person_hr_data = {}
        self.person_label_data = {} 

        # Nose Temperature
        for file in self.temp_files:
            file_path = os.path.join(temp_data_dir, file)
            df = pd.read_csv(file_path, header=None).values  
            person_name = file.split("_")[-2]  
            # if person_name != "byz":
            #     continue
            
            if person_name not in self.person_temp_data:
                self.person_temp_data[person_name] = {}

            if "min" in file:
                self.person_temp_data[person_name]["min"] = df
            elif "max" in file:
                self.person_temp_data[person_name]["max"] = df
            elif "avg" in file:
                self.person_temp_data[person_name]["avg"] = df

        # PPG
        for file in self.ppg_files:
            file_path = os.path.join(ppg_data_dir, file)
            df = pd.read_csv(file_path, header=0, names=["PPG_A0", "PPG_A1", "event_code"])
            df = df.dropna(axis=1, how='all')

            person_name = file.split("_")[0]
            # if person_name != "byz":
            #     continue

            if person_name not in self.person_ppg_data:
                self.person_ppg_data[person_name] = {}
            
            # Min-max normalization
            left = df["PPG_A0"]
            left_min = left.min()
            left_max = left.max()
            normalized_left = (left - left_min) / (left_max - left_min)

            right = df["PPG_A1"]
            right_min = right.min()
            right_max = right.max()
            normalized_right = (right - right_min) / (right_max - right_min)

            self.person_ppg_data[person_name]["A0"] = normalized_left
            self.person_ppg_data[person_name]["A1"] = normalized_right

        # HR HRV
        for file in self.hr_files:
            file_path = os.path.join(hr_data_dir, file)
            df = pd.read_csv(file_path, header=0, names=["timestamp", "hr", "hrv"])
            df = df.dropna(axis=1, how='all')
            person_name = file.split("_")[0]
            # if person_name != "byz":
            #     continue
            if person_name not in self.person_hr_data:
                self.person_hr_data[person_name] = {}
            
            self.person_hr_data[person_name]["hr"] = df["hr"]
            self.person_hr_data[person_name]["hrv"] = df["hrv"]

        # Labels
        for file in self.label_files:
            file_path = os.path.join(label_path, file)
            df = pd.read_csv(file_path, header=0, names=["timestamp", "arousal"])
            df = df.dropna(axis=1, how='all')

            person_name = file.split("_")[0]
            print(person_name)
            # if person_name != "byz":
            #     continue
            if person_name not in self.person_label_data:
                self.person_label_data[person_name] = {}

            # Min-max normalization
            arousal = df["arousal"]
            arousal_min = arousal.min()
            arousal_max = arousal.max()
            normalized_arousal = (arousal - arousal_min) / (arousal_max - arousal_min)

            self.person_label_data[person_name]["arousal"] = normalized_arousal

        self.num_subjects = len(self.person_temp_data)  # Number of subjects
        self.num_seconds = 900  # Number of seconds per subject
        self.total_samples = self.num_subjects * self.num_seconds  # Total samples
        print(f"Total samples: {self.total_samples}")

    def __len__(self):
        return self.total_samples
    
    def extract_ppg_features(self, ppg_segment):
        """
        Extract statistical and frequency domain features from PPG signals
        """
        ppg_segment = np.array(ppg_segment).astype(np.float32)  

        return [
            np.mean(ppg_segment), np.std(ppg_segment), np.max(ppg_segment), np.min(ppg_segment), 
            np.median(ppg_segment), np.ptp(ppg_segment),  # Peak-to-peak value
            np.nan_to_num(skew(ppg_segment), nan=0.0), np.nan_to_num(kurtosis(ppg_segment), nan=0.0),  # Skewness, Kurtosis
            np.sum(np.abs(fft(ppg_segment)[:10]))  # Frequency domain energy
        ]

    def extract_hr_features(self, hr_segment):
        """
        Extract HR / HRV features
        """
        hr_segment = np.array(hr_segment).astype(np.float32)  

        return [
            np.mean(hr_segment), np.std(hr_segment), np.max(hr_segment), np.min(hr_segment),
            hr_segment[-1] - hr_segment[0]  # Rate of change
        ]

    def extract_temp_features(self, temp_segment):
        """
        Extract statistical features from temperature data
        """
        temp_segment = np.array(temp_segment).astype(np.float32)
        return [
            np.mean(temp_segment), np.std(temp_segment),
            temp_segment[-1] - temp_segment[0]  # Rate of change
        ]

    def __getitem__(self, idx):
        # Determine the subject and time for the sample
        subject_idx = idx // self.num_seconds
        time_idx = idx % self.num_seconds
        person_name = list(self.person_temp_data.keys())[subject_idx]
        temp_data = self.person_temp_data[person_name]
        ppg_data = self.person_ppg_data[person_name]
        hr_data = self.person_hr_data[person_name]

        # Get 1 second of PPG data (250, 2)
        ppg_start = time_idx * 250
        ppg_end = ppg_start + 250
        ppg_segment_A0 = ppg_data["A0"][ppg_start:ppg_end]
        ppg_segment_A1 = ppg_data["A1"][ppg_start:ppg_end]

        # Get 1 second of temperature data (6, 3)
        temp_start = time_idx * 6
        temp_end = temp_start + 6
        temp_segment = np.hstack((temp_data["min"][temp_start:temp_end],
                                  temp_data["max"][temp_start:temp_end],
                                  temp_data["avg"][temp_start:temp_end])).flatten()

        # Get 1 second of HR data (50, 2)
        hr_start = time_idx * 50
        hr_end = hr_start + 50
        hr_segment_hr = hr_data["hr"][hr_start:hr_end]
        hr_segment_hrv = hr_data["hrv"][hr_start:hr_end]

        # Extract features
        ppg_features_A0 = self.extract_ppg_features(ppg_segment_A0)
        ppg_features_A1 = self.extract_ppg_features(ppg_segment_A1)
        hr_features = self.extract_hr_features(hr_segment_hr) + self.extract_hr_features(hr_segment_hrv)
        temp_features = self.extract_temp_features(temp_segment)

        # Concatenate all features (≈ 25~40 dimensions)
        feature_vector = np.array(ppg_features_A0 + ppg_features_A1 + hr_features + temp_features)

        # Get label
        label_value = self.person_label_data[person_name]["arousal"][time_idx]

        return (
            torch.tensor(feature_vector, dtype=torch.float32),  # (Feature dimensions,)
            torch.tensor(label_value, dtype=torch.float32)  # (1,)
        )



class MultiModalDatasetWithFeatureEngineering(Dataset):
    def __init__(self, temp_data_dir, ppg_data_dir, hr_data_dir, label_path, pca_components=0.95):
        self.temp_data_dir = temp_data_dir
        self.ppg_data_dir = ppg_data_dir
        self.hr_data_dir = hr_data_dir
        self.label_data_dir = label_path
        self.pca_components = pca_components  # Can be 0.95 (95% variance) or an integer (fixed dimensions)

        self.temp_files = sorted([f for f in os.listdir(temp_data_dir) if f.endswith(".csv")])
        self.ppg_files = sorted([f for f in os.listdir(ppg_data_dir) if f.endswith(".csv")])
        self.hr_files = sorted([f for f in os.listdir(hr_data_dir) if f.endswith(".csv")])
        self.label_files = sorted([f for f in os.listdir(label_path) if f.endswith(".csv")])

        self.person_temp_data = {}
        self.person_ppg_data = {}
        self.person_hr_data = {}
        self.person_label_data = {} 

        # ============== Load Data =======================
        self.load_temperature_data()
        self.load_ppg_data()
        self.load_hr_data()
        self.load_labels()

        self.num_subjects = len(self.person_temp_data)  
        self.num_seconds = 900  
        self.total_samples = self.num_subjects * self.num_seconds  

        # ============== Extract Features for All Samples =======================
        print("Extracting features for all samples...")
        self.X_features = np.array([self.extract_features(idx) for idx in range(self.total_samples)])
        self.y_labels = np.array([self.get_label(idx) for idx in range(self.total_samples)])

        # ============== Standardization + PCA Dimensionality Reduction =======================
        print("Applying PCA...")
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(self.X_features)

        self.pca = PCA(n_components=self.pca_components)
        # print(f"X_features row 2720: {self.X_features[2710:2730]}")

        self.X_pca = self.pca.fit_transform(X_scaled)
        
        print(f"Original feature dimensions: {self.X_features.shape[1]}, PCA reduced dimensions: {self.X_pca.shape[1]}")
        print(f"Explained variance ratio: {self.pca.explained_variance_ratio_}")
        print(f"Cumulative variance explained: {np.cumsum(self.pca.explained_variance_ratio_)}")

        feature_names = [f"Feature {i+1}" for i in range(31)]  
        pca_components_df = pd.DataFrame(self.pca.components_, columns=feature_names)
        print(pca_components_df)

        top_features_per_component = pca_components_df.apply(lambda x: x.abs().idxmax(), axis=1)
        print(top_features_per_component)

        plt.figure(figsize=(8,5))
        plt.plot(np.cumsum(self.pca.explained_variance_ratio_), marker='o', linestyle='--')
        plt.xlabel("Number of Principal Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.title("PCA Explained Variance")
        plt.grid()
        plt.show()

        plt.figure(figsize=(12, 6))
        sns.heatmap(pca_components_df, cmap="coolwarm", center=0, annot=False)
        plt.xlabel("Original Features")
        plt.ylabel("Principal Components")
        plt.title("PCA Component Loadings")
        plt.show()

    def load_temperature_data(self):
        """ Load nose temperature data """
        for file in self.temp_files:
            file_path = os.path.join(self.temp_data_dir, file)
            df = pd.read_csv(file_path, header=None).values  
            person_name = file.split("_")[-2]  
            if person_name not in self.person_temp_data:
                self.person_temp_data[person_name] = {}

            if "min" in file:
                self.person_temp_data[person_name]["min"] = df
            elif "max" in file:
                self.person_temp_data[person_name]["max"] = df
            elif "avg" in file:
                self.person_temp_data[person_name]["avg"] = df

    def load_ppg_data(self):
        """ Load PPG data """
        for file in self.ppg_files:
            file_path = os.path.join(self.ppg_data_dir, file)
            df = pd.read_csv(file_path, header=0, names=["PPG_A0", "PPG_A1", "event_code"])
            df = df.dropna(axis=1, how='all')
            person_name = file.split("_")[0]

            if person_name not in self.person_ppg_data:
                self.person_ppg_data[person_name] = {}

            # Min-max normalization
            self.person_ppg_data[person_name]["A0"] = (df["PPG_A0"] - df["PPG_A0"].min()) / (df["PPG_A0"].max() - df["PPG_A0"].min())
            self.person_ppg_data[person_name]["A1"] = (df["PPG_A1"] - df["PPG_A1"].min()) / (df["PPG_A1"].max() - df["PPG_A1"].min())

    def load_hr_data(self):
        """ Load HR/HRV data """
        for file in self.hr_files:
            file_path = os.path.join(self.hr_data_dir, file)
            df = pd.read_csv(file_path, header=0, names=["timestamp", "hr", "hrv"])
            df = df.dropna(axis=1, how='all')
            person_name = file.split("_")[0]
            if person_name not in self.person_hr_data:
                self.person_hr_data[person_name] = {}

            self.person_hr_data[person_name]["hr"] = df["hr"]
            self.person_hr_data[person_name]["hrv"] = df["hrv"]

    def load_labels(self):
        """ Load label data """
        for file in self.label_files:
            file_path = os.path.join(self.label_data_dir, file)
            df = pd.read_csv(file_path, header=0, names=["timestamp", "arousal"])
            df = df.dropna(axis=1, how='all')

            person_name = file.split("_")[0]
            if person_name not in self.person_label_data:
                self.person_label_data[person_name] = {}

            arousal = df["arousal"]
            self.person_label_data[person_name]["arousal"] = (arousal - arousal.min()) / (arousal.max() - arousal.min())

    def extract_features(self, idx):
        """ Compute all features for a single sample """
        subject_idx = idx // self.num_seconds
        time_idx = idx % self.num_seconds
        person_name = list(self.person_temp_data.keys())[subject_idx]
        # ppg 9 + 9, hr 5 + 5, temp 3 = 31
        ppg_features_A0 = self.extract_ppg_features(self.person_ppg_data[person_name]["A0"][time_idx * 250:(time_idx + 1) * 250])
        ppg_features_A1 = self.extract_ppg_features(self.person_ppg_data[person_name]["A1"][time_idx * 250:(time_idx + 1) * 250])
        hr_features = self.extract_hr_features(self.person_hr_data[person_name]["hr"][time_idx * 50:(time_idx + 1) * 50]) + \
                      self.extract_hr_features(self.person_hr_data[person_name]["hrv"][time_idx * 50:(time_idx + 1) * 50])
        temp_features = self.extract_temp_features(self.person_temp_data[person_name]["avg"][time_idx * 6:(time_idx + 1) * 6])

        return np.array(ppg_features_A0 + ppg_features_A1 + hr_features + temp_features)

    def get_label(self, idx):
        subject_idx = idx // self.num_seconds
        time_idx = idx % self.num_seconds
        person_name = list(self.person_label_data.keys())[subject_idx]
        return self.person_label_data[person_name]["arousal"][time_idx]

    def __len__(self):
        return self.total_samples
        
    def extract_ppg_features(self, ppg_segment):
        """
        Extract statistical and frequency domain features from PPG signals
        """
        ppg_segment = np.array(ppg_segment).astype(np.float32)  

        return list([ # 9
            np.mean(ppg_segment), np.std(ppg_segment), np.max(ppg_segment), np.min(ppg_segment), 
            np.median(ppg_segment), np.ptp(ppg_segment),  # Peak-to-peak value
            np.nan_to_num(skew(ppg_segment), nan=0.0), np.nan_to_num(kurtosis(ppg_segment), nan=0.0),  # Skewness, Kurtosis
            np.sum(np.abs(fft(ppg_segment)[:10]))  # Frequency domain energy
        ])

    def extract_hr_features(self, hr_segment):
        """
        Extract HR / HRV features
        """
        hr_segment = np.array(hr_segment).astype(np.float32)  

        return list([ # 5
            np.mean(hr_segment), np.std(hr_segment), np.max(hr_segment), np.min(hr_segment),
            hr_segment[-1] - hr_segment[0]  # Rate of change
        ])

    def extract_temp_features(self, temp_segment):
        """
        Extract statistical features from temperature data
        """
        temp_segment = np.array(temp_segment).astype(np.float32)
        return list([ # 3
            np.mean(temp_segment), np.std(temp_segment),
            (temp_segment[-1] - temp_segment[0]).item()  # Rate of change
        ])
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.X_pca[idx], dtype=torch.float32),
            torch.tensor(self.y_labels[idx], dtype=torch.float32)
        )



def train(dataloader, model_type="SVM", plot_loss_curve=True):
    X, y = [], []
    for features, labels in dataloader:
        X.append(features.numpy())
        y.append(labels.numpy())

    X = np.vstack(X)
    y = np.hstack(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type == "SVM":
        model = SVR(kernel="rbf")
    else:
        raise ValueError("Invalid model type")
    
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    return y_train, y_train_pred, y_test, y_test_pred



def test(model_type, y_train, y_train_pred, y_test, y_test_pred):
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mse_test)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)
    errors = y_test_pred - y_test
    ci = stats.norm.interval(0.95, loc=np.mean(errors), scale=np.std(errors))

    print(f"\n{model_type} Results:")
    print(f"Train Loss (MSE): {mse_train:.4f}")
    print(f"Test Loss (MSE): {mse_test:.4f}")
    print(f"RMSE: {rmse_test:.4f}")
    print(f"MAE: {mae_test:.4f}")
    print(f"R² Score: {r2_test:.4f}")
    print(f"95% Confidence Interval for Error: {ci}")

    print(f"{model_type} - Train MSE: {mse_train:.4f}, Test MSE: {mse_test:.4f}")
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.5)
    plt.plot([0, 1], [0, 1], '--', color='red', transform=plt.gca().transAxes)  # Reference line
    plt.xlabel("True Labels")
    plt.ylabel("Predicted Labels")
    plt.title("Predictions vs True Values")
    plt.grid(True)
    plt.show()

    return mse_train, mse_test


if __name__ == "__main__":
    temp_data_dir = "data/temp_preprocessed"
    ppg_data_dir = "data/ppg_preprocessed"
    label_path = "data/labels"
    hr_data_dir = "data/hr_preprocessed"

    print("No PCA")
    dataset_noPCA = MultiModalDatasetWithFeatureEngineeringNoPCA(temp_data_dir, ppg_data_dir, hr_data_dir, label_path)
    dataloader_noPCA = DataLoader(dataset_noPCA, batch_size=32, shuffle=False)
    for features, label in dataloader_noPCA:
        print(f"Features: {features.shape}")
        # print(features)
        print(f"Labels: {label.shape}")
        # print(label)
        break
    
    print("PCA")
    dataset_PCA = MultiModalDatasetWithFeatureEngineering(temp_data_dir, ppg_data_dir, hr_data_dir, label_path, pca_components=0.95)
    dataloader_PCA = DataLoader(dataset_PCA, batch_size=32, shuffle=False)
    for features, label in dataloader_PCA:
        print(f"Features: {features.shape}")
        # print(features)
        print(f"Labels: {label.shape}")
        # print(label)
        break

    print("Training on raw features (No PCA)...")
    y_train, y_train_pred, y_test, y_test_pred = train(dataloader_noPCA, model_type="SVM")
    test("SVM", y_train, y_train_pred, y_test, y_test_pred)

    print("\nTraining on PCA-transformed features...")
    y_train, y_train_pred, y_test, y_test_pred = train(dataloader_PCA, model_type="SVM")
    test("SVM", y_train, y_train_pred, y_test, y_test_pred)


