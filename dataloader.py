import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

class MultiModalDataset(Dataset):
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

        # Nose Temp
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
            
            # min-max normalization
            left = df["PPG_A0"]
            left_min = left.min()
            left_max = left.max()
            normalized_left = (left - left_min) / (left_max - left_min)

            right = df["PPG_A1"]
            right_min = right.min()
            right_max = right.max()
            normalized_right = (right - right_min) / (right_max - right_min)

            # self.person_label_data[person_name]["arousal"] = normalized_arousal
            self.person_ppg_data[person_name]["A0"] = normalized_left #df["PPG_A0"]  
            self.person_ppg_data[person_name]["A1"] = normalized_right #df["PPG_A1"]  

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

        # labels
        # self.labels = pd.read_csv(label_path, header=None).values  # (6, 900)
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

            # self.person_label_data[person_name]["arousal"] = df["arousal"]
            # Min-max normalization
            arousal = df["arousal"]
            arousal_min = arousal.min()
            arousal_max = arousal.max()
            normalized_arousal = (arousal - arousal_min) / (arousal_max - arousal_min)

            self.person_label_data[person_name]["arousal"] = normalized_arousal

        self.num_subjects = len(self.person_temp_data)  # 6 samples
        self.num_seconds = 900  #  900 
        self.total_samples = self.num_subjects * self.num_seconds  # 5400
        print(f"Total samples: {self.total_samples}")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # which person, which second
        subject_idx = idx // self.num_seconds
        time_idx = idx % self.num_seconds

        person_name = list(self.person_temp_data.keys())[subject_idx]
        temp_data = self.person_temp_data[person_name]
        ppg_data = self.person_ppg_data[person_name]
        hr_data = self.person_hr_data[person_name]

        # get 1 second of temperature data (6, 3)
        temp_start = time_idx * 6
        temp_end = temp_start + 6
        temp_slice = np.hstack((temp_data["min"][temp_start:temp_end],  
                                temp_data["max"][temp_start:temp_end], 
                                temp_data["avg"][temp_start:temp_end]))  # (6, 3)

        # get 1 second of PPG data (250, 2)
        ppg_start = time_idx * 250
        ppg_end = ppg_start + 250
        ppg_slice = np.stack((ppg_data["A0"][ppg_start:ppg_end],  
                              ppg_data["A1"][ppg_start:ppg_end]), axis=1)  # (250, 2)

        # get 1 second of HR data (50, 2)    
        hr_start = time_idx * 50
        hr_end = hr_start + 50
        hr_slice = np.stack((hr_data["hr"][hr_start:hr_end],
                            hr_data["hrv"][hr_start:hr_end]), axis=1)
        
        # get label value (1,)
        label_value = self.person_label_data[person_name]["arousal"][time_idx]#/10  # (1,), normalized to (0,1)

        return (
            torch.tensor(temp_slice, dtype=torch.float32),  # (6, 3)
            torch.tensor(ppg_slice, dtype=torch.float32),  # (250, 2)
            torch.tensor(hr_slice, dtype=torch.float32),  # (50, 2)
            torch.tensor(label_value, dtype=torch.float32),  # (1,)
        )

if __name__ == "__main__":
    temp_data_dir = "data/temp_preprocessed"
    ppg_data_dir = "data/ppg_preprocessed"
    label_path = "data/labels"
    hr_data_dir = "data/hr_preprocessed"

    dataset = MultiModalDataset(temp_data_dir, ppg_data_dir,hr_data_dir, label_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    for temp, ppg, hr, label in dataloader:
        print(f"Nose Temperature : {temp.shape}")  # (batch_size, 6, 3)
        print(temp)
        print(f"PPG Data: {ppg.shape}")  # (batch_size, 250, 2)
        print(ppg)
        print(f"HR Data: {hr.shape}") # (batch_size, 50, 2)
        print(hr)
        print(f"Labels: {label.shape}")  # (batch_size, 1)
        print(label)
        break
