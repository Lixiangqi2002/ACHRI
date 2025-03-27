from PPG_encoders import PPGEncoder  
from hr_encoders import HREncoder      
from temperature_encoders import TemperatureEncoder
from mid_fusion import FusionModule, MidFusionModel
import torch
import torch.nn as nn
from numpy import random
import numpy as np
import requests

from sensor_data_interface import get_thermal_data, get_ppg_data, get_hr_data
import time

def load_model(device):
    ppg_encoder = PPGEncoder(input_dim=2, num_layers=1, cnn_channels=64, lstm_hidden_dim=64)
    hr_encoder = HREncoder(input_dim=2, num_layers=1, cnn_channels=64, lstm_hidden_dim=64)
    thermal_encoder = TemperatureEncoder(input_dim=6, hidden_dim=16, num_layers=2, window_size=5)

    ppg_encoder.load_state_dict(torch.load("weights/best_ppg_encoder.pth", map_location=device))
    hr_encoder.load_state_dict(torch.load("weights/best_hr_encoder.pth", map_location=device))
    thermal_encoder.load_state_dict(torch.load("weights/best_temperature_encoder.pth", map_location=device))
    fusion_module = FusionModule(ppg_feat_dim=64, thermal_feat_dim=16, hr_feat_dim=64, fused_dim=144)
    model = MidFusionModel(ppg_encoder, hr_encoder, thermal_encoder, fusion_module)
    model.to(device)
    model.load_state_dict(torch.load("weights/mid_fusion_model_best.pth", map_location=device))
    
    fusion_module = FusionModule(ppg_feat_dim=64, thermal_feat_dim=16, hr_feat_dim=64, fused_dim=144)
    model.to(device)
    model.eval()  
    return model


def real_time_inference(model, ppg_data, thermal_data, hr_data, device):
    if not torch.is_tensor(ppg_data):
        ppg_data = torch.tensor(ppg_data, dtype=torch.float32)
    if not torch.is_tensor(thermal_data):
        thermal_data = torch.tensor(thermal_data, dtype=torch.float32)
    if not torch.is_tensor(hr_data):
        hr_data = torch.tensor(hr_data, dtype=torch.float32)
    print(ppg_data.shape) # [1, 250, 2]
    print(thermal_data.shape) # [1, 6, 6]
    print(hr_data.shape) # [1, 50, 2]
    if ppg_data.dim() == 2:
        ppg_data = ppg_data.unsqueeze(0)
        print(ppg_data.shape)
    if thermal_data.dim() == 2:
        thermal_data = thermal_data.unsqueeze(0)
        print(thermal_data.shape)
    if hr_data.dim() == 2:
        hr_data = hr_data.unsqueeze(0)
        print(hr_data.shape)
    
    ppg_data = ppg_data.to(device)
    thermal_data = thermal_data.to(device)
    hr_data = hr_data.to(device)
    print(ppg_data.device) # cuda:0
    print(thermal_data.device) # cuda:0
    print(hr_data.device) # cuda:0
    
    with torch.no_grad():
        prediction = model(ppg_data, thermal_data, hr_data)
    prediction_value = prediction.cpu().numpy()
    prediction_value_post = float(prediction.cpu().numpy().flatten()[0])  
    
    print(f"Predicted Emotion Score: {prediction_value_post:.4f}")
    try:
        response = requests.post("http://localhost:5000/predict_level_offset", json={"prediction": prediction_value_post})
        print("Sent prediction to server:", response.json())
    except Exception as e:
        print("Error sending prediction:", str(e))
    return prediction_value


def real_time_sensor():
    # ppg_data = np.random.rand(250, 2).astype(np.float32)  # [A0, A1]
    # thermal_data = np.random.rand(6, 6).astype(np.float32)  # [min, max, avg, min_diff, max_diff, avg_diff]
    # hr_data = np.random.rand(50, 2).astype(np.float32)  # [hr, hrv]

    ppg_data = get_ppg_data()
    thermal_data = get_thermal_data()
    hr_data = get_hr_data()

    print(f"PPG Data: {ppg_data.shape}, Thermal Data: {thermal_data.shape}, HR Data: {hr_data.shape}")
    return ppg_data, thermal_data, hr_data


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = load_model(device)
    while True:
        ppg_data, thermal_data, hr_data = real_time_sensor()
        prediction = real_time_inference(model, ppg_data, thermal_data, hr_data, device)
        print("Prediction:", prediction)
        time.sleep(1) # 模拟1秒钟获取一次数据