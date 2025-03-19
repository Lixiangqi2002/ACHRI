import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression  
import scipy.stats as stats

from dataloader import MultiModalDataset

from temperature_encoders import TemperatureEncoder, EmotionRegressor as TempRegressor
from PPG_encoders import PPGEncoder, EmotionRegressor as PPGRegressor
from hr_encoders import HREncoder,  EmotionRegressor as HRRegressor

def main():
    # 1. 加载数据集
    temp_data_dir = "data/temp_preprocessed"
    ppg_data_dir  = "data/ppg_preprocessed"
    hr_data_dir   = "data/hr_preprocessed"
    label_path    = "data/labels"

    dataset = MultiModalDataset(temp_data_dir, ppg_data_dir, hr_data_dir, label_path)

    train_size = int(0.7 * len(dataset))
    val_size   = int(0.15 * len(dataset))
    test_size  = len(dataset) - train_size - val_size
    train_dataset, _, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 初始化模型并加载权重
    def load_model(encoder_class, regressor_class, encoder_path, regressor_path, input_dim):
        encoder = encoder_class().to(device)
        regressor = regressor_class(input_dim=input_dim).to(device)
        encoder.load_state_dict(torch.load(encoder_path))
        regressor.load_state_dict(torch.load(regressor_path))
        encoder.eval()
        regressor.eval()
        return encoder, regressor

    temp_encoder, temp_regressor = load_model(
        TemperatureEncoder, TempRegressor, 
        "weights/best_temperature_encoder.pth", 
        "weights/best_temperature_emotion_regressor.pth", 16)

    ppg_encoder, ppg_regressor = load_model(
        PPGEncoder, PPGRegressor, 
        "weights/best_ppg_encoder.pth", 
        "weights/best_ppg_emotion_regressor.pth", 64)

    hr_encoder, hr_regressor = load_model(
        HREncoder, HRRegressor, 
        "weights/best_hr_encoder.pth", 
        "weights/best_hr_emotion_regressor.pth", 64)

    # 3. 训练线性回归
    train_features = []
    train_labels = []

    with torch.no_grad():
        for batch_temp, batch_ppg, batch_hr, batch_y in train_loader:
            batch_temp, batch_ppg, batch_hr, batch_y = (
                batch_temp.to(device),
                batch_ppg.to(device),
                batch_hr.to(device),
                batch_y.to(device)
            )

            temp_pred = temp_regressor(temp_encoder(batch_temp)).squeeze().cpu().numpy()
            ppg_pred  = ppg_regressor(ppg_encoder(batch_ppg)).squeeze().cpu().numpy()
            hr_pred   = hr_regressor(hr_encoder(batch_hr)).squeeze().cpu().numpy()
            batch_y   = batch_y.cpu().numpy()

            train_features.append(np.stack([temp_pred, ppg_pred, hr_pred], axis=1))
            train_labels.append(batch_y)

    train_features = np.concatenate(train_features)
    train_labels = np.concatenate(train_labels)

    # 训练线性回归模型
    voting_regressor = LinearRegression()
    voting_regressor.fit(train_features, train_labels)

    print("Learned Late Fusion Weights:", voting_regressor.coef_)

    # 4. 在测试集中进行 Late Fusion Voting
    all_labels = []
    all_preds = []

    criterion = nn.MSELoss()
    test_loss = 0.0

    with torch.no_grad():
        for batch_temp, batch_ppg, batch_hr, batch_y in test_loader:
            batch_temp, batch_ppg, batch_hr, batch_y = (
                batch_temp.to(device),
                batch_ppg.to(device),
                batch_hr.to(device),
                batch_y.to(device)
            )

            temp_pred = temp_regressor(temp_encoder(batch_temp)).squeeze().cpu().numpy()
            ppg_pred  = ppg_regressor(ppg_encoder(batch_ppg)).squeeze().cpu().numpy()
            hr_pred   = hr_regressor(hr_encoder(batch_hr)).squeeze().cpu().numpy()
            batch_y   = batch_y.cpu().numpy()


            fused_pred = voting_regressor.predict(np.stack([temp_pred, ppg_pred, hr_pred], axis=1))

            loss = mean_squared_error(batch_y, fused_pred)
            test_loss += loss

            all_labels.append(batch_y)
            all_preds.append(fused_pred)

    test_loss /= len(test_loader)
    all_labels = np.concatenate(all_labels)
    all_preds  = np.concatenate(all_preds)

    # 5. 统计指标
    mse  = mean_squared_error(all_labels, all_preds)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(all_labels, all_preds)
    r2   = r2_score(all_labels, all_preds)

    errors = all_preds - all_labels
    conf_int = stats.norm.interval(0.95, loc=np.mean(errors), scale=np.std(errors))
    worst_idx = np.argsort(np.abs(errors))[-10:]

    print("\n======== Late Fusion Test Results ========")
    print(f"Late-Fusion Test Loss (MSE): {test_loss:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R² :  {r2:.4f}")
    print(f"95% Confidence Interval for Error: {conf_int}")

    print("\nWorst Predictions:")
    for i in worst_idx:
        print(f"True: {all_labels[i]:.4f}, Predicted: {all_preds[i]:.4f}, Error: {errors[i]:.4f}")

if __name__ == "__main__":
    main()
