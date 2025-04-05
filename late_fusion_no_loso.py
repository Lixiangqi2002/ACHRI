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

class LateFusionModel(nn.Module):
    def __init__(self, ppg_encoder, hr_encoder, thermal_encoder, ppg_regressor, hr_regressor, thermal_regressor):
        super(LateFusionModel, self).__init__()
        self.ppg_encoder = ppg_encoder
        self.hr_encoder = hr_encoder
        self.thermal_encoder = thermal_encoder
        
        self.ppg_regressor = ppg_regressor
        self.hr_regressor = hr_regressor
        self.thermal_regressor = thermal_regressor
        self.fusion = nn.Linear(3, 1)

    def forward(self, ppg_data, thermal_data, hr_data):
        ppg_features = self.ppg_encoder(ppg_data)
        hr_features = self.hr_encoder(hr_data)
        thermal_features = self.thermal_encoder(thermal_data)

        ppg_pred = self.ppg_regressor(ppg_features)
        hr_pred = self.hr_regressor(hr_features)
        thermal_pred = self.thermal_regressor(thermal_features)
        inputs = torch.cat([ppg_pred, hr_pred, thermal_pred], dim=1)
        return self.fusion(inputs)   
    

def train(model, train_loader, val_loader, device, num_epochs=500, lr=0.0005):
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        train_loss = 0.0

        for temp_data, ppg_data, hr_data, labels in train_loader:
            temp_data, ppg_data, hr_data, labels = temp_data.to(device), ppg_data.to(device), hr_data.to(device), labels.to(device)

            optimizer.zero_grad()
            final_pred = model(ppg_data, temp_data, hr_data) 
            loss = criterion(final_pred, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for temp, ppg, hr, label in val_loader:
                temp, ppg, hr, label = temp.to(device), ppg.to(device), hr.to(device), label.to(device)
                val_pred = model(ppg, temp, hr)
                loss = criterion(val_pred, label.unsqueeze(1))
                val_running_loss += loss.item() * temp.size(0)

        val_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"Saving best model at epoch {epoch+1}")
            torch.save(model.state_dict(), "weights/late_fusion_model_best.pth")

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

    print("Training Complete!")

    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.show()

    return train_losses, val_losses


def evaluate(model, train_loader, val_loader, test_loader, device):
    model.load_state_dict(torch.load("weights/late_fusion_model_best.pth"))
    criterion = nn.MSELoss()
    model.eval()
    test_running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for temp, ppg, hr, label in test_loader:
            temp, ppg, hr, label = temp.to(device), ppg.to(device), hr.to(device), label.to(device)
            predictions = model(ppg, temp, hr)
            loss = criterion(predictions, label.unsqueeze(1))
            test_running_loss += loss.item() * temp.size(0)
            all_preds.append(predictions.cpu().numpy())
            all_labels.append(label.cpu().numpy())

    test_loss = test_running_loss / len(test_loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    print('max of pred label',max(all_preds))
    print('max of actual label',max(all_labels))
    print(f"Test Loss: {test_loss:.4f}")

    mae = mean_absolute_error(all_labels, all_preds)
    mse = mean_squared_error(all_labels, all_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_labels, all_preds)

    print(f"Test MSE:   {mse:.4f}")
    print(f"Test RMSE:  {rmse:.4f}")
    print(f"Test MAE:   {mae:.4f}")
    print(f"Test R^2:   {r2:.4f}")

    # Prediction vs Groundtruth
    plt.figure(figsize=(6,6))
    plt.scatter(all_labels, all_preds, alpha=0.5)
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Perfect prediction line
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("Actual Score")
    plt.ylabel("Predicted Score")
    plt.title("Predicted vs. Actual Emotion Scores")
    plt.show()

    subset_preds = all_preds[:200]
    subset_labels = all_labels[:200]
    t = np.arange(200) 

    plt.figure(figsize=(10,4))
    plt.plot(t, subset_labels, label="Actual")
    plt.plot(t, subset_preds, label="Predicted")
    plt.xlabel("Time Index")
    plt.ylabel("Emotion Score")
    plt.title("Predicted vs. Actual Over Time (sample window)")
    plt.legend()
    plt.show()
    
    

def main():
    temp_data_dir = "data/temp_preprocessed"
    ppg_data_dir  = "data/ppg_preprocessed"
    hr_data_dir   = "data/hr_preprocessed"
    label_path    = "data/labels"

    dataset = MultiModalDataset(temp_data_dir, ppg_data_dir, hr_data_dir, label_path)
    train_size = int(0.7 * len(dataset))
    val_size   = int(0.15 * len(dataset))
    test_size  = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ppg_encoder = PPGEncoder().to(device)
    hr_encoder = HREncoder().to(device)
    thermal_encoder = TemperatureEncoder().to(device)
    model = LateFusionModel(ppg_encoder, hr_encoder, thermal_encoder,
                            PPGRegressor().to(device), HRRegressor().to(device), TempRegressor().to(device)).to(device)

    # Load pre-trained weights:
    ppg_encoder.load_state_dict(torch.load("weights/best_ppg_encoder.pth", map_location=torch.device("cpu")))
    hr_encoder.load_state_dict(torch.load("weights/best_hr_encoder.pth", map_location=torch.device("cpu")))
    thermal_encoder.load_state_dict(torch.load("weights/best_temperature_encoder.pth", map_location=torch.device("cpu")))

    # Freeze the encoder parameters if you want to train only the fusion module.
    for param in ppg_encoder.parameters():
        param.requires_grad = False
    for param in hr_encoder.parameters():
        param.requires_grad = False
    for param in thermal_encoder.parameters():
        param.requires_grad = False

    # train_losses, val_losses = train(model, train_loader, val_loader, device)
    evaluate(model, train_loader, val_loader, test_loader, device)


if __name__ == "__main__":
    main()
