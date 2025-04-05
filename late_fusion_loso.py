import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from dataloader import MultiModalDataset
from temperature_encoders import TemperatureEncoder, EmotionRegressor as TempRegressor
from PPG_encoders import PPGEncoder, EmotionRegressor as PPGRegressor
from hr_encoders import HREncoder, EmotionRegressor as HRRegressor
import random

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
        return self.fusion(torch.cat([ppg_pred, hr_pred, thermal_pred], dim=1))


def train_and_eval(train_loader, val_loader, test_loader, device, subject_id):
    # Initialize the model
    ppg_encoder = PPGEncoder().to(device)
    hr_encoder = HREncoder().to(device)
    temp_encoder = TemperatureEncoder().to(device)
    model = LateFusionModel(ppg_encoder, hr_encoder, temp_encoder,
                            PPGRegressor().to(device), HRRegressor().to(device), TempRegressor().to(device)).to(device)

    # Load encoder weights and freeze them
    ppg_encoder.load_state_dict(torch.load("weights/best_ppg_encoder.pth", map_location=device))
    hr_encoder.load_state_dict(torch.load("weights/best_hr_encoder.pth", map_location=device))
    temp_encoder.load_state_dict(torch.load("weights/best_temperature_encoder.pth", map_location=device))
    for enc in [ppg_encoder, hr_encoder, temp_encoder]:
        for p in enc.parameters():
            p.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    for epoch in range(50):
        model.train()
        train_loss = 0
        for temp, ppg, hr, label, _ in train_loader:
            temp, ppg, hr, label = temp.to(device), ppg.to(device), hr.to(device), label.to(device)
            optimizer.zero_grad()
            pred = model(ppg, temp, hr)
            loss = criterion(pred.squeeze(), label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for temp, ppg, hr, label, _ in val_loader:
                temp, ppg, hr, label = temp.to(device), ppg.to(device), hr.to(device), label.to(device)
                pred = model(ppg, temp, hr)
                val_loss += criterion(pred.squeeze(), label).item()
        val_loss /= len(val_loader)

        print(f"[{subject_id}] Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"weights/late_fusion_best_{subject_id}.pth")

    # Testing
    model.load_state_dict(torch.load(f"weights/late_fusion_best_{subject_id}.pth"))
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for temp, ppg, hr, label, _ in test_loader:
            temp, ppg, hr = temp.to(device), ppg.to(device), hr.to(device)
            out = model(ppg, temp, hr).squeeze().cpu().numpy()
            preds.append(out)
            labels.append(label.numpy())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)

    mse = mean_squared_error(labels, preds)
    mae = mean_absolute_error(labels, preds)
    r2 = r2_score(labels, preds)
    print(f"[{subject_id}] Test MSE: {mse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")

    plt.figure(figsize=(10, 4))
    plt.plot(labels, label="True", alpha=0.7, marker="o", linestyle="")
    plt.plot(preds, label="Predicted", alpha=0.7, marker="x", linestyle="")
    plt.legend()
    plt.title(f"Subject {subject_id} - True vs. Predicted")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"figures/late_fusion_loso_{subject_id}.png")
    plt.close()

    return mse, mae, r2

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For CUDA determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    set_seed(0)
    temp_data_dir = "data/temp_preprocessed"
    ppg_data_dir  = "data/ppg_preprocessed"
    hr_data_dir   = "data/hr_preprocessed"
    label_path    = "data/labels"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MultiModalDataset(temp_data_dir, ppg_data_dir, hr_data_dir, label_path, loso=True)
    subject_ids = np.array([dataset[i][-1] for i in range(len(dataset))])
    all_subjects = sorted(set(subject_ids))

    all_results = []
    for test_subject in all_subjects:
        test_indices = [i for i, s in enumerate(subject_ids) if s == test_subject]
        trainval_indices = [i for i in range(len(dataset)) if i not in test_indices]

        # Split train/val
        train_len = int(len(trainval_indices) * 0.75)
        train_subset = Subset(dataset, trainval_indices[:train_len])
        val_subset = Subset(dataset, trainval_indices[train_len:])
        test_subset = Subset(dataset, test_indices)

        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)

        print(f"\n=== LOSO Fold: Leave Out {test_subject} ===")
        mse, mae, r2 = train_and_eval(train_loader, val_loader, test_loader, device, test_subject)
        all_results.append((test_subject, mse, mae, r2))

    print("\n===== LOSO Final Summary =====")
    all_mse = []
    all_mae = []
    all_r2 = []
    all_subjects = []    
    for sid, mse, mae, r2 in all_results:
        all_mse.append(mse)
        all_mae.append(mae)
        all_r2.append(r2)
        all_subjects.append(sid)
        print(f"{sid} | MSE: {mse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")

    print("\nAverage Results:")
    print(f"MSE: {np.mean([r[1] for r in all_results]):.4f}")
    print(f"MAE: {np.mean([r[2] for r in all_results]):.4f}")
    print(f"R² : {np.mean([r[3] for r in all_results]):.4f}")
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.bar(all_subjects, all_mse, color='skyblue')
    plt.xticks(rotation=45)
    plt.ylabel("MSE")
    plt.title("LOSO - MSE per Subject")

    plt.subplot(1, 3, 2)
    plt.bar(all_subjects, all_mae, color='orange')
    plt.xticks(rotation=45)
    plt.ylabel("MAE")
    plt.title("LOSO - MAE per Subject")

    plt.subplot(1, 3, 3)
    plt.bar(all_subjects, all_r2, color='green')
    plt.xticks(rotation=45)
    plt.ylabel("R²")
    plt.title("LOSO - R² per Subject")

    plt.tight_layout()
    plt.savefig("figures/late_loso_results.png")
    plt.show()

if __name__ == "__main__":
    main()
