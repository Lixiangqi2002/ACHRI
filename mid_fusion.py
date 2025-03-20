import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Assume these are your encoder classes defined in your files.
from PPG_encoders import PPGEncoder  
from hr_encoders import HREncoder      
from temperature_encoders import TemperatureEncoder
from dataloader import MultiModalDataset
# And your fusion module (or you can define one as below).

class FusionModule(nn.Module):
    def __init__(self, ppg_feat_dim=64, thermal_feat_dim=16, hr_feat_dim=64, fused_dim=128):
        super(FusionModule, self).__init__()
        self.fusion = nn.Sequential(
            nn.Linear(ppg_feat_dim + thermal_feat_dim + hr_feat_dim, fused_dim),
            nn.ReLU(),
            nn.Linear(fused_dim, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
        
    def forward(self, ppg_features, thermal_features, hr_features):
        # Concatenate features from each modality: 64+16+64 = 144.
        fused = torch.cat([ppg_features, thermal_features, hr_features], dim=1)
        return self.fusion(fused)   


class MidFusionModel(nn.Module):
    def __init__(self, ppg_encoder, hr_encoder, thermal_encoder, fusion_module):
        super(MidFusionModel, self).__init__()
        self.ppg_encoder = ppg_encoder
        self.hr_encoder = hr_encoder
        self.thermal_encoder = thermal_encoder
        self.fusion = fusion_module
        
    def forward(self, ppg_data, thermal_data, hr_data):
        ppg_features = self.ppg_encoder(ppg_data)
        hr_features = self.hr_encoder(hr_data)
        thermal_features = self.thermal_encoder(thermal_data)
        return self.fusion(ppg_features, thermal_features, hr_features)


def train(model, train_loader, val_loader, device, num_epochs=500, lr=0.0005):
    # Define loss function and optimizer.
    criterion = nn.MSELoss()  # Assuming a regression task (emotion score in 0-9).
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

    # Training loop.
    num_epochs = 500
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for temp, ppg, hr, label in train_loader:
            # Move data to device.
            temp, ppg, hr, label = temp.to(device), ppg.to(device), hr.to(device), label.to(device)
            
            optimizer.zero_grad()
            # Assuming your model forward signature is: model(ppg_data, temp_data, hr_data)
            # (Make sure the order matches your model definition.)
            predictions = model(ppg, temp, hr)
            loss = criterion(predictions, label.unsqueeze(1))  # Make sure label shape is (batch_size, 1)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * temp.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # Validation loop.
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for temp, ppg, hr, label in val_loader:
                temp, ppg, hr, label = temp.to(device), ppg.to(device), hr.to(device), label.to(device)
                predictions = model(ppg, temp, hr)
                loss = criterion(predictions, label.unsqueeze(1))
                val_running_loss += loss.item() * temp.size(0)
        
        val_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save the best model weights.
            print(f"Saving model weights at epoch {epoch+1}")
            torch.save(model.state_dict(), "weights/mid_fusion_model_best.pth")
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

    print("Training complete!")
    # Analysis on training.
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.show()


def test(model, test_loader, device):
    criterion = nn.MSELoss()
    model.load_state_dict(torch.load("weights/mid_fusion_model_best.pth"))
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

    # all_preds, all_labels are your final predicted and ground-truth arrays
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

    # Suppose we pick the first 200 samples from the test set for demonstration
    subset_preds = all_preds[:200]
    subset_labels = all_labels[:200]
    t = np.arange(200)  # "time" index for plotting, or real timestamps if available

    plt.figure(figsize=(10,4))
    plt.plot(t, subset_labels, label="Actual")
    plt.plot(t, subset_preds, label="Predicted")
    plt.xlabel("Time Index")
    plt.ylabel("Emotion Score")
    plt.title("Predicted vs. Actual Over Time (sample window)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Instantiate your dataset:
    temp_data_dir = "data/temp_preprocessed"
    ppg_data_dir = "data/ppg_preprocessed"
    label_path = "data/labels"
    hr_data_dir = "data/hr_preprocessed"

    dataset = MultiModalDataset(temp_data_dir, ppg_data_dir,hr_data_dir, label_path)
    print('this is the full lenght of the dataset', len(dataset))
    # Split dataset into training, validation, and testing sets.
    total_samples = len(dataset)
    train_size = int(0.7 * total_samples)
    val_size = int(0.15 * total_samples)
    test_size = total_samples - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")

    # Create DataLoaders.
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Instantiate pre-trained encoders.
    ppg_encoder = PPGEncoder(input_dim=2, num_layers=1, cnn_channels=64, lstm_hidden_dim=64)
    hr_encoder = HREncoder(input_dim=2, num_layers=1, cnn_channels=64, lstm_hidden_dim=64)
    thermal_encoder = TemperatureEncoder(input_dim=6, hidden_dim=16, num_layers=2, window_size=5)

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

    # Set up your model.
    # Create fusion module and full late fusion model.
    fusion_module = FusionModule(ppg_feat_dim=64, thermal_feat_dim=16, hr_feat_dim=64, fused_dim=144)
    model = MidFusionModel(ppg_encoder, hr_encoder, thermal_encoder, fusion_module)
    # For example, if you have a MidFusionModel defined (which uses your pre-trained encoders and a fusion module):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    print("Using device:", device)

    # train(model, train_loader, val_loader, device)

    test(model, test_loader, device)