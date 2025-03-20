import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import scipy.stats as stats
import matplotlib.pyplot as plt
from dataloader import MultiModalDataset

class HREncoder(nn.Module):
    def __init__(self, input_dim=2, num_layers=1, cnn_channels=64, lstm_hidden_dim=64):
        super(HREncoder, self).__init__()

        # 1D CNN: Extract local features
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=cnn_channels, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # LSTM: Extract time series features
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # Fully connected layer: Map features to 64 dimensions
        self.fc = nn.Linear(lstm_hidden_dim, 64)

    def forward(self, x):
        """
        x: (batch_size, sequence_length, input_dim) -> (batch_size, 50, 2)  # HR data (1 second)
        """
        x = x.permute(0, 2, 1)  # Adjust dimensions to fit 1D CNN (batch_size, 2, 50)

        x = self.pool(torch.relu(self.conv1(x)))  # CNN extracts local patterns
        x = self.pool(torch.relu(self.conv2(x)))

        x = x.permute(0, 2, 1)  # Revert to format required by LSTM (batch_size, new_seq_len, cnn_channels)
        x, _ = self.lstm(x)  # Process time information through LSTM
        x = x[:, -1, :]  # Take the last output of LSTM

        return self.fc(x)  # (batch_size, 128)


class EmotionRegressor(nn.Module):
    def __init__(self, input_dim=64):
        super(EmotionRegressor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Normalize output range to (0,1)
        )

    def forward(self, x):
        return self.mlp(x)  # (batch_size, 1)


if __name__=="__main__":
    temp_data_dir = "data/temp_preprocessed"
    ppg_data_dir = "data/ppg_preprocessed"
    label_path = "data/labels"
    hr_data_dir = "data/hr_preprocessed"

    dataset = MultiModalDataset(temp_data_dir, ppg_data_dir, hr_data_dir, label_path)

    # Split into training, validation, and test sets
    train_size = int(0.7 * len(dataset))  # 70% training set
    val_size = int(0.15 * len(dataset))   # 15% validation set
    test_size = len(dataset) - train_size - val_size  # 15% test set

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = HREncoder().to(device)
    regressor = EmotionRegressor().to(device)

    ##################################################
    # Training
    ##################################################
    optimizer = optim.Adam(list(encoder.parameters()) + list(regressor.parameters()), lr=0.0005)
    import torch.optim.lr_scheduler as lr_scheduler
    epochs = 300

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=5)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        encoder.train()
        regressor.train()
        train_loss = 0

        for _, _, batch_hr, batch_y in train_loader:
            batch_hr, batch_y = batch_hr.to(device), batch_y.to(device)

            optimizer.zero_grad()
            features = encoder(batch_hr)
            output = regressor(features).squeeze()
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        encoder.eval()
        regressor.eval()
        val_loss = 0

        with torch.no_grad():
            for _, _, batch_hr, batch_y in val_loader:
                batch_hr, batch_y = batch_hr.to(device), batch_y.to(device)
                features = encoder(batch_hr)
                output = regressor(features).squeeze()
                loss = criterion(output, batch_y)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        if val_loss < best_val_loss:
            print(f"Saving model with validation loss: {val_loss:.4f}")
            best_val_loss = val_loss
            torch.save(encoder.state_dict(), "weights/best_hr_encoder.pth")
            torch.save(regressor.state_dict(), "weights/best_hr_emotion_regressor.pth")

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        scheduler.step(val_loss)

    print("Complete training!")

    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss", marker="o")
    plt.plot(val_losses, label="Validation Loss", marker="s")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    ##################################################
    # Testing
    ##################################################
    print("Testing...")
    encoder.load_state_dict(torch.load("weights/best_hr_encoder.pth"))
    regressor.load_state_dict(torch.load("weights/best_hr_emotion_regressor.pth"))

    encoder.to(device).eval()
    regressor.to(device).eval()

    test_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for _, _, batch_hr, batch_y in test_loader:
            batch_hr, batch_y = batch_hr.to(device), batch_y.to(device)
            features = encoder(batch_hr)
            output = regressor(features).squeeze()

            all_preds.append(output.cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())

            loss = criterion(output, batch_y)
            test_loss += loss.item()

    test_loss /= len(test_loader)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Calculate RMSE, MAE, R²
    mse = mean_squared_error(all_labels, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)

    # Calculate 95% confidence interval
    errors = all_preds - all_labels
    conf_int = stats.norm.interval(0.95, loc=np.mean(errors), scale=np.std(errors))

    # Worst (most erroneous) samples
    worst_idx = np.argsort(np.abs(errors))[-10:]  # Take the 10 samples with the largest errors

    # Print evaluation metrics
    print(f"Test Loss (MSE): {test_loss:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")  # Close to 1 indicates good fit
    print(f"95% Confidence Interval for Error: {conf_int}")

    print("\nWorst Predictions:")
    for i in worst_idx:
        print(f"True: {all_labels[i]:.4f}, Predicted: {all_preds[i]:.4f}, Error: {errors[i]:.4f}")

    # Plot predictions vs true values scatter plot
    plt.figure(figsize=(6, 6))
    plt.scatter(all_labels, all_preds, alpha=0.5)
    plt.plot([0, 1], [0, 1], '--', color='red', transform=plt.gca().transAxes)  # Reference line
    plt.xlabel("True Labels")
    plt.ylabel("Predicted Labels")
    plt.title("Predictions vs True Values")
    plt.grid(True)
    plt.show()

    # Plot error distribution histogram
    plt.hist(errors, bins=50, alpha=0.7, color="blue", edgecolor="black")
    plt.axvline(x=0, color='red', linestyle='--')  # Mean error reference line
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.title("Distribution of Prediction Errors")
    plt.show()