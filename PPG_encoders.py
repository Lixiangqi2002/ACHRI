import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from dataloader import MultiModalDataset
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import scipy.stats as stats
import matplotlib.pyplot as plt

class PPGEncoder(nn.Module):
    def __init__(self,
                 input_dim=2,
                 cnn_channels=64,         # 卷积通道数增大
                 lstm_hidden_dim=64,
                 num_layers=3,            # LSTM 层数从 1 改为 2
                 bidirectional=True,      # 可选：使用双向 LSTM
                 dropout_prob=0.1):       # Dropout 概率
        super(PPGEncoder, self).__init__()

        # ---------------------- 卷积部分（增加层数 + BatchNorm）----------------------
        self.conv1 = nn.Conv1d(in_channels=input_dim,
                               out_channels=cnn_channels,
                               kernel_size=5,
                               stride=1,
                               padding=2)
        self.bn1 = nn.BatchNorm1d(cnn_channels)

        self.conv2 = nn.Conv1d(in_channels=cnn_channels,
                               out_channels=cnn_channels,
                               kernel_size=5,
                               stride=1,
                               padding=2)
        self.bn2 = nn.BatchNorm1d(cnn_channels)

        # 额外添加一层卷积
        self.conv3 = nn.Conv1d(in_channels=cnn_channels,
                               out_channels=cnn_channels,
                               kernel_size=5,
                               stride=1,
                               padding=2)
        self.bn3 = nn.BatchNorm1d(cnn_channels)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # ---------------------- LSTM 部分（双向 + 多层 + Dropout）----------------------
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size=cnn_channels,
                            hidden_size=lstm_hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout_prob if num_layers > 1 else 0.0,
                            bidirectional=bidirectional)

        # 如果使用双向 LSTM，最终输出维度要 x2
        final_lstm_dim = lstm_hidden_dim * (2 if bidirectional else 1)

        # ---------------------- 全连接层 & Dropout ----------------------
        # 这里的线性层输出依然是 64 维
        self.fc = nn.Linear(final_lstm_dim, 64)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        """
        x: (batch_size, sequence_length, input_dim) -> (batch_size, 250, 2)  # PPG data
        """
        x = x.permute(0, 2, 1)  # Transform to (batch_size, input_dim, seq_length) to fit 1D CNN

        x = self.pool(torch.relu(self.conv1(x)))  # (batch_size, cnn_channels, new_seq_len)
        x = self.pool(torch.relu(self.conv2(x)))  # (batch_size, cnn_channels, new_seq_len)

        x = x.permute(0, 2, 1)  # Transform back to (batch_size, new_seq_len, cnn_channels)
        x, _ = self.lstm(x)  # LSTM processes temporal features
        x = x[:, -1, :]  # Take the output of the last time step of LSTM
        x = self.dropout(x)
        return self.fc(x)  # (batch_size, 128)


class EmotionRegressor(nn.Module):
    def __init__(self, input_dim=64):
        super(EmotionRegressor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Sigmoid normalization output (0,1)
        )

    def forward(self, x):
        return self.mlp(x)  # (batch_size, 1)



if __name__=="__main__":
    temp_data_dir = "data/temp_preprocessed"
    ppg_data_dir = "data/ppg_preprocessed"
    label_path = "data/labels"
    hr_data_dir = "data/hr_preprocessed"


    dataset = MultiModalDataset(temp_data_dir, ppg_data_dir,hr_data_dir, label_path)

    train_size = int(0.7 * len(dataset))  # 70%
    val_size = int(0.15 * len(dataset))   # 15%
    test_size = len(dataset) - train_size - val_size  # 15%

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Training Dataset: {len(train_dataset)}")
    print(f"Validating Dataset: {len(val_dataset)}")
    print(f"Testing Dataset: {len(test_dataset)}")

    for temp, ppg, hr, label in train_loader:
        print(f"Nose Temperature : {temp.shape}")  # (batch_size, 6, 3)
        # print(temp)
        print(f"PPG Data: {ppg.shape}")  # (batch_size, 250, 2)
        # print(ppg)
        print(f"Labels: {label.shape}")  # (batch_size, 1)
        print(label)
        break
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = PPGEncoder().to(device)
    regressor = EmotionRegressor().to(device)


    ##################################################
    # training
    ##################################################
    optimizer = optim.Adam(list(encoder.parameters()) + list(regressor.parameters()), lr=0.0001, weight_decay=1e-5)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    criterion = nn.MSELoss()

    epochs = 300
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    best_val_loss = float("inf")

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        encoder.train()
        regressor.train()
        train_loss = 0

        for _, batch_ppg, _, batch_y in train_loader:
            batch_ppg, batch_y = batch_ppg.to(device), batch_y.to(device)

            optimizer.zero_grad()
            features = encoder(batch_ppg)
            output = regressor(features).squeeze()
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)  # Store training loss

        encoder.eval()
        regressor.eval()
        val_loss = 0

        with torch.no_grad():
            for _, batch_ppg,_, batch_y in val_loader:
                batch_ppg, batch_y = batch_ppg.to(device), batch_y.to(device)
                features = encoder(batch_ppg)
                output = regressor(features).squeeze()
                loss = criterion(output, batch_y)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)  # Store validation loss
        if val_loss < best_val_loss:
            print(f"Saving model with validation loss: {val_loss:.4f}")
            best_val_loss = val_loss
            torch.save(encoder.state_dict(), "weights/best_ppg_encoder.pth")
            torch.save(regressor.state_dict(), "weights/best_ppg_emotion_regressor.pth")

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        scheduler.step(val_loss)
    print("Complete training!")

    ##################################################
    # testing
    ##################################################
    print("Testing...")
    encoder.load_state_dict(torch.load("weights/best_ppg_encoder.pth"))
    regressor.load_state_dict(torch.load("weights/best_ppg_emotion_regressor.pth"))

    encoder.to(device).eval()
    regressor.to(device).eval()

    test_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for _, batch_ppg, _, batch_y in test_loader:
            batch_ppg, batch_y = batch_ppg.to(device), batch_y.to(device)
            features = encoder(batch_ppg)
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
