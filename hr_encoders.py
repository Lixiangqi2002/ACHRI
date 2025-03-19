import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import scipy.stats as stats
import matplotlib.pyplot as plt
from dataloader import MultiModalDataset
import torch.optim.lr_scheduler as lr_scheduler
class HREncoder(nn.Module):
    def __init__(self,
                 input_dim=2,
                 cnn_channels=128,         # 卷积通道数增大
                 lstm_hidden_dim=64,
                 num_layers=2,            # LSTM 层数从 1 改为 2
                 bidirectional=True,      # 可选：使用双向 LSTM
                 dropout_prob=0.1):       # Dropout 概率
        super(HREncoder, self).__init__()

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
        x: (batch_size, sequence_length, input_dim) -> (batch_size, 50, 2)  # HR data (1 second)
        """
        x = x.permute(0, 2, 1)  # Adjust dimensions to fit 1D CNN (batch_size, 2, 50)

        x = self.pool(torch.relu(self.conv1(x)))  # CNN extracts local patterns
        x = self.pool(torch.relu(self.conv2(x)))

        x = x.permute(0, 2, 1)  # Revert to format required by LSTM (batch_size, new_seq_len, cnn_channels)
        x, _ = self.lstm(x)  # Process time information through LSTM
        x = x[:, -1, :]  # Take the last output of LSTM
        # 在全连接之前再做一次 dropout
        x = self.dropout(x)
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

    # Split data
    train_size = int(0.7 * len(dataset))  # 70%
    val_size = int(0.15 * len(dataset))   # 15%
    test_size = len(dataset) - train_size - val_size  # 15%

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = HREncoder().to(device)
    regressor = EmotionRegressor().to(device)

    # -------------------- Training Setup --------------------
    optimizer = optim.Adam(list(encoder.parameters()) + list(regressor.parameters()), lr=0.0005)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=1e-6)
    criterion = nn.MSELoss()

    epochs = 500
    best_val_loss = float("inf")

    train_losses = []
    val_losses = []

    # ============= Early Stopping 参数 =============
    patience = 30  # 若验证集连续10个epoch不提升，就提早停止
    no_improve_count = 0

    for epoch in range(epochs):
        encoder.train()
        regressor.train()
        train_loss = 0.0

        # -------------------- 训练循环 --------------------
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

        # -------------------- 验证循环 --------------------
        encoder.eval()
        regressor.eval()
        val_loss = 0.0

        with torch.no_grad():
            for _, _, batch_hr, batch_y in val_loader:
                batch_hr, batch_y = batch_hr.to(device), batch_y.to(device)
                features = encoder(batch_hr)
                output = regressor(features).squeeze()
                loss = criterion(output, batch_y)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # 如果验证集表现更好，则保存最优模型，并重置 no_improve_count
        if val_loss < best_val_loss:
            print(f"Saving model with validation loss: {val_loss:.4f}")
            best_val_loss = val_loss
            no_improve_count = 0
            torch.save(encoder.state_dict(), "weights/best_hr_encoder.pth")
            torch.save(regressor.state_dict(), "weights/best_hr_emotion_regressor.pth")
        else:
            no_improve_count += 1

        print(f"Epoch {epoch+1}/{epochs}, "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"No Improve Count: {no_improve_count}")

        # 学习率调度器
        scheduler.step()

        # -------------------- Early Stopping 检查 --------------------
        if no_improve_count >= patience:
            print(f"Validation loss has not improved for {patience} consecutive epochs. Stopping early.")
            break

    print("Complete training or Early Stopping triggered!")

    # ================= 测试阶段 =================
    print("Testing with the best saved model...")
    encoder.load_state_dict(torch.load("weights/best_hr_encoder.pth"))
    regressor.load_state_dict(torch.load("weights/best_hr_emotion_regressor.pth"))

    encoder.to(device).eval()
    regressor.to(device).eval()

    test_loss = 0.0
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

    # Calculate metrics
    mse = mean_squared_error(all_labels, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)

    errors = all_preds - all_labels
    conf_int = stats.norm.interval(0.95, loc=np.mean(errors), scale=np.std(errors))
    worst_idx = np.argsort(np.abs(errors))[-10:]

    # Print evaluation metrics
    print(f"Test Loss (MSE): {test_loss:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"95% Confidence Interval for Error: {conf_int}")

    print("\nWorst Predictions:")
    for i in worst_idx:
        print(f"True: {all_labels[i]:.4f}, Predicted: {all_preds[i]:.4f}, Error: {errors[i]:.4f}")

    # Scatter plot: predictions vs. true
    plt.figure(figsize=(6, 6))
    plt.scatter(all_labels, all_preds, alpha=0.5)
    plt.plot([0, 1], [0, 1], '--', color='red', transform=plt.gca().transAxes)
    plt.xlabel("True Labels")
    plt.ylabel("Predicted Labels")
    plt.title("Predictions vs True Values")
    plt.grid(True)
    plt.show()

    # Histogram of errors
    plt.hist(errors, bins=50, alpha=0.7, color="blue", edgecolor="black")
    plt.axvline(x=0, color='red', linestyle='--')
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.title("Distribution of Prediction Errors")
    plt.show()