# ACHRI-Coursework

## Web-based Game for Various Difficulties

This project provides a comprehensive method to calculate the difficulty of 2D platformer game maps. The difficulty score is evaluated from several key factors, including:

- **Jump Complexity (J)**: Measures the difficulty of required jumps.
- **Lava Number (L)**: Counts the number of lava hazards.
- **Coin Complexity (C)**: Evaluates how difficult it is to collect coins.
- **Solution Length (S)**: The optimal shortest path required to complete the level.
- **Platform Layout Difficulty (P)**: Measures the complexity of the level’s platforms.
- **Coins Count**: The total number of collectible coins.

We designers evaluated the difficulties of each level game map ourselves. 

## Model

### Mid Level Fusion
#### Overall Metrics
```
Test Loss: 0.0071
Test MSE:   0.0071
Test RMSE:  0.0841
Test MAE:   0.0571
Test R^2:   0.8206
```
#### LOSO Metrics
```
===== LOSO Final Summary =====
byz | MSE: 0.0078 | MAE: 0.0653 | R²: 0.6207
lhb | MSE: 0.0187 | MAE: 0.1008 | R²: 0.2463
lxq | MSE: 0.0147 | MAE: 0.0852 | R²: 0.3462
qzw | MSE: 0.0137 | MAE: 0.0925 | R²: 0.2815
wyk | MSE: 0.0240 | MAE: 0.1273 | R²: 0.0434
zxj | MSE: 0.0085 | MAE: 0.0731 | R²: 0.1291

Average Results:
MSE: 0.0146
MAE: 0.0907
R² : 0.2779
```

### Late Level Fusion
#### Overall Metrics
```
Test Loss: 0.0094
Test MSE:   0.0094
Test RMSE:  0.0967
Test MAE:   0.0685
Test R^2:   0.7569
```
#### LOSO Metrics
```
===== LOSO Final Summary =====
byz | MSE: 0.0100 | MAE: 0.0746 | R²: 0.5121
lhb | MSE: 0.0208 | MAE: 0.1141 | R²: 0.1598
lxq | MSE: 0.0153 | MAE: 0.0898 | R²: 0.3172
qzw | MSE: 0.0148 | MAE: 0.0932 | R²: 0.2249
wyk | MSE: 0.0305 | MAE: 0.1472 | R²: -0.2153
zxj | MSE: 0.0366 | MAE: 0.1828 | R²: -2.7619

Average Results:
MSE: 0.0214
MAE: 0.1169
R² : -0.2938
```

### Early Level Fusion
#### Without PCA
##### Overall Metrics
```
SVM Results:
Train Loss (MSE): 0.0265
Test Loss (MSE): 0.0248
RMSE: 0.1574
MAE: 0.1221
R² Score: 0.3333
95% Confidence Interval for Error: (np.float64(-0.3135153544494832), np.float64(0.30318525426938037))
```
##### LOSO Metrics
```
LOSO Fold: Test Subject = byz
  MSE: 0.0154, MAE: 0.0965, R²: 0.2513

LOSO Fold: Test Subject = lhb
  MSE: 0.0264, MAE: 0.1221, R²: -0.0630

LOSO Fold: Test Subject = lxq
  MSE: 0.0251, MAE: 0.1170, R²: -0.1179

LOSO Fold: Test Subject = qzw
  MSE: 0.0224, MAE: 0.1102, R²: -0.1692

LOSO Fold: Test Subject = wyk
  MSE: 0.0228, MAE: 0.1196, R²: 0.0936

LOSO Fold: Test Subject = zxj
  MSE: 0.0099, MAE: 0.0627, R²: -0.0199

LOSO Final Results:
Average MSE: 0.0203
Average MAE: 0.1047
Average R²:  -0.0042
```

#### With PCA
##### Overall Metrics
```
SVM Results:
Train Loss (MSE): 0.0111
Test Loss (MSE): 0.0132
RMSE: 0.1148
MAE: 0.0851
R² Score: 0.6453
95% Confidence Interval for Error: (np.float64(-0.21820402425365654), np.float64(0.2311350794491045))
```
##### LOSO Metrics
```
LOSO Fold: Test Subject = byz
  MSE: 0.0161, MAE: 0.0953, R²: 0.2146

LOSO Fold: Test Subject = lhb
  MSE: 0.0233, MAE: 0.1108, R²: 0.0612

LOSO Fold: Test Subject = lxq
  MSE: 0.0267, MAE: 0.1165, R²: -0.1911

LOSO Fold: Test Subject = qzw
  MSE: 0.0195, MAE: 0.1080, R²: -0.0209

LOSO Fold: Test Subject = wyk
  MSE: 0.0220, MAE: 0.1159, R²: 0.1240

LOSO Fold: Test Subject = zxj
  MSE: 0.0093, MAE: 0.0628, R²: 0.0419

LOSO Final Results:
Average MSE: 0.0195
Average MAE: 0.1015
Average R²:  0.0383
```

## Pre-trained Encoders (all Overall Metrics)

#### HR & HRV：
```
Test Loss (MSE): 0.0130
RMSE: 0.1153
MAE: 0.0829
R² Score: 0.6305
95% Confidence Interval for Error: (np.float64(-0.23347236297361104), np.float64(0.2175795555793449))

Worst Predictions:
True: 0.8235, Predicted: 0.4701, Error: -0.3535
True: 1.0000, Predicted: 0.6418, Error: -0.3582
True: 0.2941, Predicted: 0.6645, Error: 0.3703
True: 0.8667, Predicted: 0.4892, Error: -0.3775
True: 0.7500, Predicted: 0.3600, Error: -0.3900
True: 0.7059, Predicted: 0.3069, Error: -0.3990
True: 0.7333, Predicted: 0.3262, Error: -0.4071
True: 0.9000, Predicted: 0.4562, Error: -0.4438
True: 0.9667, Predicted: 0.4881, Error: -0.4785
```

#### PPG：
```
Test Loss (MSE): 0.0154
RMSE: 0.1250
MAE: 0.0887
R² Score: 0.5763
95% Confidence Interval for Error: (np.float64(-0.24196718590376382), np.float64(0.24777509415564541))

Worst Predictions:
True: 0.1250, Predicted: 0.5456, Error: 0.4206
True: 0.7333, Predicted: 0.2993, Error: -0.4340
True: 0.1250, Predicted: 0.5696, Error: 0.4446
True: 0.1000, Predicted: 0.5666, Error: 0.4666
True: 0.9000, Predicted: 0.4208, Error: -0.4792
True: 0.8667, Predicted: 0.3088, Error: -0.5579
True: 0.8627, Predicted: 0.3029, Error: -0.5599
True: 0.0000, Predicted: 0.5636, Error: 0.5636
True: 1.0000, Predicted: 0.3996, Error: -0.6004
True: 1.0000, Predicted: 0.3791, Error: -0.6209
```

#### Temperature:
```
Test Loss (MSE): 0.0137
RMSE: 0.1170
MAE: 0.0829
R² Score: 0.6317
95% Confidence Interval for Error: (np.float64(-0.23693073029796274), np.float64(0.22071815143505724))

Worst Predictions:
True: 0.7000, Predicted: 0.2919, Error: -0.4081
True: 0.7667, Predicted: 0.3560, Error: -0.4106
True: 0.8333, Predicted: 0.4134, Error: -0.4199
True: 0.8000, Predicted: 0.3694, Error: -0.4306
True: 0.8235, Predicted: 0.3921, Error: -0.4315
True: 0.8000, Predicted: 0.3363, Error: -0.4637
True: 0.2000, Predicted: 0.6929, Error: 0.4929
True: 0.9000, Predicted: 0.3665, Error: -0.5335
True: 0.2000, Predicted: 0.7395, Error: 0.5395
True: 1.0000, Predicted: 0.3693, Error: -0.6307
```


## To Run this code
### Environment Requirments

```
pip install -r requirements.txt
```

### Web-Based Game

Start the server:
```
python3 server.py
```

Double Click the `index.html`, use F12 (or your personal settings) to observe the flask logging.

### Data Collection Code

In `PysioKit`, run it on Windows system. Please use two PPG configurations with 250 Hz.

In `PyThermalCamera-Segmentation`, run `main.py` on Linux system. This has integrated the nose segmentation in inference, which will generate the nose temperature min, max and avg in the `.csv` files.

### Train Models and Encoders
#### Early Fusion Strategy
Train, Test and Leave-One-Subject-Out Evaluation:
```
python3 early_fusion_with_feature_engineering.py
```

#### Middle Fusion Strategy
Train, Test: 
```
python3 mid_fusion_no_loso.py
```
Leave-One-Subject-Out Evaluation:
```
python3 mid_fusion_loso.py
```
#### Late Fusion Strategy
Train, Test: 
```
python3 late_fusion_no_loso.py
```
Leave-One-Subject-Out Evaluation:
```
python3 late_fusion_loso.py
```
### Real-time Predictions
#### Linux
Run `PyThermalCamera-Segmentation/src/main.py` and `windows_client.py` on Windows, this will use TCP connection to transmit the temperature data to Windows (use port 5000).
#### Windows
Run `PysioKit` with two PPG sensor on both ears.

Run the `real_time_prediction.py` to use mid-level fusion models for doing emotion scores predictions in real-time.
```
python3 real_time_prediction.py
```
Run the `real_time_user_study.py`, this will trigger the dynamic game level adapter according to the predicted emotion scores.
```
python3 real_time_user_study.py
```