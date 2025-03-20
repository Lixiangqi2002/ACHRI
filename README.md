# ACHRI-Coursework

## Web-based Game for Various Difficulties

This project provides a comprehensive method to calculate the difficulty of 2D platformer game maps. The difficulty score is derived from several key factors, including:

- **Jump Complexity (J)**: Measures the difficulty of required jumps.
- **Lava Number (L)**: Counts the number of lava hazards.
- **Coin Complexity (C)**: Evaluates how difficult it is to collect coins.
- **Solution Length (S)**: The optimal shortest path required to complete the level.
- **Platform Layout Difficulty (P)**: Measures the complexity of the level’s platforms.
- **Coins Count**: The total number of collectible coins.

The final score is computed using a weighted formula to give an objective measure of a map's difficulty.


## Model

### Mid Level Fusion
"""
Test Loss: 0.0071
Test MSE:   0.0071
Test RMSE:  0.0841
Test MAE:   0.0571
Test R^2:   0.8206
"""

### Late Level Fusion
"""
Test Loss: 0.0094
Test MSE:   0.0094
Test RMSE:  0.0967
Test MAE:   0.0685
Test R^2:   0.7569
"""

### Early Level Fusion
Without PCA
"""
SVM Results:
Train Loss (MSE): 0.0265
Test Loss (MSE): 0.0248
RMSE: 0.1574
MAE: 0.1221
R² Score: 0.3333
95% Confidence Interval for Error: (np.float64(-0.3135153544494832), np.float64(0.30318525426938037))
"""

With PCA
"""
SVM Results:
Train Loss (MSE): 0.0111
Test Loss (MSE): 0.0132
RMSE: 0.1148
MAE: 0.0851
R² Score: 0.6453
95% Confidence Interval for Error: (np.float64(-0.21820402425365654), np.float64(0.2311350794491045))
"""

## Pre-trained Encoders

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
True: 0.1667, Predicted: 0.7209, Error: 0.5543
"""

#### PPG：
"""
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
"""

#### Temperature:
"""
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
"""