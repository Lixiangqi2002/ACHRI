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

#### HR & HRV：
```
Test Loss (MSE): 0.0130
RMSE: 0.1163
MAE: 0.0847
R² Score: 0.6337
95% Confidence Interval for Error: (np.float64(-0.2359455668053243), np.float64(0.21888186527999376))

Worst Predictions:
True: 0.8333, Predicted: 0.4897, Error: -0.3437
True: 0.1667, Predicted: 0.5140, Error: 0.3473
True: 0.6863, Predicted: 0.3241, Error: -0.3621
True: 0.9333, Predicted: 0.5288, Error: -0.4045
True: 0.7333, Predicted: 0.3262, Error: -0.4071
True: 0.7500, Predicted: 0.3164, Error: -0.4336
True: 0.9000, Predicted: 0.4458, Error: -0.4542
True: 0.0000, Predicted: 0.5023, Error: 0.5023
True: 0.1667, Predicted: 0.7209, Error: 0.5543
True: 0.9412, Predicted: 0.3652, Error: -0.5760
```

#### PPG：
```
Test Loss (MSE): 0.0151
RMSE: 0.1265
MAE: 0.0912
R² Score: 0.5779
95% Confidence Interval for Error: (np.float64(-0.2441793914332587), np.float64(0.2513741298026938))

Worst Predictions:
True: 0.1667, Predicted: 0.5522, Error: 0.3855
True: 0.1250, Predicted: 0.5139, Error: 0.3889
True: 0.8235, Predicted: 0.4011, Error: -0.4225
True: 0.7647, Predicted: 0.3284, Error: -0.4363
True: 0.8824, Predicted: 0.3601, Error: -0.5223
True: 0.0417, Predicted: 0.5777, Error: 0.5360
True: 0.8333, Predicted: 0.2715, Error: -0.5618
True: 0.9000, Predicted: 0.3211, Error: -0.5789
True: 1.0000, Predicted: 0.3780, Error: -0.6220
True: 0.0833, Predicted: 0.7269, Error: 0.6436
```

#### Temperature (with freq, no channing rate):
```
Test Loss (MSE): 0.0159
RMSE: 0.1266
MAE: 0.0922
R² Score: 0.5740
95% Confidence Interval for Error: (-0.24776952998291013, 0.24837447288955805)

Worst Predictions:
True: 0.0333, Predicted: 0.4238, Error: 0.3905
True: 0.9333, Predicted: 0.5336, Error: -0.3997
True: 0.9394, Predicted: 0.5322, Error: -0.4072
True: 0.7667, Predicted: 0.3344, Error: -0.4323
True: 0.8824, Predicted: 0.4461, Error: -0.4362
True: 0.8000, Predicted: 0.3486, Error: -0.4514
True: 0.7000, Predicted: 0.2425, Error: -0.4575
True: 0.0196, Predicted: 0.5063, Error: 0.4867
True: 0.9000, Predicted: 0.3509, Error: -0.5491
True: 1.0000, Predicted: 0.3559, Error: -0.6441
```
