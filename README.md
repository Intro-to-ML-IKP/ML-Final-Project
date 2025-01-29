# Ensemble Model for Stock Price Prediction

## Overview
This project implements an ensemble model combining a **Multilayer Perceptron (MLP)** and a **Long Short-Term Memory (LSTM) Recurrent Neural Network (RNN)** to predict stock closing prices. The MLP predicts residuals (differences between stock prices and a trendline derived from a Simple Moving Average, SMA), while the LSTM extrapolates the SMA to model underlying trends. The model is trained and evaluated using stock data from Apple Inc.

## Authors
- **Iva I. Ivanova** (s5614260)
- **Katya T. Toncheva** (s5460786)
- **Petar I. Penchev** (s4683099)

## Dataset
The dataset is sourced from **Yahoo Finance** via the `yfinance` Python library. It includes **500 daily candlesticks** of Apple Inc.'s stock prices, covering the period from **March 21, 2021, to January 15, 2025**.

## Model Architecture
### Multilayer Perceptron (MLP)
- Predicts residuals (closing price minus SMA).
- Trained using 5,600 configurations with hyperparameter tuning.
- Best architecture: **Two hidden layers (10 and 8 neurons), learning rate 0.005, batch size 5**.

### Long Short-Term Memory (LSTM)
- Models the trend by predicting the next SMA value.
- Trained using 5,000 configurations.
- Best architecture: **Single LSTM layer with 18 neurons, learning rate 0.0025, batch size 1**.

### Ensemble Model
- Combines MLP residual predictions and LSTM trend predictions.
- Achieved a **Mean Absolute Error (MAE) of 2.14 USD** on test data.

## Preventing Overfitting
- **L2 Regularization** applied to MLP.
- **Normalization** used for LSTM.
- **Early Stopping** (4-epoch patience) implemented.

## Hyperparameter Tuning
Both models underwent extensive hyperparameter tuning to find optimal configurations. The hyperparameters tuned include:
- **Number of hidden layers and neurons** (for MLP and LSTM).
- **Learning rate**.
- **Batch size**.

## Results
| Model  | Best Architecture | MAE |
|--------|------------------|-----|
| MLP    | 10-8 neurons, LR 0.005, batch size 5 | 0.21 |
| LSTM   | 18 neurons, LR 0.0025, batch size 1  | 5.20 |
| Ensemble | MLP + LSTM | **2.14** |

## Conclusion
The ensemble approach successfully leverages both models to improve prediction accuracy. While the MLP struggled with residual prediction due to the randomness in stock prices, the LSTM effectively captured the trend. Future work can explore additional financial indicators like **Bollinger Bands, trading volume, and long-term trends** to further enhance performance.

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/Intro-to-ML-IKP/ML-Final-Project.git
   cd ML-Final-Project
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the main script:
   ```sh
   python main.py
   ```

## References
- **Yahoo Finance API** (`yfinance` library)
- Hornik et al. (1989) - *MLPs as Universal Function Approximators*
- He et al. (2024) - *Regression and Ensemble Models for Financial Forecasting*

## License
This project is licensed under the MIT License.

