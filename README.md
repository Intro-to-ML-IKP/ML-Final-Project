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

## Structure
```plaintext
ML-Final-Project/
│── data_parser/                          # Directory for handling and processing data
│   ├── __init__.py                       # Initializes the package
│   ├── dataFactory.py                    # Orchestrates the whole data pipeline
│   ├── dataProcessor.py                  # Cleans and preprocesses the raw data
│   ├── dataReader.py                     # Reads the preprocessed data for use in training
│   ├── stockGetter.py                    # Extracts stock-related data from Yahoo Finance API
│
│── forcast/                               # Directory for model forecasting and results
│   ├── __init__.py                       # Initializes the package
│   ├── ensembleModel.py                  # Combines predictions from MLP and LSTM model
│   ├── forcastFactory.py                 # Factory for creating forecasts using only an MLP and Linear Regression. Orchestrates the forcasting pipeline
│   ├── forcastFactoryEnsemble.py         # Factory for creates ensemble forecast model. Orchestrates the forcasting pipeline of the ensemble model
│   ├── forcastFactory_initializer.py     # Initializes and sets up the forecast factories
│
│── graphs_for_project/                   # Contains graphs used in the paper
│
│── models/                                # The folder containing the saved models
│
│── network/                               # Source code for defining and training neural network models
│   ├── __init__.py                       # Initializes the package
│   ├── network.py                        # Defines the neural network architecture and training logic
│   ├── networkFactory.py                 # Creates a neural network
│   ├── network_constructor.py            # Constructs different types of neural networks
│   ├── parameterConstructor.py           # Defines MLP (Multi-Layer Perceptron) and LSTM (Long Short-Term Memory) model architectures
│
│── results/                               # Stores the output, evaluation metrics, and visualizations of model performance
│
│── trend_model/                           # Source code for trend prediction models (LSTM)
│   ├── __init__.py                       # Initializes the package
│   ├── base_model.py                     # Defines the basic model architecture and training procedure of an LSTM
│
│── visualisation/                         # Source code for visualizing results, predictions and forcasts
│   ├── __init__.py                       # Initializes the package
│   ├── vizualize.py                      # Contains logic for generating visualizations (e.g., plots, graphs) of results
│
│── LICENSE                                # MIT License for the project
│── README.md                              # Project documentation, describing the purpose and usage
│── requirements.txt                       # List of dependencies required to run the project
│── .gitignore                             # Files and directories to exclude from git tracking
│── environment.yml                        # Environment file
│── main.py                                # The main script
│── utils.py                               # Contains simple utitils scripts
```

## Notable contributions
This is a continuation on a previous work done by Thomas Smeman, Gideon Jadael Wiersma, and Petar I. Penchev.
## References
- **Yahoo Finance API** (`yfinance` library)
- Hornik et al. (1989) - *MLPs as Universal Function Approximators*
- He et al. (2024) - *Regression and Ensemble Models for Financial Forecasting*

## License
This project is licensed under the MIT License.

