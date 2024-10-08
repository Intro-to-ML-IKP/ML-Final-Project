# NeuralNetwork_project

Plan: 
build a feedforward network on stock data of the last n candlesticks
Input = n candlestick with high,low,open,close = 4n input neurons
Output is next k prices = next k closing values
Analyze the trend in post processing

Proposal 2:
Feedforward network on stock data of the last n candlesticks
Predict the next high,low,open,close of the subsequent candlestick
Use the prediction as the input for the next prediction -> make k predictions
Analyze the trend -> profit

Proposal 3:
Interpolation of the training data via later determined method
Subtract interpolation from real training data = residual
Train model on residual to predict next step
Iteratively predict next k steps this way by shifting window -> k residual predictions
Extrapolate trend and add predicted residual
Post process for buy/sell -> get some numerical measure of quality of the network i.e. cost
Generate a matrix for hyperparameters i.e. learning rate
Train the model on a set of markets(1 market==1 stock) with different hyperparameters
Analyze which hyperparamer sets perform best
Get rich quick


Objectives:
Find training parameters i.e. learning rate/weight initialization/loss func to find the best local minimum
Find ways to preprocess input data to optimize performance
Find a network architecture that maximizes performance without overfitting/extreme complexity