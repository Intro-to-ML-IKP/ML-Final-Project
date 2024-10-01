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

Objectives:
Find training parameters i.e. learning rate/weight initialization/loss func to find the best local minimum
Find ways to preprocess input data to optimize performance
Find a network architecture that maximizes performance without overfitting/extreme complexity