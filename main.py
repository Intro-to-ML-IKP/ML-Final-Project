# Main program
from dataReader import DataReader
from visualize import plot_candlestick

def main():
    dR = DataReader("AAPL")             # Initialize for AAPL stock
    stock_data = dR.getData(10, 5)      # Download sufficient data for 5 sets of 10 datapoints
    data, labels = dR.getLabels(10, 3)  # Retrieve the labels of the next 3 datapoints for each set of 10 points, splitting data from labels
    data = dR.preprocess()              # Apply preproccesing on data if applicable
    # Apply a train, test, validation split on the data where 80% of the data is for training, 10% for validation and the rest for testing
    training_data, training_labels, validation_data, validation_labels, testing_data, testing_labels = dR.split_data(0.8, 0.1)

    print(training_data, training_labels, validation_data, validation_labels, testing_data, testing_labels)
    
    

if __name__ == "__main__":
    main()