from utils import print_nice_title # type: ignore
from network.parameterConstructor import ParameterConstructor, ParameterConstructorLSTM
from network.network_constructor import NetworksConstructor, NetworksDict
from results.result_handler import ResultsHandler
from data_parser.dataFactory import StockDataFactory, DataReader
from forecast.forecastFactory_initializer import ForcastFactoryInitializer
from forecast.forcastFactory import ForcastFactory
from forecast.forecastFactoryEnsemble import ForcastFactoryEnsemble
from visualisation.visualize import PlotStocks
from network.network import Model
from trend_model.base_model import LstmModel


def explore_different_architectures(
        stockCode: str,
        results_filename: str,
        results_foldername: str,
        maxLayers: int = 2,
        minNeurons: int = 4,
        maxNeurons: int = 16,
        dNeurons: int = 2,
        minLearningRate: float = 0.0005,
        maxLearningRate: float = 0.01,
        dLearningRate: float = 0.0005,
        minBatchSize: int = 1,
        maxBatchSize: int = 5,
        dBatchSize: int = 1,
        pointsPerSet: int = 10,
        numSets: int = 50,
        labelsPerSet: int = 1,
        trainingPercentage: float = 0.8,
        validationPercentage: float = 0.1,
        maxEpochs: int = 50,
        save_param_list: bool = False,
        lstm: bool = False,
        save_model: bool = False
        ) -> dict:
    """
    Explores different neural network architectures for a specified stock 
    code by varying network layers, neurons, learning rates, and batch 
    sizes. Saves the results and optionally the generated parameter list.

    :param stockCode: The stock code for retrieving datasets.
    :type stockCode: str
    :param results_filename: Name of the file to save the results.
    :type results_filename: str
    :param results_foldername: Folder where results will be saved.
    :type results_foldername: str
    :param maxLayers: Maximum number of hidden layers in the network. 
        Default is 2.
    :type maxLayers: int
    :param minNeurons: Minimum neurons per hidden layer. Default is 4.
    :type minNeurons: int
    :param maxNeurons: Maximum neurons per hidden layer. Default is 16.
    :type maxNeurons: int
    :param dNeurons: Step size for neurons per layer. Default is 2.
    :type dNeurons: int
    :param minLearningRate: Minimum learning rate. Default is 0.0005.
    :type minLearningRate: float
    :param maxLearningRate: Maximum learning rate. Default is 0.01.
    :type maxLearningRate: float
    :param dLearningRate: Step size for learning rate increments. 
        Default is 0.0005.
    :type dLearningRate: float
    :param minBatchSize: Minimum batch size. Default is 1.
    :type minBatchSize: int
    :param maxBatchSize: Maximum batch size. Default is 5.
    :type maxBatchSize: int
    :param dBatchSize: Step size for batch size increments. Default is 1.
    :type dBatchSize: int
    :param pointsPerSet: Number of data points per set. Default is 10.
    :type pointsPerSet: int
    :param numSets: Number of data sets to download. Default is 50.
    :type numSets: int
    :param labelsPerSet: Number of labels per set. Default is 1.
    :type labelsPerSet: int
    :param trainingPercentage: Percentage of data used for training. 
        Default is 0.8.
    :type trainingPercentage: float
    :param validationPercentage: Percentage of data used for validation. 
        Default is 0.1.
    :type validationPercentage: float
    :param maxEpochs: Maximum number of epochs for training each model. 
        Default is 50.
    :type maxEpochs: int
    :param save_param_list: If True, saves the parameter list to 
        'paramsList.txt'. Default is False.
    :type save_param_list: bool

    :return: A dictionary of sorted results from the network exploration.
    :rtype: dict
    """
    # Generate list of parameters
    if lstm is False:
        pConst = ParameterConstructor()
    else:
        pConst = ParameterConstructorLSTM()

    # Calculate possible permutations of architectures
    pConst.calcNetworkArchitectures(
        maxLayers,
        minNeurons,
        maxNeurons,
        dNeurons
        )
    
    # Calculate possible permutations of learning rates
    pConst.calcLearningRates(
        minLearningRate,
        maxLearningRate,
        dLearningRate
        )
    
    # Calculate possible permutations of batch sizes
    pConst.calcBatchSize(
        minBatchSize,
        maxBatchSize,
        dBatchSize
        )
    
    # Calculate possible permutations of the parameter list
    pConst.calcParamList()

    # Get the parameter list
    paramList = pConst.getParamList()

    #paramList = [tuple(([9, 20], 0.0050, 5)) for _ in range(100)]
    #paramList = [tuple(([10,8], 0.0050, 5)) for _ in range(200)]

    if save_param_list:
        with open("paramsList.txt", "w") as f:
            for params in paramList:
                f.write(f"{params}")
                f.write("\n")

    # Create a StockDataFactory
    dataFactory = StockDataFactory(
        stockCode,
        pointsPerSet,
        numSets,
        labelsPerSet,
        trainingPercentage,
        validationPercentage
        )
    
    # Get the data from the data factory
    if lstm is False:
        (
            training_data,
            validation_data,
            testing_data,
            training_labels,
            validation_labels,
            testing_labels
            ) = dataFactory.get_stock_data()
    else:
        (
            training_data,
            validation_data,
            testing_data,
            training_labels,
            validation_labels,
            testing_labels
            ) = dataFactory.get_stock_data(sma_data=True)
    
    # Construct a network
    input_size = len(training_data[0])
    output_size = len(training_labels[0])
    if lstm is False:
        netConst = NetworksConstructor(
            Model,
            input_size,
            output_size,
            maxEpochs)
    else:
        netConst = NetworksConstructor(
            LstmModel,
            input_size,
            output_size,
            maxEpochs)
    
    # Explore different model based on the generated parameters list
    netConst.explore_different_architectures(
        training_data,
        training_labels,
        validation_data,
        validation_labels,
        testing_data,
        testing_labels,
        paramList,
        results_filename,
        results_foldername,
        save_model
        )
    
    # Get the sorted results from the exploration as dictionary
    sorted_results = NetworksDict().sort_results_list(netConst.results)

    # Save the results
    result_handler = ResultsHandler(sorted_results, mae=False)
    result_handler.save_results(
        results_filename,
        results_foldername
        )

    result_handler.results = netConst.training_loss
    result_handler.save_results(
        f"{results_filename}_training_loss",
        results_foldername
        )

    result_handler.results = netConst.validation_loss
    result_handler.save_results(
        f"{results_filename}_validation_loss",
        results_foldername
        )

def forcast_closing_prices(
        stock_name: str = "AAPL",
        number_of_predictions: int = 5,
        raw_data_amount: int = 50,
        sma_lookback_period: int = 3,
        regression_window: int | None = None,
        end_date: str = "2024-09-01",
        interval: str = "1d",
        architecture: list[int] = [13, 24],
        learning_rate: int = 0.01,
        loss_function: str = "mse",
        metrics: list[str] = ["mae"],
        epochs: int = 50,
        batch_size: int = 5,
        points_per_set: int = 10,
        num_sets: int = 50,
        labels_per_set: int = 1,
        training_percentage: float = 0.8,
        validation_percentage: float = 0.1
        ) -> tuple[list[float], float]:
    """
    Forecasts closing prices for a specified stock using a neural network 
    model. Generates data, trains the model, and plots predictions 
    against observed values.

    :param stock_name: Name of the stock to forecast.
    :type stock_name: str
    :param number_of_predictions: Number of closing price predictions to 
        generate.
    :type number_of_predictions: int
    :param raw_data_amount: Amount of raw historical data to retrieve. 
        Default is 50.
    :type raw_data_amount: int
    :param sma_lookback_period: Lookback period for calculating the Simple 
        Moving Average (SMA). Default is 3.
    :type sma_lookback_period: int
    :param regression_window: Optional window size for regression. 
        Default is None.
    :type regression_window: int or None
    :param end_date: End date for historical data retrieval in "YYYY-MM-DD" 
        format. Default is "2024-09-01".
    :type end_date: str
    :param interval: Interval for data points, e.g., "1d" for daily. 
        Default is "1d".
    :type interval: str
    :param architecture: List specifying the number of neurons in each 
        network layer. Default is [13, 24].
    :type architecture: list[int]
    :param learning_rate: Learning rate for model training. Default is 0.01.
    :type learning_rate: float
    :param loss_function: Loss function for model training, e.g., "mse" 
        (Mean Squared Error). Default is "mse".
    :type loss_function: str
    :param metrics: List of metrics to monitor during training, e.g., ["mae"].
    :type metrics: list[str]
    :param epochs: Number of epochs for training the model. Default is 50.
    :type epochs: int
    :param batch_size: Batch size for model training. Default is 5.
    :type batch_size: int
    :param points_per_set: Number of data points per set in the training data. 
        Default is 10.
    :type points_per_set: int
    :param num_sets: Number of sets of data to generate. Default is 50.
    :type num_sets: int
    :param labels_per_set: Number of labels per data set. Default is 1.
    :type labels_per_set: int
    :param training_percentage: Percentage of data used for testing. 
        Default is 0.8.
    :type training_percentage: float
    :param validation_percentage: Percentage of data used for validation. 
        Default is 0.1.
    :type validation_percentage: float

    :return: A tuple containing the list of predicted closing prices and 
        the Mean Squared Error (MSE) between predictions and actual values.
    :rtype: tuple[list[float], float]
    """
    # Create a forecast factory initializer
    param_getter = ForcastFactoryInitializer()

    # Generate model parameters
    model_parameters = param_getter.generate_model_parameters(
        architecture,
        learning_rate,
        loss_function,
        metrics,
        epochs,
        batch_size
        )
    
    # Generate datafacotry parameters
    datafactory_parameters = param_getter.generate_datafactory_parameters(
        points_per_set,
        num_sets,
        labels_per_set,
        training_percentage,
        validation_percentage
        )

    # Create a forecast factory
    forcaster = ForcastFactory(
        stock_name,
        model_parameters,
        datafactory_parameters
        )

    # Make the forecast
    forcaster.predict(
        number_of_predictions,
        raw_data_amount,
        sma_lookback_period,
        regression_window,
        end_date,
        interval
        )
    
    # Plot the forecast and the raw data
    forcaster.plot_predictions()

    # Compare the predictions with observations
    mse = forcaster.compare_predictions_with_observations()

    # Plot the observed data and the predicted data.
    forcaster.plot_comparison()

    predicted_closing_prices = forcaster.predicted_closing_prices

    return predicted_closing_prices, mse

def test_ensemble_model(
        stock_name: str = "AAPL",
        number_of_predictions: int = 5,
        raw_data_amount: int = 90,
        sma_lookback_period: int = 3,
        regression_window: int | None = None,
        end_date: str = "2025-01-15",
        interval: str = "1d",
        points_per_set: int = 10,
        num_sets: int = 50,
        labels_per_set: int = 1,
        training_percentage: float = 0.8,
        validation_percentage: float = 0.1
        ) -> tuple[list[float], float]:
    """
    Forecasts closing prices for a specified stock using a neural network 
    model. Generates data, trains the model, and plots predictions 
    against observed values.

    :param stock_name: Name of the stock to forecast.
    :type stock_name: str
    :param number_of_predictions: Number of closing price predictions to 
        generate.
    :type number_of_predictions: int
    :param raw_data_amount: Amount of raw historical data to retrieve. 
        Default is 50.
    :type raw_data_amount: int
    :param sma_lookback_period: Lookback period for calculating the Simple 
        Moving Average (SMA). Default is 3.
    :type sma_lookback_period: int
    :param regression_window: Optional window size for regression. 
        Default is None.
    :type regression_window: int or None
    :param end_date: End date for historical data retrieval in "YYYY-MM-DD" 
        format. Default is "2024-09-01".
    :type end_date: str
    :param interval: Interval for data points, e.g., "1d" for daily. 
        Default is "1d".
    :type interval: str
    :param architecture: List specifying the number of neurons in each 
        network layer. Default is [13, 24].
    :type architecture: list[int]
    :param learning_rate: Learning rate for model training. Default is 0.01.
    :type learning_rate: float
    :param loss_function: Loss function for model training, e.g., "mse" 
        (Mean Squared Error). Default is "mse".
    :type loss_function: str
    :param metrics: List of metrics to monitor during training, e.g., ["mae"].
    :type metrics: list[str]
    :param epochs: Number of epochs for training the model. Default is 50.
    :type epochs: int
    :param batch_size: Batch size for model training. Default is 5.
    :type batch_size: int
    :param points_per_set: Number of data points per set in the training data. 
        Default is 10.
    :type points_per_set: int
    :param num_sets: Number of sets of data to generate. Default is 50.
    :type num_sets: int
    :param labels_per_set: Number of labels per data set. Default is 1.
    :type labels_per_set: int
    :param training_percentage: Percentage of data used for testing. 
        Default is 0.8.
    :type training_percentage: float
    :param validation_percentage: Percentage of data used for validation. 
        Default is 0.1.
    :type validation_percentage: float

    :return: A tuple containing the list of predicted closing prices and 
        the Mean Squared Error (MSE) between predictions and actual values.
    :rtype: tuple[list[float], float]
    """
    # Create a forecast factory initializer
    param_getter = ForcastFactoryInitializer()
    
    # Generate datafacotry parameters
    datafactory_parameters = param_getter.generate_datafactory_parameters(
        points_per_set,
        num_sets,
        labels_per_set,
        training_percentage,
        validation_percentage
        )

    # Create a forecast factory
    forcaster = ForcastFactoryEnsemble(
        stock_name=stock_name,
        residual_model="AAPL_145_model.keras",
        residual_model_folder="early_stoppage",
        trend_model="test",
        trend_model_folder="test",
        datafactory_param_dict=datafactory_parameters
        )

    # Make the forecast
    forcaster.predict(
        number_of_predictions,
        raw_data_amount,
        sma_lookback_period,
        regression_window,
        end_date,
        interval
        )

def perform_statistical_analysis(filename: str, foldername: str) -> None:
    """
    Performs statistical analysis on the results stored in a specified 
    file and folder. This includes calculating correlation coefficients, 
    p-values, conducting regression analysis, and generating visualizations 
    such as scatterplot matrices and correlation heatmaps. It also prints
    the top 5 NNs.

    :param filename: The name of the file containing the results to analyze.
    :type filename: str
    :param foldername: The folder where the results file is located.
    :type foldername: str
    """
    # Load the results
    results = ResultsHandler(mae=False)
    results.load_results(filename, foldername)
    print(len(results.results))

    # Define the title
    corr_coef_title = "Correlation Coefficients"
    p_val_title = "P-Values"
    reg_analysis = "Regression Analysis"
    par_ranges = "Parameter ranges"
    top_5_networks = "Top 5 NNs"

    # Calculate the correlation coeficients and p-values
    print_nice_title(corr_coef_title)
    mae_correlations, p_values = results.calculate_correlation_coefficients()
    print(mae_correlations)

    # Print the p-values
    print_nice_title(p_val_title)
    print(p_values)

    # Perform a regression analysis
    print_nice_title(reg_analysis)
    print(results.perform_regression_analysis())

    # Create a scatterplot matrix
    results.create_scatterplot_matrix()

    # Create a correlation heatmap
    results.create_correlation_heatmap()

    # Get the parameter's ranges
    print_nice_title(par_ranges)
    param_ranges = results.get_parmeter_ranges()
    print(param_ranges)

    # Get the top 5 NNs
    print_nice_title(top_5_networks)
    print(results.df[:5])

def plot_data(
        number_of_points: int = 10,  # Value used by final results.
        number_of_sets: int = 50  # Dito.
):
    raw_data = DataReader("AAPL").getData(
        number_of_points,
        number_of_sets
    )
    plot_stocks = PlotStocks(raw_data)
    plot_stocks.plot_candlestick()

def plot_train_val_losses(filename: str, folder: str, id: int|None = None):
    import os
    import pickle
    import matplotlib.pyplot as plt
    import numpy as np

    def adjust_to_median_length(data, median_length):
        """
        Adjusts all sequences in the data to the median length:
        - Shorter sequences are padded with their last value.
        - Longer sequences are truncated.
        """
        adjusted_data = []
        for seq in data:
            seq = list(seq)  # Ensure sequence is mutable
            if len(seq) < median_length:
                print("padded")
                # Pad with the last value
                seq.extend([seq[-1]] * (median_length - len(seq)))
            else:
                print(f"truncated: {len(seq)} to {median_length}")
                # Truncate to median length
                seq = seq[:median_length]
            adjusted_data.append(seq)
        return np.array(adjusted_data)

    filename_train = f"results\\{folder}\\{filename}_training_loss"
    filename_val = f"results\\{folder}\\{filename}_validation_loss"

    # Load the data from the files
    if os.path.exists(filename_train):
        with open(filename_train, "rb") as file:
            training_loss_data_ = pickle.load(file)
    training_loss_data = [tup[1] for tup in training_loss_data_]

    if os.path.exists(filename_val):
        with open(filename_val, "rb") as file:
            validation_loss_data_ = pickle.load(file)
    validation_loss_data = [tup[1] for tup in validation_loss_data_]

    if id is not None:
        for count in range(len(training_loss_data_)):
            training_id, training_loss = training_loss_data_[count]
            validation_id, validation_loss = validation_loss_data_[count]

            if training_id == id:
                training_loss_data = training_loss

            if validation_id == id:
                validation_loss_data = validation_loss
            
    if id is not None:
        plt.plot(validation_loss_data, label='Validation Loss', color='red')
        plt.plot(training_loss_data, label="Trainign Loss", color="blue")

        # Labels and title
        plt.title(f'Training and Validation Loss of model id: {id}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    else:
        # Determine the median length of all sequences
        all_lengths = [len(seq) for seq in (training_loss_data + validation_loss_data)]
        median_length = int(np.median(all_lengths))

        min_val_loss = 10
        min_val_loss_index = 0
        for index, dat in enumerate(validation_loss_data):
            dat_min = np.array(dat).min()
            if dat_min < min_val_loss:
                min_val_loss = dat_min
                min_val_loss_index = index

        biggests_sigma_val_loss = validation_loss_data[min_val_loss_index]
        biggests_sigma_train_loss = training_loss_data[min_val_loss_index]

        for index in range(len(validation_loss_data)):
            plt.plot(validation_loss_data[index], label='Validation Loss', color='red')
            plt.plot(training_loss_data[index], label="Trainign Loss", color="blue")
        plt.show()
        # Adjust sequences to the median length
        training_loss_data = adjust_to_median_length(training_loss_data, median_length)
        validation_loss_data = adjust_to_median_length(validation_loss_data, median_length)


        # Calculate average and standard deviation
        avg_train_loss = np.mean(training_loss_data, axis=0)
        avg_val_loss = np.mean(validation_loss_data, axis=0)
        std_train_loss = np.std(training_loss_data, axis=0)
        std_val_loss = np.std(validation_loss_data, axis=0)

        # Plotting
        epochs = range(1, median_length + 1)
        plt.figure(figsize=(10, 6))

        # Plot the average training and validation loss
        plt.plot(epochs, avg_train_loss, label='Average Training Loss', color='blue')
        plt.plot(epochs, avg_val_loss, label='Average Validation Loss', color='red')

        plt.plot(biggests_sigma_train_loss, label="Best performance training loss", color="green")
        plt.plot(biggests_sigma_val_loss, label="Best performance validation loss", color="yellow")

        # Add shaded areas for the variance (standard deviation)
        plt.fill_between(epochs, avg_train_loss - std_train_loss, avg_train_loss + std_train_loss, color='blue', alpha=0.2)
        plt.fill_between(epochs, avg_val_loss - std_val_loss, avg_val_loss + std_val_loss, color='red', alpha=0.2)

        # Add a vertical line at the median length
        #plt.axvline(x=median_length, color='green', linestyle='--', label=f'Median Length ({median_length})')

        # Labels and title
        plt.title('Average Training and Validation Loss with Variance')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Show the plot
        plt.show()


if __name__ == "__main__":
    #test_ensemble_model()
    #explore_different_architectures("AAPL", "lstm_test", "lstm_test", maxEpochs=100)
    #explore_different_architectures("AAPL", "lstm_test", "lstm_test", maxEpochs=100, lstm=True, minNeurons=9, maxNeurons=81, dNeurons=3)
    plot_train_val_losses("lstm_test", "lstm_test")
    #perform_statistical_analysis("lstm_test", "lstm_test")
    #from trend_model.test_run import run
    #run()