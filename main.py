from utils import print_nice_title
from network.parameterConstructor import ParameterConstructor
from network.network_constructor import NetworksConstructor, NetworksDict
from results.result_handler import ResultsHandler
from data_parser.dataFactory import StockDataFactory, DataReader
from forcast.forecastFactory_initializer import ForcastFactoryInitializer
from forcast.forcastFactory import ForcastFactory
from visualisation.visualize import PlotStocks


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
        testingPercentage: float = 0.8,
        validationPercentage: float = 0.1,
        maxEpochs: int = 50,
        save_param_list: bool = False
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
    :param testingPercentage: Percentage of data used for testing. 
        Default is 0.8.
    :type testingPercentage: float
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
    pConst = ParameterConstructor()

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
        testingPercentage,
        validationPercentage
        )
    
    # Get the data from the data factory
    (
        training_data,
        validation_data,
        testing_data,
        training_labels,
        validation_labels,
        testing_labels
        ) = dataFactory.get_stock_data()
    
    # Construct a network
    input_size = len(training_data[0])
    output_size = len(training_labels[0])
    netConst = NetworksConstructor(
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
        results_foldername
        )
    
    # Get the sorted results from the exploration as dictionary
    sorted_results = NetworksDict()(netConst.results)

    # Save the results
    result_hanler = ResultsHandler(sorted_results)
    result_hanler.save_results(results_filename, results_foldername)

    return sorted_results

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
        testing_percentage: float = 0.8,
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
    :param testing_percentage: Percentage of data used for testing. 
        Default is 0.8.
    :type testing_percentage: float
    :param validation_percentage: Percentage of data used for validation. 
        Default is 0.1.
    :type validation_percentage: float

    :return: A tuple containing the list of predicted closing prices and 
        the Mean Squared Error (MSE) between predictions and actual values.
    :rtype: tuple[list[float], float]
    """
    # Create a forcast factory initializer
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
        testing_percentage,
        validation_percentage
        )

    # Create a forcast factory
    forcaster = ForcastFactory(
        stock_name,
        model_parameters,
        datafactory_parameters
        )

    # Make the forcast
    forcaster.predict(
        number_of_predictions,
        raw_data_amount,
        sma_lookback_period,
        regression_window,
        end_date,
        interval
        )
    
    # Plot the forcast and the raw data
    forcaster.plot_predictions()

    # Compare the predictions with observations
    mse = forcaster.compare_predictions_with_observations()

    # Plot the observed data and the predicted data.
    forcaster.plot_comparison()

    predicted_closing_prices = forcaster.predicted_closing_prices

    return predicted_closing_prices, mse

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
    results = ResultsHandler()
    results.load_results(filename, foldername)

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

def plot_data():
    raw_data = DataReader("AAPL").getData(
        number_of_points=10,
        number_of_sets=50
    )
    plot_stocks = PlotStocks(raw_data)
    plot_stocks.plot_candlestick()

    

if __name__ == "__main__":
    pass
