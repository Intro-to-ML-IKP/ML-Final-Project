from network.network import Model
import os
import pickle
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import gc
from tensorflow.keras import backend as K # type: ignore
from sklearn.linear_model import LinearRegression
from multiprocessing import Pool
from scipy.stats import pearsonr


LIST_BB = list(range(1000, 1000001, 1000))

class NetworkConstructor:
    results = []

    def __init__(self, input_size: int, output_size: int, epochs: int):
        """Initializes parameters used throughout the class
        
        Parameters:
        input_size      (int)   - Number of datapoints in input data
        output_size     (int)   - Number of points in output data
        epochs          (int)   - Maximum number of epochs used in training"""
        self.input_size = input_size
        self.output_size = output_size
        self.epochs = epochs
    
    def build_model(
            self,
            architecture: list[int],
            activations: list[str],
            learning_rate: float = 0.001,
            lossFunc: str = "mse",
            metrics: list[str] = ["mae"]
            ) -> Model:
        # self._layers_activations_compatability(neurons_per_layer, activations)        # Idk why this is here 
        # neurons_per_layer = neurons_per_layer
        # neurons_per_layer.append(self.output_shape)
        # self.model.create_sequential_model(neurons_per_layer, activations, self.input_shape)
        # self.model.compileModel(learning_rate, lossFunc, metrics)

        # architecture.append(self.output_size)
        model = Model()
        model.create_sequential_model(architecture, activations, self.input_size, self.output_size)
        model.compileModel(learning_rate, lossFunc, metrics)
        return model

    def helper(self, params):
        paramSet, count, maxCount, training_data, training_labels, validation_data, validation_labels, testing_data, testing_labels = params
        architecture, learning_rate, batch_size = paramSet

        # Create list of activation functions for the network, which will be relu for all but the output layer
        activations = ["relu" for _ in range(len(architecture))]    
        activations.append("linear")

        # Create and train the model
        self.model = self.build_model(architecture=architecture, activations=activations, learning_rate=learning_rate)
        self.model.trainModel(training_data, training_labels, validation_data, validation_labels, self.epochs, batch_size)

        # Evaluate the model
        mae = self.model.compute_mae(testing_data, testing_labels)
        self.results.append([mae, paramSet])

        # Clear Keras session and free memory
        K.clear_session()
        del self.model
        gc.collect()  # Force garbage collection

        print(f"Now training the model {count}/{maxCount}")

        if count in LIST_BB:
            maes = NetworksDict()
            results_handler = ResultsHandler(maes)
            results_handler.save_results(f"NN_results_{count}")

    def explore_different_architectures(
        self,
        training_data: list[float],
        training_labels: list[float],
        validation_data: list[float],
        validation_labels: list[float],
        testing_data: list[float],
        testing_labels: list[float],
        paramList: list[tuple[list[int], float, int]],
    ) -> list[list[float, tuple[list[int], float, int]]]:
        """Makes a list of sets of parameters from which different networks can be trained
        
        Parameters:
        training_data           - Data used to train the model for optimal performance
        training_labels         - Preferred outcomes for training
        validation_data         - Data used to prevent overfitting of the data
        validation_labels       - Labels used for preventing overfitting
        testing_data            - Data used to evaluate the model performance
        testing_labels          - Labels for evaluating model performance
        paramList               (list[tuple[list[int], float, int]])    - Precomputed list of parameters from which networks will be built and trained. Consists of:
            architectures       (list[int]])                            - The amount of neurons in each layer of the network
            learning_rates      (float)                                 - The step size scaling used in training
            batch_sizes         (int)                                   - The number of datapoints to process before updating the network in training
        
        Returns:
        allErrors   (list[list[float, tuple[list[int], float, int]]])   - All maes of every parameter combination with the accompanying parameter combination"""
        fullParamList = [(paramSet, count, len(paramList), training_data, training_labels, validation_data, validation_labels, testing_data, testing_labels) for count,paramSet in enumerate(paramList)]
        with Pool(processes=25) as p:
            p.map(self.helper, fullParamList)

    # def explore_different_architectures(
    #     self,
    #     training_data: list[float],
    #     training_labels: list[float],
    #     validation_data: list[float],
    #     validation_labels: list[float],
    #     testing_data: list[float],
    #     testing_labels: list[float],
    #     paramList: list[tuple[list[int], float, int]],
    # ) -> list[list[float, tuple[list[int], float, int]]]:
    #     """Makes a list of sets of parameters from which different networks can be trained
        
    #     Parameters:
    #     training_data           - Data used to train the model for optimal performance
    #     training_labels         - Preferred outcomes for training
    #     validation_data         - Data used to prevent overfitting of the data
    #     validation_labels       - Labels used for preventing overfitting
    #     testing_data            - Data used to evaluate the model performance
    #     testing_labels          - Labels for evaluating model performance
    #     paramList               (list[tuple[list[int], float, int]])    - Precomputed list of parameters from which networks will be built and trained. Consists of:
    #         architectures       (list[int]])                            - The amount of neurons in each layer of the network
    #         learning_rates      (float)                                 - The step size scaling used in training
    #         batch_sizes         (int)                                   - The number of datapoints to process before updating the network in training
        
    #     Returns:
    #     allErrors   (list[list[float, tuple[list[int], float, int]]])   - All maes of every parameter combination with the accompanying parameter combination"""
    #     count = 0
    #     for paramSet in paramList:
    #         architecture, learning_rate, batch_size = paramSet

    #         # Create list of activation functions for the network, which will be relu for all but the output layer
    #         activations = ["relu" for _ in range(len(architecture))]    
    #         activations.append("linear")

    #         # Create and train the model
    #         self.model = self.build_model(architecture=architecture, activations=activations, learning_rate=learning_rate)
    #         self.model.trainModel(training_data, training_labels, validation_data, validation_labels, self.epochs, batch_size)

    #         # Evaluate the model
    #         mae = self.model.compute_mae(testing_data, testing_labels)
    #         self.results.append([mae, paramSet])

    #         # Clear Keras session and free memory
    #         K.clear_session()
    #         del self.model
    #         gc.collect()  # Force garbage collection
            
    #         count += 1

    #         print(f"Now training the {count}th model, {9300-count} more to go.")

    #         if count in LIST_BB:
    #             maes = NetworksDict()
    #             results_handler = ResultsHandler(maes)
    #             results_handler.save_results(f"NN_results_{count}")


class NetworksDictMeta(type):
    """
    This class is the base metaclass for NetworksDict.
    """
    def __call__(cls, *args, **kwargs) -> dict:
        """
        Allows the class to be called like a function.
        """
        sorted_results = cls._sort_results()
        for mae, params in sorted_results.items():
            print(f"MAE: {mae}, Hidden Layers: {params[0]}; Learning Rate: {params[1]}; Batch Size: {params[2]}")
        return sorted_results
    
class NetworksDict(metaclass=NetworksDictMeta):
    """
    This metaclass is resposible for accessing the NetworkConstructor's attribute
    results, convert the list to a dictionarry and orders it. It also prints a nice
    human readable overview of the parameters of the Networks.

    :call: Prints a human readable sorted NNs with their parameters.
    :return: A sorted dictionary of the results parameter in the class
    NetworkConstructor.
    """
    @classmethod
    def _list_to_dict(self) -> dict:
        """
        Gets the NetworkConstructor's results and makes them into a dictionary.
        """
        nnDict = {}

        # Access the result list in NetworkConstructor
        nn_results = NetworkConstructor.results

        # Create a dictionary with MAEs as keys and parameters as values
        for mae, params in nn_results:
            nnDict[mae] = params  # Store params associated with each MAE
        return nnDict

    @classmethod
    def _sort_results(self) -> dict:
        """
        Sort the results by the mae (smallest to largest)
        """
        nnDict = self._list_to_dict()
        sorted_keys = sorted(nnDict.keys())
        sorted_results = {key: nnDict[key] for key in sorted_keys}
        return sorted_results
    

class ResultsHandler:
    """
    This class is used to handle saving and loading of results.
    """
    def __init__(self, results: dict = None):
        self._results = results
        self._df = None

    @property
    def results(self) -> dict:
        """
        Used to retrieve the results.

        :return: the results for the different networks
        :rtype: dict
        """
        return self._results

    def save_results(self, filename: str):
        """
        Saving the dictionary to a file using pickle.

        :param results: a dictionary of results
        :results type: dict
        :param filename: the name of the file to be saved
        :filename type: str
        """
        filename = "network\\" + filename + ".pkl"
        if self._results is not None:
            if os.path.exists(filename):
                self._options_if_file_exists(filename)
            else:
                with open(filename, "wb") as file:
                    pickle.dump(self._results, file)
                print(f"Results saved successfully in dir `{filename}.pkl`.")
        else:
            print("There aren't any results to save. The saving was unsuccessful.")

    def load_results(self, filename: str):
        """
        Loading the dictionary from a pickle file.

        :param filename: the name of the file to be loaded
        :filename type: str
        """
        filename = "network\\" + filename + ".pkl"
        if os.path.exists(filename):
            with open(filename, "rb") as file:
                loaded_data = pickle.load(file)
            self._results = loaded_data
            self._df = self._df
            print("Results loaded successfully!")
        else:
            print(f"There is no dir `{filename}` found")

    def load_multiple_results(self, filenames: list[str]):
        results = pd.DataFrame()
        for filename in filenames:
            self.load_results(filename)
            df2 = self._generate_pd_dataframe()
            self._results = None
            results = pd.concat([results, df2], ignore_index=True)
        print(results, type(results))
        self._df = results
        print(self._results, type(self._results))

    def calculate_correlation_coefficients(self):
        """
        Calculates the correlation coefficients between the parameters and MAE.
        """
        df = self._df

        # Calculate the correlation matrix
        correlation_matrix = df.corr()

        # Calculate p-values for each pair of features
        p_values = pd.DataFrame(index=df.columns, columns=df.columns)

        for col1 in df.columns:
            for col2 in df.columns:
                if col1 != col2:
                    # Calculate the p-value
                    _, p_val = pearsonr(df[col1], df[col2])
                    p_values.loc[col1, col2] = p_val
                else:
                    p_values.loc[col1, col2] = None  # No p-value for self-correlation

        # Focusing on the correlation of parameters with MAE
        mae_correlations = correlation_matrix["MAES"].drop("MAES")

        return mae_correlations, p_values
    
    def perform_regression_analysis(self):
        """
        Performs regression analysis by examining the relationship
        between the MAE and the rest of the parameters.
        """
        df = self._df

        # Prepare the data
        param_space = df[["Neurons Layer 1", "Neurons Layer 2", "Learning Rate", "Batch Size", "Number of Layers"]]  # Features
        target_param = df["MAES"]  # Target

        # Fit the model
        model = LinearRegression()
        model.fit(param_space, target_param)

        # Get coefficients
        coefficients = model.coef_
        param_importance = pd.DataFrame(coefficients, index=param_space.columns, columns=['Coefficient'])

        return param_importance
    
    def get_parmeter_ranges(self):
        neuron_layer1 = self._get_min_max_parameter("Neurons Layer 1")
        neuron_layer2 = self._get_min_max_parameter("Neurons Layer 2")
        learning_rate = self._get_min_max_parameter("Learning Rate")
        batch_size = self._get_min_max_parameter("Batch Size")
        number_of_layers = self._get_min_max_parameter("Number of Layers")
        maes = self._get_min_max_parameter("MAES")

        data = {
            "Neurons Layer 1": neuron_layer1,
            "Neurons Layer 2": neuron_layer2,
            "Learning Rate": learning_rate,
            "Batch Size": batch_size,
            "Number of Layers": number_of_layers,
            "MAES": maes
            }
        
        df = pd.DataFrame(data)
        df = df.rename(index={0: "Max Value", 1: "Min Value", 2: "Expected Value (Mean)"})

        return df

    
    def _get_min_max_parameter(self, parameters: str) -> list[float,float]:
        df = self._df
        max_val = df[parameters].max()
        min_val = df[parameters].min()
        expected_val = df[parameters].mean()
        return [max_val, min_val, expected_val]

    def create_scatterplot_matrix(self):
        """
        Creates a scatterplot matrix and plots it.
        """
        # Generate the DataFrame
        df = self._df

        # Create a scatterplot matrix using seaborn's pairplot
        # Using fill=True to avoid the deprecated shade warning
        pairplot = sns.pairplot(
            df,                            # DataFrame to plot
            diag_kind='kde',              # Diagonal kind: 'hist' for histograms, 'kde' for KDE plots
            markers='o',                  # Marker style for scatter plots
            height=2.5,                   # Height of each facet
            aspect=1.2,                   # Aspect ratio
            plot_kws={'alpha': 0.6},      # Scatter plot transparency
            diag_kws={'fill': True}       # Use fill=True for KDE plots
        )

        # Set the overall title for the pairplot (this sets the title for the entire figure)
        plt.suptitle("Scatterplot Matrix", fontsize=12, y=1.02)  # Adjust y for title spacing
        
        # Show the plot
        plt.show()


    def create_correlation_heatmap(self):
        """
        Creates a correlation heatmap and plots it.
        """
        # Generate the DataFrame
        df = self._df

        # Compute the correlation matrix
        corr_matrix = df.corr()

        # Set the figure size for better visibility
        plt.figure(figsize=(10, 8))

        # Create a heatmap using seaborn with improved aesthetics
        sns.heatmap(
            corr_matrix,                  # Data for the heatmap
            annot=True,                   # Annotate cells with the correlation values
            fmt=".2f",                   # Format of the annotation
            cmap='coolwarm',             # Color map
            center=0,                    # Center the colormap at 0
            square=True,                 # Square cells
            linewidths=0.5,             # Line width between cells
            linecolor='white',           # Color of lines between cells
            cbar_kws={"shrink": .8},    # Colorbar size
            annot_kws={"size": 10}       # Font size of annotations
        )

        # Improve layout
        plt.title("Correlation Heatmap", fontsize=16)  # Title of the heatmap
        plt.xticks(rotation=45, ha='right', fontsize=12)  # Rotate x-axis labels for better readability
        plt.yticks(fontsize=12)                           # y-axis label size

        # Show the plot
        plt.tight_layout()  # Adjust layout to make room for the title and labels
        plt.show()

    def _options_if_file_exists(self, filename: str):
        """
        This class method is used if you are tring to save a file with a name
        that already exists in the directory. It prompts the user for actions.

        :param filename: the name of the file
        :type filename: str
        """
        user_input = input(
            f"File with the name `{filename}` already exists!\n"
            "Do you want to overwrite its contents? (y/n)\n"
            )
        if user_input == "y":
            with open(f"network\\{filename}", "wb") as file:
                pickle.dump(self._results, file)
            print(f"The file `{filename}.pkl` has been overwriten with the current results.") 
        elif user_input == "n":
            user_input = input("Chose a different name without any `.suffix` at the end:\n")
            return self.save_results(str(user_input))
        else:
            return self.save_results(filename)
    
    # def _generate_pd_dataframe(self):
    #     # Get the data from the results
    #     maes = list(self._results.keys())
    #     values = list(self._results.values())
        
    #     # Create a new list with the modified tuples including the number of layers
    #     parameters = [(t[0], t[1], t[2], len(t[0])) for t in values]

    #     # Create a DataFrame with separate columns for keys and values
    #     df = pd.DataFrame(parameters, columns=["Neurons", "Learning Rate", "Batch Size", "Number of Layers"])
    #     df["MAES"] = maes
    #     return df

    def _generate_pd_dataframe(self):
        """
        Generates a pd dataFrame from the results of the NN training.
        Puts each parameter into its own key.
        """
        # Get the data from the results
        maes = list(self._results.keys())
        values = list(self._results.values())

        rows = []

        for tup in values:
            # Create a dictionary to hold the parameters for the current row
            row_data = {}
            
            # Loop through the number of neurons for each layer
            for count, neurons in enumerate(tup[0]):
                row_data[f"Neurons Layer {count + 1}"] = neurons

            # Ensure that at least two layers are represented
            num_layers = len(tup[0])
            if num_layers < 2:
                for i in range(2 - num_layers):  # Fill missing layers with 0 neurons
                    row_data[f"Neurons Layer {num_layers + 1 + i}"] = 0

            # Add the remaining parameters to the dictionary
            row_data["Learning Rate"] = tup[1]
            row_data["Batch Size"] = tup[2]
            row_data["Number of Layers"] = num_layers
            row_data["MAES"] = maes.pop(0)  # Get the corresponding MAES for this row

            # Append the row data to the list of rows
            rows.append(row_data)

        # Create the DataFrame from the list of rows
        df = pd.DataFrame(rows)

        return df