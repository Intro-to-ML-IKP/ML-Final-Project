import os
import pickle
import seaborn as sns
import pandas as pd
from typing import Callable
from pandas.core.series import Series
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import numpy as np
from copy import deepcopy


class ResultsHandler:
    """
    This class is used to handle saving and loading of results.
    """
    def __init__(
            self,
            results: dict|list[list[float]]|None = None,
            mae: bool = True,
            lstm: bool = False
    ) -> None:
        """
        Instantiates a result handler that saves, loads and
        perfrorms stastitical analysis on results from the training of
        a ML model.

        :param results: the results,
        defaults to None
        :type results: dict | list[list[float]] | None, optional
        :param mae: if True generates a dataframe from the results,
        defaults to True
        :type mae: bool, optional
        :param lstm: if True interprets results in accordance to the LSTM
        experiment, defaults to False
        :type lstm: bool, optional
        """
        self._lstm_results = lstm
        self._results = results
        if mae:
            self._df = self._generate_pd_dataframe()

    @property
    def results(self) -> dict|list[float]:
        """
        Used to retrieve the results.

        :return: the results for the different networks
        :rtype: dict
        """
        return deepcopy(self._results)
    
    @results.setter
    def results(self, results: dict|list[float]) -> None:
        """
        Used to set the results attribute.

        :param results: the results
        :type results: dict|list[float]
        """
        self._results = results
    
    @property
    def df(self) -> pd.DataFrame:
        """
        Used to retrieve the dataframe.

        :return: the dataframe containiong the information
        for the different networks
        :rtype: dict
        """
        df = self._df

        # Reformulates the dataframe if LSTM
        if self._lstm_results:
            df = df.drop(columns="Neurons Layer 1")
            df = df.drop(columns="Number of Layers")
            df["Number of neurons"] = df.pop("Neurons Layer 2")

        return df

    def save_results(
            self,
            filename: str,
            foldername: str
            ) -> None:
        """
        Saving the dictionary to a file using pickle.

        :param results: a dictionary of results
        :results type: dict
        :param filename: the name of the file to be saved
        :filename type: str
        """
        # Define the base directory path for results
        base_dir = os.path.join(os.getcwd(), "results", foldername)

        # Ensure the directory exists
        os.makedirs(base_dir, exist_ok=True)

        # Define the complete filename with path
        full_filename = os.path.join(base_dir, filename)

        with open(full_filename, "wb") as file:
            pickle.dump(self.results, file)
            
    def save_results_readable(
            self,
            filename: str,
            foldername: str
    ) -> None:
        """
        Saving the dictionary to a file in human-readable format.

        :param results: a dictionary of results
        :results type: dict
        :param filename: the name of the file to be saved
        :filename type: str
        """

        filename = f"results\\{foldername}\\{filename}.pkl"
        if self._results is not None:
            if os.path.exists(filename):
                self._options_if_file_exists(filename)
            else:
                np.savetxt(filename, self._results)
        else:
            print(
                "There aren't any results to save."
                "The saving was unsuccessful."
                )

    def load_results(
            self,
            filename: str,
            foldername: str
            ) -> None:
        """
        Loading the dictionary from a pickle file.

        :param filename: the name of the file to be loaded
        :filename type: str
        """
        filename = f"results\\{foldername}\\{filename}"
        if os.path.exists(filename):
            with open(filename, "rb") as file:
                loaded_data = pickle.load(file)
            self._results = loaded_data
            self._df = self._generate_pd_dataframe()
            print("Results loaded successfully!")
        else:
            print(f"There is no dir `{filename}` found")

    def calculate_correlation_coefficients(
            self
            ) -> tuple[Series, pd.DataFrame]:
        """
        Calculates the correlation coefficients between the parameters
        and MAE.

        :return: a tuple of the MAE correlations and their respective
        p-values
        :rtype: tuple[Series, pd.DataFrame]
        """
        df = self._df

        # Drop the id column
        df = df.drop(columns=["id"])

        # Reformulates the dataframe if LSTM
        if self._lstm_results:
            df = df.drop(columns="Neurons Layer 1")
            df = df.drop(columns="Number of Layers")
            df["Number of neurons"] = df.pop("Neurons Layer 2")

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
                    # No p-value for self-correlation
                    p_values.loc[col1, col2] = None 

        # Focusing on the correlation of parameters with MAE
        mae_correlations = correlation_matrix["MAES"].drop("MAES")

        return mae_correlations, p_values
    
    def perform_regression_analysis(self) -> pd.DataFrame:
        """
        Performs regression analysis by examining the relationship
        between the MAE and the rest of the parameters.

        :return param_importance: the importance of each
        parameter
        :return type: pd.DataFrame
        """
        df = self._df

        # Drop the id column
        df = df.drop(columns=["id"])
        
        # Reformulates the dataframe if LSTM
        if self._lstm_results:
            df = df.drop(columns="Neurons Layer 1")
            df = df.drop(columns="Number of Layers")
            df["Number of neurons"] = df.pop("Neurons Layer 2")

            # Prepare the data        
            param_space = df[
                [
                    "Number of neurons",
                    "Learning Rate",
                    "Batch Size",
                    ]
                ]  # Features
        else:
            param_space = df[
                [
                    "Neurons Layer 1",
                    "Neurons Layer 2",
                    "Learning Rate",
                    "Batch Size",
                    "Number of Layers"
                    ]
                ]  # Features

        
        target_param = df["MAES"]  # Target

        # Fit the model
        model = LinearRegression()
        model.fit(param_space, target_param)

        # Get coefficients
        coefficients = model.coef_
        param_importance = pd.DataFrame(
            coefficients,
            index=param_space.columns,
            columns=['Coefficient']
            )

        return param_importance
    
    def get_parmeter_ranges(self) -> pd.DataFrame:
        """
        Get ranges for the explored parameter space of each
        parameter. Not implimented for LSTM yet.

        :return: a dataframe with the min-max values of each
        parmaeters along iwth their mean value.
        :rtype: pd.DataFrame
        """
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
        df = df.rename(
            index = {
                0: "Max Value",
                1: "Min Value",
                2: "Expected Value (Mean)"
                }
                )

        return df

    
    def _get_min_max_parameter(
            self,
            parameters: str
            ) -> list[float,float,float]:
        """
        Get the min, max and mean value of a parameter space.

        :param parameters: the parameter's name
        :type parameters: str
        :return: a list of max, min, and expected value (mean)
        :rtype: list[float,float,float]
        """
        df = self._df
        max_val = df[parameters].max()
        min_val = df[parameters].min()
        expected_val = df[parameters].mean()
        return [max_val, min_val, expected_val]

    def create_scatterplot_matrix(self) -> None:
        """
        Create a scatterplot matrix using seaborn's pairplot.
        Using fill=True to avoid the deprecated shade warning
        """
        df = self._df

        # Drop the id column
        df = df.drop(columns=["id"])

        # Reformulates the dataframe if LSTM
        if self._lstm_results:
            df = df.drop(columns="Neurons Layer 1")
            df = df.drop(columns="Number of Layers")
            df["Number of neurons"] = df.pop("Neurons Layer 2")

        sns.pairplot(
            df,
            diag_kind='kde',              # Diagonal kind: 'kde' for KDE plots
            markers='o',
            height=2.5,
            aspect=1.2,
            plot_kws={'alpha': 0.6},      # Scatter plot transparency
            diag_kws={'fill': True}       # Use fill=True for KDE plots
        )

        # The title for the figure
        plt.suptitle("Scatterplot Matrix", fontsize=12, y=1.02)  # Adjust y for title spacing
        
        plt.show()

    def create_correlation_heatmap(self) -> None:
        """
        Creates a correlation heatmap and plots it.
        """
        df = self._df

        # Drop the id column
        df = df.drop(columns=["id"])

        # Reformulates the dataframe if LSTM
        if self._lstm_results:
            df = df.drop(columns="Neurons Layer 1")
            df = df.drop(columns="Number of Layers")
            df["Number of neurons"] = df.pop("Neurons Layer 2")

        # Compute the correlation matrix
        corr_matrix = df.corr()

        # Set the figure size for better visibility
        plt.figure(figsize=(10, 8))

        # Create a heatmap using seaborn with improved aesthetics
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=0.5,
            linecolor='white',
            cbar_kws={"shrink": .8},
            annot_kws={"size": 10}
        )

        # Improve layout
        plt.title("Correlation Heatmap", fontsize=16)
        plt.xticks(rotation=45, ha='right', fontsize=12) # Rotate x-labels 
        plt.yticks(fontsize=12)                          # Y-axis label size

        # Adjust layout to make room for the title and labels
        plt.tight_layout()

        plt.show()

    def _options_if_file_exists(
            self,
            filename: str
            ) -> Callable:
        """
        This class method is used for if you are tring to save a file with a name
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
            print(
                f"The file `{filename}.pkl` has "
                "been overwriten with the current results.") 
            
        elif user_input == "n":
            user_input = input(
                "Chose a different name without any `.suffix` at the end:\n"
                )
            return self.save_results(str(user_input))
        
        else:
            return self.save_results(filename)
    
    def _generate_pd_dataframe(self) -> pd.DataFrame:
        """
        Generates a pd dataFrame from the results of the NN training.
        Puts each parameter into its own key.
        """
        # Get the data from the results
        maes = list(self._results.keys())
        values = list(self._results.values())

        rows = []

        for val in values:
            # Get the first tuple
            tup = val[0]

            # Create a dictionary to hold the parameters for the current row
            row_data = {}
            
            # Loop through the number of neurons for each layer
            for count, neurons in enumerate(tup[0]):
                row_data[f"Neurons Layer {count + 1}"] = neurons

            # Ensure that at least two layers are represented
            num_layers = len(tup[0])
            if num_layers < 2:
                # Fill missing layers with 0 neurons
                for i in range(2 - num_layers):
                    row_data[f"Neurons Layer {num_layers + 1 + i}"] = 0

            # Add the remaining parameters to the dictionary
            row_data["Learning Rate"] = tup[1]
            row_data["Batch Size"] = tup[2]
            row_data["Number of Layers"] = num_layers

            # Get the corresponding MAES for this row
            row_data["MAES"] = maes.pop(0)

            # Get the id of the model
            row_data["id"] = val[1]
            
            # Append the row data to the list of rows
            rows.append(row_data)

        # Create the DataFrame from the list of rows
        df = pd.DataFrame(rows)

        return df
