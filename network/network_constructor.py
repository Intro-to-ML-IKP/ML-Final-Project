from multiprocessing import Pool
from typing import Tuple
import tensorflow as tf
from functools import partial
from copy import deepcopy

from network.network import Model
from results.result_handler import ResultsHandler

# Disable progress bars
tf.keras.utils.disable_interactive_logging()

# Define the type for a single parameter tuple
ParamTuple = list[
    Tuple[
        Tuple[
            list[int], float, int
            ],   # The current parameter set
        int,                            # Index of the current parameter set
        int,                            # Total number of parameter sets
        list[float],                   # Training data
        list[float],                   # Training labels
        list[float],                   # Validation data
        list[float],                   # Validation labels
        list[float],                   # Testing data
        list[float]                    # Testing labels
        ]
    ]

LIST_BB = list(range(1000, 1000001, 1000))

class NetworksConstructor:
    """
    A way of constructing NNs with different parameters to be
    used in performing a statistical analysis to deremine the
    optimal set of parameters.
    """
    def __init__(
            self,
            model: Model,
            input_size: int,
            output_size: int,
            epochs: int
            ) -> None:
        """
        Initializes parameters used throughout the class.

        :param input_size: Number of data points in the input data.
        :type input_size: int
        :param output_size: Number of points in the output data.
        :type output_size: int
        :param epochs: Maximum number of epochs for training.
        :type epochs: int
        """
        self._results = []
        self._training_loss = []
        self._validation_loss = []
        self.input_size = input_size
        self.output_size = output_size
        self.epochs = epochs
        self.model: Model = model

    @property
    def training_loss(self):
        return deepcopy(self._training_loss)
    
    @property
    def validation_loss(self):
        return deepcopy(self._validation_loss)

    @property
    def results(self):
        return deepcopy(self._results)
    
    def _build_model(
            self,
            architecture: list[int],
            activations: list[str],
            learning_rate: float = 0.001,
            lossFunc: str = "mse",
            metrics: list[str] = ["mae"]
            ) -> Model:
        """
        Builds a model of a network.

        :param architecture: the architecture of the NN
        :type architecture: list[int]
        :param activations: list with actiavation functions
        :type activations: list[str]
        :param learning_rate: the learning rate of the model,
        defaults to 0.001
        :type learning_rate: float, optional
        :param lossFunc: the loss function that we are minimising,
        defaults to "mse"
        :type lossFunc: str, optional
        :param metrics: a list of metrics that we are tracking,
        defaults to ["mae"]
        :type metrics: list[str], optional
        :return: the model
        :rtype: Model
        """
        model = self.model()

        model.create_sequential_model(
            architecture,
            activations,
            self.input_size,
            self.output_size
            )
        
        model.compileModel(
            learning_rate,
            lossFunc,
            metrics
            )
        return model

    def _explore(
            self,
            params: ParamTuple,
            results_filename: str,
            results_foldername: str
            ) -> None:
        """
        Explores one model at a time

        :param params: the parameters of the model.
        :type params: ParamTuple
        :param results_filename: the name of the file to be used
        for intermidiate saving
        :type results_filename: str
        :param results_foldername: the folder where the intermidiate files
        are saved
        :type results_foldername: str
        """
        # Unpacking the parameters
        (
            paramSet,
            count,
            maxCount,
            training_data,
            training_labels,
            validation_data,
            validation_labels,
            testing_data,
            testing_labels
            ) = params
        
        # Unpacking the parameter sets
        (
            architecture,
            learning_rate,
            batch_size
            ) = paramSet

        # Create list of activation functions for the network,
        # which will be relu for all but the output layer
        activations = ["relu" for _ in range(len(architecture))]    
        activations.append("linear")

        # Create and train the model
        model = self._build_model(
            architecture=architecture,
            activations=activations,
            learning_rate=learning_rate
            )
        
        # Get the training and validation losses
        training_loss, validation_loss = model.trainModel(
            training_data,
            training_labels,
            validation_data,
            validation_labels,
            self.epochs,
            batch_size
            )

        # Evaluate the model
        mae = model.compute_mae(
            testing_data,
            testing_labels
            )

        # Adding the model id (just the count) this paramSet belongs to
        paramSet = tuple((paramSet, count))

        # Print the progress on which model is currently being trained
        print(f"Now training the model {count}/{maxCount}")
        
        # Intermediatly saves duiring simulations
        if len(self.validation_loss) in LIST_BB:
            results_handler = ResultsHandler(self.validation_loss, mae=False)
            results_handler.save_results(
                f"{results_filename}_validation_loss_{count}",
                results_foldername
                )
            
            results_handler.results = self.training_loss
            results_handler.save_results(
                f"{results_filename}_training_loss_{count}",
                results_foldername
                )
            
            maes = NetworksDict().sort_results_list(self.results)
            results_handler.results = maes
            results_handler.save_results(
                f"{results_filename}_{count}",
                results_foldername
                )
            
        return [mae, paramSet], (count+1, validation_loss), (count+1, training_loss), model, count+1

    def explore_different_architectures(
        self,
        training_data: list[float],
        training_labels: list[float],
        validation_data: list[float],
        validation_labels: list[float],
        testing_data: list[float],
        testing_labels: list[float],
        paramList: list[tuple[list[int], float, int]],
        results_filename: str,
        results_foldername: str,
        save_model: bool,
        ) -> None:
        """
        Generates a list of parameter sets to train different network models.

        :param training_data: Data used to train the model.
        :type training_data: list[float]
        :param training_labels: Desired output labels for training.
        :type training_labels: list[float]
        :param validation_data: Data used for validation.
        :type validation_data: list[float]
        :param validation_labels: Labels used for validation.
        :type validation_labels: list[float]
        :param testing_data: Data used to evaluate model performance.
        :type testing_data: list[float]
        :param testing_labels: Labels for evaluating model performance.
        :type testing_labels: list[float]
        :param paramList: Precomputed list of parameters for building and 
        training networks. Each tuple includes:
                - architectures: list of neuron counts per layer in the network.
                - learning_rates: The learning rate used in training.
                - batch_sizes: Number of samples processed before weight update.
        :type paramList: list[tuple[list[int], float, int]]

        :return: All mean absolute errors (maes) for each parameter combination, 
        with the corresponding parameter sets.
        :rtype: list[list[float, tuple[list[int], float, int]]]
        """
        # Initialize an empty list to store all parameter sets with metadata
        full_param_list = []

        # Loop over paramList with index and parameter set
        for count, param_set in enumerate(paramList):
            # Create a tuple with the parameter set and associated metadata
            param_tuple = (
                param_set,                  # The current parameter set
                count,                      # Index of the current parameter set
                len(paramList),             # Total number of parameter sets
                training_data,              # Training data
                training_labels,            # Training labels
                validation_data,            # Validation data
                validation_labels,          # Validation labels
                testing_data,               # Testing data
                testing_labels              # Testing labels
            )
            
            # Append the tuple to the full_param_list
            full_param_list.append(param_tuple)
        
        # Locks self._explore's filename and foldername parameters
        # in place so pool().map() can be applied to it
        explore_with_args = partial(
            self._explore,
            results_filename=results_filename,
            results_foldername=results_foldername
            )
        
        # Perform exploration on each parameter combination
        with Pool(processes=5) as p: # processes - how many NNs we train at a time
            results = p.map(explore_with_args, full_param_list)

        for result, validation_loss, training_loss, model, count in results:
            self._validation_loss.append(validation_loss)
            self._training_loss.append(training_loss)
            self._results.append(result)
            if save_model:
                model.save_model("early_stoppage", f"AAPL_{count}")
            
class NetworksDict:
    def sort_results_list(self, result_list):
        sorted_results = self._sort_results(result_list)
        return sorted_results
    
    def _list_to_dict(self, result_list: list[list[float]]) -> dict:
        """
        Gets the NetworksConstructor's results and makes
        them into a dictionary.

        :return: a dictionary
        :type return: dict
        """
        nnDict = {}

        # Access the result list in NetworksConstructor
        nn_results = result_list

        # Create a dictionary with MAEs as keys and parameters as values
        for mae, params in nn_results:
            nnDict[mae] = params  # Store params associated with each MAE
        return nnDict

    def _sort_results(self, result_list: list[list[float]]) -> dict:
        """
        Sort the results by the mae (smallest to largest)

        :return: a sorted dict
        :type return: dict
        """
        nnDict = self._list_to_dict(result_list)
        sorted_keys = sorted(nnDict.keys())
        sorted_results = {key: nnDict[key] for key in sorted_keys}
        return sorted_results
    