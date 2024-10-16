from network.network import Model
import os
import pickle

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
        for paramSet in paramList:
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
            print("Results loaded successfully!")
        else:
            print(f"There is no dir `{filename}` found")

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
    