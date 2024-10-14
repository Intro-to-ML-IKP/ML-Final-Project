from network.network import Model

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

        architecture.append(self.output_size)
        model = Model()
        model.create_sequential_model(architecture, activations, self.input_size)
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
        return cls._sort_results()
    
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
    