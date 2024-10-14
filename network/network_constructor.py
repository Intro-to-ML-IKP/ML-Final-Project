from network.network import Model
from typing import Any


class NetworkConstructor:
    results = []

    def __init__(self, model: Model, input_shape: Any, output_shape: Any, epochs: int):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model = model
        self.epochs = epochs
        #self._results = []
    
    def build_model(
            self,
            neurons_per_layer: list[int] = [32, 16],
            activations: list[str] = ["relu", "relu", "linear"],
            learning_rate: float = 0.001,
            lossFunc: str = "mse",
            metrics: list[str] = ["mae"]
            ) -> None:
        self._layers_activations_compatability(neurons_per_layer, activations)
        neurons_per_layer = neurons_per_layer
        neurons_per_layer.append(self.output_shape)
        self.model.create_sequential_model(neurons_per_layer, activations, self.input_shape)
        self.model.compileModel(learning_rate, lossFunc, metrics)
    
    def explore_different_architectures(
            self,
            training_data,
            training_labels,
            validation_data,
            validation_labels,
            neuron_variations: list[list[int]],
            learning_rates: list[float],
            batch_sizes: list[int]
            ):
        for neurons in neuron_variations:
            for learning_rate in learning_rates:
                for batch_size in batch_sizes:                    
                    # Creating activation functions ["relu", "relu", ..., "linear"]
                    activations = []
                    for _ in range(len(neurons)):
                        activations.append("relu")
                    activations.append("linear")

                    # Build and train the model
                    self.build_model(neurons_per_layer=neurons, activations = activations, learning_rate = learning_rate)
                    self.model.trainModel(training_data, training_labels, validation_data, validation_labels, self.epochs, batch_size)

                    # Compute the mae on the training data
                    mae = self.model.compute_mae(training_data, training_labels)

                    # Create the architecutre in the form of a list
                    architecture = [self.input_shape]
                    architecture.append(neuron for neuron in neurons)
                    architecture.append(self.output_shape)

                    nn_to_append = [mae, architecture, learning_rate, batch_size]
                    self.results.append(nn_to_append)
                    
                    # Append the results to key=mae as: 
                    # [[input shape, all the hidden layers, output shape], learning_rate, batch_size]
                    # self._results[mae] = [architecture, learning_rate, batch_size]

    def sort_results(self) -> dict:
        """
        Sort the results by the mae (smallest to largest)
        """
        sorted_keys = sorted(self._results.keys())
        sorted_results = {key: self._results[key] for key in sorted_keys}
        return sorted_results
    