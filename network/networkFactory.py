from network.network import Model

class NetworkFactory:
    def __init__(
            self,
            model_shape : list[float],
            activations: list[str],
            input_shape: int,
            output_shape: int = 1
            ) -> None:
        """
        Instanitates a Network factory that is used to
        construct a Neural Network.

        :param model_shape: the shape iof the model
        :type model_shape: list[float]
        :param activations: the activation functions
        :type activations: list[str]
        :param input_shape: the input shape
        :type input_shape: int
        :param output_shape: the output shape,
        defaults to 1
        :type output_shape: int
        """
        self._model_shape = model_shape
        self._activations = activations
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._model = Model()#self._construct_model(model_shape, activations, input_shape, output_size)

    def train(
            self,
            training_data: list[float],
            training_labels: list[float],
            validation_data: list[float],
            validation_labels: list[float],
            learning_rate: float,
            lossFunc: str,
            metrics: list[str],
            epochs: int,
            batch_size: int
            ) -> None:
        # Create Sequential model
        self._model.create_sequential_model(self._model_shape, self._activations, self._input_shape, self._output_shape)

        # Compile the model
        self._model.compileModel(learning_rate, lossFunc, metrics)

        # Train the model
        self._model.trainModel(training_data, training_labels, validation_data, validation_labels, epochs, batch_size)

    def predict(self, data: list[float], number_of_predictions: int) -> list[float]:
        if number_of_predictions > len(data):
            difference = number_of_predictions - len(data)

        sliding_data = data
        for _ in range(number_of_predictions):
            # Make the prediction
            prediction = self._model.predict(sliding_data)

            if number_of_predictions > len(data):
                difference = number_of_predictions - len(data)
                
            # Add the prediction to the sliding data
            sliding_data.append(prediction)

            # Remove the first datum in the data
            sliding_data.pop(0)
        
        return sliding_data
