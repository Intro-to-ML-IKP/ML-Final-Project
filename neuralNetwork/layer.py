class Layer:
    """A base class for layers in a neural network.

    Attributes:
    input   (list) - The input to the layer.
    output  (list) - The output from the layer.
    """

    def __init__(self):
        """Initializes the Layer with default input and output attributes set to None."""
        self.input = None
        self.output = None

    def forward(self, input):
        """Performs the forward pass through the layer.

        Parameters:
        input (list) - The input data to the layer.

        Returns:
        (list) - The output of the layer after applying the layer's forward transformation.
        """
        # TODO: return output
        pass

    def backward(self, output_gradient, learning_rate):
        """Performs the backward pass through the layer, updating the layer's parameters and computing the gradient for the input.

        Parameters:
        output_gradient (list)  - The gradient of the loss with respect to the layer's output.
        learning_rate   (float) - The learning rate to use for updating the layer's parameters.

        Returns:
        (list) - The gradient of the loss with respect to the layer's input.
        """
        # TODO: update parameters and return input gradient
        pass

    def saveLayer(self):
        """Return a JSON object of the relevant layer data"""
        pass

    def loadLayer(self, obj):
        """Loads a layer from JSON object. Inserted object must correspond to layer type as no checks are performed at the moment"""
        pass