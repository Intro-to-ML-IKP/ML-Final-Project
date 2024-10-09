from matplotlib.pyplot import figure,show
import json
import numpy as np

from .dense import Dense
from .activations import Tanh, Sigmoid

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class Network:
    def __init__(self, network, learning_rate = 0.01, overrideInit = False):
        """Initialize the network
        
        Parameters:
        network         (list)  - List containing the layers of classes and activation functions
        learning_rate   (float) - Scale factor for updating training parameters. Default = 0.01
        overrideInit    (bool)  - True if default init should be excluded. Default = False

        Returns:
        None
        """
        if not overrideInit:
            self.network = network
            self.learning_rate = learning_rate
            self.errors = None
            self.epochs = None

    def predict(self, input):
        """Forward propagates the entire network
        
        Parameters:
        input (list)    - List containing the input to the first layer of the network
        
        Returns:
        (list) - output of the network"""
        output = input
        for layer in self.network:
            output = layer.forward(output)
        return output

    def train(self, loss, loss_prime, x_train, y_train, epochs = 1000, verbose = True):
        """Trains a given network on a given training set with given inputs
        
        Parameters:
        loss            (func)  - Function used to compute the loss, used for computing the errors
        loss_prime      (func)  - Function for computing the derivative of the loss function, used for gradient descent
        x_train         (list)  - List containing the input variables for the neural network
        y_train         (list)  - List containing the labels for the training data given in x_train
        epochs          (int)   - The amount of training cycles used. Default = 1000
        
        verbose         (bool)  - Whether to print output or not. Default = True
        
        Returns:
        (list) - Errors of the network at each epoch"""
        errors = []
        self.epochs = epochs
        for e in range(epochs):
            error = 0
            for x, y in zip(x_train, y_train):
                # forward
                output = self.predict(x)

                # error
                error += loss(y, output)

                # backward
                grad = loss_prime(y, output)
                for layer in reversed(self.network):
                    grad = layer.backward(grad, self.learning_rate)

            error /= len(x_train)
            errors.append(error)
            if verbose:
                print(f"{e + 1}/{epochs}, error={error}")
        self.errors = errors
        return errors
    
    def dispPrediction(self, x):
        output = self.predict(x)
        fig = figure()
        frame = fig.add_subplot()
        frame.set_xlabel("Neuron index")
        frame.set_ylabel("Confidence level")
        frame.plot(output)
        fig.tight_layout()
        show()
        
    def dispErrors(self):
        # Display errors
        if self.errors == None:
            print("Couldn't display errors. Train first")
            return None
        fig = figure()
        frame = fig.add_subplot()
        frame.hist(self.errors)
        frame.set_title("Error evolution of training")
        frame.set_xlabel("Epoch")
        frame.set_ylabel("Error")
        fig.tight_layout()
        show()

    def saveNetwork(self, filename):
        networkData = {
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "layers": [layer.saveLayer() for layer in self.network]
        }
        js2 = json.dumps(networkData, indent=4, sort_keys=True, cls=NumpyEncoder)
        fp2 = open(f'{filename}.json', 'w', encoding='utf-8')
        fp2.write(js2)
        fp2.close()

    def loadNetwork(self, filename):
        with open(f"{filename}.json", "r") as f:
            data = json.load(f)
        self.learning_rate = data["learning_rate"]
        self.epochs = data["epochs"]

        newLayers = []
        for layer in data["layers"]:
            if layer["type"] == "Dense":
                layerObj = Dense(1, 1, overrideInit=True)
                layerObj.weights = np.array(layer["weights"])
                layerObj.bias = np.array(layer["bias"])
            elif layer["type"] == "Tanh":
                layerObj = Tanh()
            elif layer["type"] == "Sigmoid":
                layerObj = Sigmoid()
            newLayers.append(layerObj)
        self.network = newLayers


