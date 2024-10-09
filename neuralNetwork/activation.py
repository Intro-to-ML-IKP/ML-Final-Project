import numpy as np
from .layer import Layer

class Activation(Layer):
    def __init__(self, activation, activation_prime, type):
        self.activation = activation
        self.activation_prime = activation_prime
        self.type = type

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))
    
    def saveLayer(self):
        layerObject = {
            "type": self.type
        }
        return layerObject
