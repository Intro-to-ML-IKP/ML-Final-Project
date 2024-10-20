import numpy as np
import itertools

class ParameterConstructor:
    def calcNetworkArchitectures(
        self, 
        maxLayers: int, 
        minNeurons: int,
        maxNeurons: int,
        dNeurons: int
    ) -> None:
        """Determines the list of different network architectures based on user-defined specifications
    
        Parameters:
        maxLayers   (int)   - The maximum number of hidden layers in the network
        minNeurons  (int)   - The minimum number of neurons per hidden layer
        maxNeurons  (int)   - The maximum number of neurons per hidden layer
        dNeurons    (int)   - The step size taken in the amount of neurons per layer
        
        Returns:
        None"""
        neuronCountList = np.arange(minNeurons, maxNeurons+dNeurons, dNeurons)      # Possible amounts of neurons per layer
        self.architectures = [list(perm) for n in range(1, maxLayers+1) for perm in itertools.product(neuronCountList, repeat=n)]      # Creates a 1D list of all possible neuron counts over all layers
    
    def calcLearningRates(
        self, 
        minLearningRate: float,
        maxLearningRate: float,
        dLearningRate: float,
    ) -> None:
        """Determines the list of possible learning rates for the networks"""
        self.learningRates = np.arange(minLearningRate, maxLearningRate+dLearningRate, dLearningRate)

    def calcBatchSize(
        self, 
        minBatchSize: int,
        maxBatchSize: int,
        dBatchSize: int,
    ) -> None:
        """Determines the list of possible batch sizes for the networks"""
        self.batchSizes = np.arange(minBatchSize, maxBatchSize+dBatchSize, dBatchSize)

    def calcParamList(
        self
    ) -> None:
        """Determines all the possible parameter lists based on the computed parameters"""
        # Check if the necessary lists are available
        if not hasattr(self, 'architectures') or not hasattr(self, 'learningRates') or not hasattr(self, 'batchSizes'):
            raise ValueError("You need to compute architectures, learning rates, and batch sizes before calling this method.")

        # Compute the Cartesian product of architectures, learningRates, and batchSizes
        self.paramList = list(itertools.product(self.architectures, self.learningRates, self.batchSizes))

        # Flatten the architectures tuple into a single structure for each tuple
        self.paramList = [(arch, lr, batch) for arch, lr, batch in self.paramList]
