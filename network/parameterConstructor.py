import numpy as np
import itertools
import random

class ParameterConstructor:
    """
    This class serves to generate permutations of different parameter
    combinations based on user input for MLP.
    """
    @property
    def paramList(self):
        if self._paramList is not None:
            return self._paramList
        else:
            raise ValueError(
                "There is no list of parameters avaliable!"
                "You need to make sure that you compute the"
                "parameter list first, you can do that by calling"
                "`calcParamList()` method.")
        
    # This is ugly I know, dont really care
    def getParamList(self):
        return self._paramList

    def calcNetworkArchitectures(
        self, 
        maxLayers: int, 
        minNeurons: int,
        maxNeurons: int,
        dNeurons: int
        ) -> None:
        """
        Determines the list of different network architectures based on 
        user-defined specifications.

        :param maxLayers: The maximum number of hidden layers in the network.
        :type maxLayers: int
        :param minNeurons: The minimum number of neurons per hidden layer.
        :type minNeurons: int
        :param maxNeurons: The maximum number of neurons per hidden layer.
        :type maxNeurons: int
        :param dNeurons: The step size for the number of neurons per layer.
        :type dNeurons: int
        """
        neuronCountList = np.arange(
            start = minNeurons,
            stop = maxNeurons+dNeurons,
            step = dNeurons
            )      # Possible amounts of neurons per layer
        
        self._architectures = [
            list(perm) for n in range(1, maxLayers+1)
            for perm in itertools.product(neuronCountList, repeat=n)
            ]      # Creates a 1D list of all possible neuron counts over all layers
    
    def calcLearningRates(
        self, 
        minLearningRate: float,
        maxLearningRate: float,
        dLearningRate: float,
        ) -> None:
        """
        Calculates the list of possible learning rates.

        :param minLearningRate: minimum learning rate
        :type minLearningRate: float
        :param maxLearningRate: maximum learning rate
        :type maxLearningRate: float
        :param dLearningRate: stepsize for the creation of the list
        :type dLearningRate: float
        """
        self._learningRates = np.arange(
            start = minLearningRate,
            stop = maxLearningRate+dLearningRate,
            step = dLearningRate
            )

    def calcBatchSize(
        self, 
        minBatchSize: int,
        maxBatchSize: int,
        dBatchSize: int,
        ) -> None:
        """
        Calculates the list of possible batch sizes.

        :param minBatchSize: minimum batch size
        :type minBatchSize: int
        :param maxBatchSize: maximum batch size
        :type maxBatchSize: int
        :param dBatchSize: stepsize for the creation of the list
        :type dBatchSize: int
        """
        self._batchSizes = np.arange(
            start = minBatchSize,
            stop = maxBatchSize+dBatchSize,
            step = dBatchSize
            )

    def calcParamList(self) -> None:
        """
        Determines all the possible parameter lists
        based on the computed parameters.

        :raises ValueError: if architectures, learning rates, and batch sizes
        are not computed beforehand
        """
        # Check if the necessary lists are available
        if (not hasattr(self, '_architectures')
            and not hasattr(self, '_learningRates')
            and not hasattr(self, '_batchSizes')
            ):
            raise ValueError(
                "You need to compute architectures, learning rates," + 
                "and batch sizes before calling this method."
                )

        # Compute the Cartesian product of architectures, learningRates, and batchSizes
        self._paramList = list(
            itertools.product(
                self._architectures,
                self._learningRates,
                self._batchSizes
                )
                )

        # Flatten the architectures tuple into a single structure for each tuple
        self._paramList = [
            (arch, lr, batch)
            for arch, lr, batch in self.paramList
            ]

class ParameterConstructorLSTM(ParameterConstructor):
    """
    This class serves to generate permutations of different parameter
    combinations based on user input for LSTM network.
    """
    def calcNetworkArchitectures(
        self, 
        maxLayers: int, 
        minNeurons: int,
        maxNeurons: int,
        dNeurons: int
        ) -> None:
        """
        Determines the list of different network architectures based on 
        user-defined specifications.

        :param maxLayers: The maximum number of hidden layers in the network.
        :type maxLayers: int
        :param minNeurons: The minimum number of neurons per hidden layer.
        :type minNeurons: int
        :param maxNeurons: The maximum number of neurons per hidden layer.
        :type maxNeurons: int
        :param dNeurons: The step size for the number of neurons per layer.
        :type dNeurons: int
        """
        neuronCountList = np.arange(
            start = minNeurons,
            stop = maxNeurons+dNeurons,
            step = dNeurons
            )      # Possible amounts of neurons per layer
        
        self._architectures = [
                [9, neuronCount]
                for neuronCount in neuronCountList
                for _ in range(maxLayers)
            ]     # Creates a 1D list of all possible neuron counts over all layers
        