class ForcastFactoryInitializer:
    """
    Initializes parameters for ForcastFactory and DataFactory model. 
    """
    def generate_model_parameters(
            self,
            architecture: list[int] = [13, 24],
            learning_rate: int = 0.01,
            loss_function : str = "mse",
            metrics: list[str] = ["mae"],
            epochs: int = 50,
            batch_size: int = 5
            ) -> dict:
        """
        Generates a dictionary of model parameters for network architecture, 
        training configurations, and performance metrics.

        :param architecture: List specifying the number of neurons in each layer 
        of the network. Default is [13, 24].
        :type architecture: list[int]
        :param learning_rate: The step size for adjusting model weights during 
        training. Default is 0.01.
        :type learning_rate: float
        :param loss_function: The loss function used in training, e.g., "mse" 
        for Mean Squared Error. Default is "mse".
        :type loss_function: str
        :param metrics: List of metrics to monitor during training, e.g., ["mae"]. 
        Default is ["mae"].
        :type metrics: list[str]
        :param epochs: Number of training epochs. Default is 50.
        :type epochs: int
        :param batch_size: Number of data points per batch in training. 
        Default is 5.
        :type batch_size: int

        :return: A dictionary containing model parameters.
        :rtype: dict
        """
        model_param_dict = locals()
        model_param_dict.pop("self")
        return model_param_dict

    def generate_datafactory_parameters(
            self,
            points_per_set: int = 10,
            num_sets: int = 50,
            labels_per_set: int = 1,
            testing_percentage: float = 0.8,
            validation_percentage: float = 0.1
            ) -> dict:
        """
        Generates a dictionary of DataFactory parameters for controlling 
        data generation and split settings for training, validation, and 
        testing.

        :param points_per_set: Number of data points per set. Default is 10.
        :type points_per_set: int
        :param num_sets: Number of data sets to generate. Default is 50.
        :type num_sets: int
        :param labels_per_set: Number of labels per set. Default is 1.
        :type labels_per_set: int
        :param testing_percentage: Percentage of data allocated to testing. 
        Default is 0.8.
        :type testing_percentage: float
        :param validation_percentage: Percentage of data allocated to validation. 
        Default is 0.1.
        :type validation_percentage: float

        :return: A dictionary containing data factory parameters.
        :rtype: dict
        """
        datafactory_param_dict = locals()
        datafactory_param_dict.pop("self")
        return datafactory_param_dict
    