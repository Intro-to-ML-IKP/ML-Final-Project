class ForecastFactoryInitializer:
    def generate_model_parameters(
            self,
            architecture: list[int] = [13, 24],
            learning_rate: int = 0.01,
            loss_function : str = "mse",
            metrics: list[str] = ["mae"],
            epochs: int = 50,
            batch_size: int = 5
            ) -> dict:
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
        datafactory_param_dict = locals()
        datafactory_param_dict.pop("self")
        return datafactory_param_dict
    