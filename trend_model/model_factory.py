import numpy as np
from sklearn import preprocessing


class DataNormalizer:

    def __init__(self):
        self.min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

    def _reshape_input(self, input_data: np.ndarray) -> np.ndarray:
        x = np.reshape(input_data, (input_data.shape[0], 1, input_data.shape[1]))
        return x

    def scale_data(self, data: np.ndarray) -> np.ndarray:
        # Scaling data
        scaled_data = self.min_max_scaler.fit_transform(data)  # .reshape(-1, 1)
        transformed_data = self._reshape_input(scaled_data)
        return transformed_data

    def inverse_scaled_data(self, data: np.ndarray) -> np.ndarray:
        transformed_data = self.min_max_scaler.inverse_transform(data)
        return transformed_data

