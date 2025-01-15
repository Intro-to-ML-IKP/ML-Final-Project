import math

import numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

from trend_model import get_training_data
from trend_model.base_model import LstmModel


def run():
    (
        training_data,
        validation_data,
        testing_data,
        training_labels,
        validation_labels,
        testing_labels
    ) = get_training_data()

    initial_datasets = (training_data, validation_data, testing_data,
                        training_labels, validation_labels, testing_labels
                        )

    lstm = LstmModel()

    lstm.create_sequential_model([9, 20], None, training_data[0].shape[0], training_labels[0].shape[0])

    lstm.compileModel(0.001, "mean_squared_error", metrics=["mae"])

    scaler = DataNormalizer()

    datasets = [scaler.scale_data(data) for data in initial_datasets]

    x_train = datasets[0]
    y_train = datasets[3]

    x_val = datasets[1]
    y_val = datasets[4]

    x_test = datasets[2]
    y_test = datasets[5]

    lstm.trainModel(x_train, y_train, x_val, y_val, epochs=20, batch_size=15)

    '''scaled_testing_data = scale_data(testing_data)
    x_test = reshape_input(scaled_testing_data)'''
    # scaled_testing_labels = scale_data(testing_labels)

    y_scaled_predictions = lstm.predict(x_test)
    y_predictions = scaler.inverse_scaled_data(y_scaled_predictions)
    print(x_test)
    print(y_test)
    print(y_predictions)

    mse = mean_squared_error(y_test, y_predictions)
    print('Train Score: %.2f MSE' % (mse))


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
