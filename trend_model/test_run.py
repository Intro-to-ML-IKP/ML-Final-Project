import math

import numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.preprocessing import normalize

from trend_model import get_training_data
from trend_model.base_model import LstmModel
from trend_model.model_factory import DataNormalizer


def run():
    (
        training_data,
        validation_data,
        testing_data,
        training_labels,
        validation_labels,
        testing_labels
    ) = get_training_data()

    '''initial_datasets = (training_data, validation_data, testing_data,
                        training_labels, validation_labels, testing_labels
                        )'''

    lstm = LstmModel()

    lstm.create_sequential_model([9, 20], None, training_data[0].shape[0], training_labels[0].shape[0])

    lstm.compileModel(0.001, "mean_squared_error", metrics=["mae"])

    lstm.trainModel(training_data, training_labels, validation_data, validation_labels, epochs=10, batch_size=15)

    y_pred = lstm.predict(testing_data)
    print(y_pred)

    mse = mean_squared_error(testing_labels, y_pred)
    rmse = root_mean_squared_error(testing_labels, y_pred)
    print('Score: %.2f MSE' % mse)
    print('Score: %.2f RMSE' % rmse)
    #print('Normalized: %.2f RMSE' % (rmse/(max(testing_labels)-min(testing_labels)))) # not good
