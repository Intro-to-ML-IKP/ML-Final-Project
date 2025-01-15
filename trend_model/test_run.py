import math

import numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

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

    lstm.trainModel(training_data, training_labels, validation_data, validation_labels, epochs=30, batch_size=15)

    for i in range(len(testing_data)):
        pred = lstm.predict([testing_data[i]])
        print(pred)


    '''
    scaler = DataNormalizer()

    datasets = [scaler.scale_data(data) for data in initial_datasets]

    x_train = datasets[0]
    y_train = datasets[3]

    x_val = datasets[1]
    y_val = datasets[4]

    x_test = datasets[2]
    y_test = datasets[5]

    lstm.trainModel(x_train, y_train, x_val, y_val, epochs=20, batch_size=15)

    scaled_testing_data = scale_data(testing_data)
    x_test = reshape_input(scaled_testing_data)
    # scaled_testing_labels = scale_data(testing_labels)

    y_scaled_predictions = lstm.predict(x_test)
    y_predictions = scaler.inverse_scaled_data(y_scaled_predictions)
    print(x_test)
    print(y_test)
    print(y_predictions)'''

    mse = mean_squared_error(y_test, y_predictions)
    print('Train Score: %.2f MSE' % (mse))

