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

    print("training_data shape ",training_data.shape)
    print("training_labels shape ",training_labels.shape[0])
    lstm = LstmModel()
    

    lstm.create_sequential_model(9, 20, training_data.shape, training_labels.shape[0])

    lstm.compileModel(0.001, "mean_squared_error", metrics=["mae"])


    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

    scaled_training_data = scale_data(training_data, min_max_scaler)
    scaled_validation_data = scale_data(validation_data, min_max_scaler)
    
    print("scaled_training_data shape ",scaled_training_data.shape)
    print("scaled_validation_data shape ",scaled_validation_data.shape)
    print("..................")
    x_train = reshape_input(scaled_training_data)
    x_val = reshape_input(scaled_validation_data)
    print("x_train shape ",x_train.shape)
    print("x_val shape ",x_val.shape)
    print("validation_labels shape ",validation_labels.shape)

    lstm.trainModel(x_train, training_labels, x_val, validation_labels, epochs=20, batch_size=15)
    scaled_testing_data = scale_data(testing_data, min_max_scaler)
    x_test = reshape_input(scaled_testing_data)
    # scaled_testing_labels = scale_data(testing_labels)

    y_scaled_predictions = lstm.predict(x_test)
    print("y_scaled_predictions shape ", y_scaled_predictions.shape)
    print(testing_labels.shape)
    y_predictions = min_max_scaler.inverse_transform(y_scaled_predictions)

    print(y_predictions.shape)

    rmse = math.sqrt(mean_squared_error(testing_labels, y_predictions))
    print('Train Score: %.2f RMSE' % (rmse)) 




def reshape_input(input_data) -> np.ndarray:
    x = np.reshape(input_data, (input_data.shape[0], 1, input_data.shape[1]))
    return x


def scale_data(data, min_max_scaler) -> tuple:
    # Scaling data
    #for row in range(len(data)):
    scaled_data = min_max_scaler.fit_transform(data) # .reshape(-1, 1)
    # scaled_validation_data = min_max_scaler.fit_transform(validation_data.values.reshape(-1, 1))
    return scaled_data
