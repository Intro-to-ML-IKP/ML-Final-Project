from trend_model import get_training_data
from trend_model.base_model import LstmModel


(
    training_data,
    validation_data,
    testing_data,
    training_labels,
    validation_labels,
    testing_labels
    ) = get_training_data()

lstm = LstmModel() # needs params

lstm.create_sequential_model()#(FILL SHIT IN THIS)

lstm.compileModel(0.001, "mean square error", metrics=["mae"])

lstm.trainModel(training_data, training_labels, validation_data, validation_labels, epochs=50, batch_size=5)
