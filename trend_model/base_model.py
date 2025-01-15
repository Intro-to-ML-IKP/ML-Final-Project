from keras.src.layers import ReLU
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.layers import LSTM # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from typing_extensions import override
from network.network import Model
from tensorflow.keras.optimizers import Adam # type: ignore



# TO DO:
### Research:
# ARIMA
# What data format for the model -> normalization?
# syntax (plotting & model)
### Code:
# begin Model class (template at network\network)
## attribute model = None
## method create_model...
## method train -> returns loss/error (check template)
## method fit
## method predict -> returns loss/error
## method plotting metrics -> RMSE


class LstmModel(Model):
    """
    The base class for all models.
    """
    def create_sequential_model(
            self,
            architecture: list[int],
            activations: list[str],
            input_shape: int,
            output_size: int
    ) -> None:
        """
        Creates the model architecture and assigns it to the model attribute.

        :param model_shape: Number of neurons in each layer.
        :type model_shape: list[float]
        :param activations: Activation function for each layer.
        :type activations: list[str]
        :param input_shape: Number of data points used for input.
        :type input_shape: int
        :param output_size: Number of neurons in the output layer.
        :type output_size: int
        """
        look_back = architecture[0]
        model_shape = architecture[1]

        print(input_shape)
        model = Sequential([
            LSTM(units=model_shape, input_shape=(input_shape, look_back)),
            ReLU(),
            Dense(units=output_size)
        ])

        self.model = model
