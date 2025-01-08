from abc import abstractmethod, ABC
import numpy as np


# TO DO:
### Research:
# LSTM or another model? - optional
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


class Model:
    """
    The base class for all models.
    """
    _model = None



    @abstractmethod
    def fit(self, train_X: np.ndarray, train_y: np.ndarray) -> None:
        """
        The abstract method that fits the training data to the model.
        :param train_X: np.ndarray training observations data
        :param train_y: np.ndarray training ground truth data
        :return: None
        """
        return None
