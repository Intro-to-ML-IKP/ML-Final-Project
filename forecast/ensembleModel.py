import numpy as np
import tensorflow as tf
import pandas as pd
import os
from sklearn.metrics import mean_squared_error
from typing import Any
from copy import deepcopy

from network.network import Model
from trend_model.base_model import LstmModel

from data_parser.dataFactory import StockDataFactory
from visualisation.visualize import PlotStocks, PlotForcastComparison


MODEL_FOLDER = os.path.join(os.getcwd(), "models")


class EnsembleModel:
    def __init__(self, residual_model: str, residual_model_folder: str, trend_model: str, trend_model_folder: str) -> None:
        # Gets the models filepath
        residual_model_filepath = self._get_filepath(residual_model, residual_model_folder)
        trend_model_filepath = self._get_filepath(trend_model, trend_model_folder)

        # Instantiates the model
        self._residual_model = Model()
        self._trend_model = LstmModel()

        # Loads the models
        self._residual_model.load_model(residual_model_filepath)
        self._trend_model.load_model(trend_model_filepath)

    def predict_residuals(self, data_sets: list[list[float]]):
        all_predictions = []
        for dat in data_sets:
            residuals_tensor = np.array(dat).reshape(1,9)
    
            prediction = self._residual_model.predict(residuals_tensor)

            all_predictions.append(prediction)

        return all_predictions

    def predict_sma(self, data_sets: list[list[float]]):#(--------args--------)
        data_sets_arr = np.array(data_sets)
        prediction = self._trend_model.predict(data_sets_arr)
        return prediction

    def _get_filepath(self, filename: str, foldername: str) -> str:
        """
        Gets the filepath of a model

        :param filename: the model's name
        :type filename: str
        :param foldername: the model's folder
        :type foldername: str
        :return: the filepath
        :rtype: str
        """
        model_folderpath = os.path.join(MODEL_FOLDER, foldername)
        model_filepath = os.path.join(model_folderpath, filename)
        return model_filepath
