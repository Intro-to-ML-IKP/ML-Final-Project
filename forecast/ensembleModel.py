import numpy as np
import tensorflow as tf
import pandas as pd
import os
from sklearn.metrics import mean_squared_error
from typing import Any
from copy import deepcopy

from network.network import Model

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
        self._trend_model = Model()

        # Loads the models
        self._residual_model.load_model(residual_model_filepath)
        self._trend_model.load_model(trend_model_filepath)

    def predict_residuals(self, preprocessed_residuals):
        self._residual_model.predict(preprocessed_residuals)

    def predict_sma(self, sma_data):#(--------args--------)
        self._trend_model.predict(sma_data)

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
