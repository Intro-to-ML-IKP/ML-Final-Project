from typing_extensions import override

from network.networkFactory import NetworkFactory
from trend_model.base_model import LstmModel


class LSTMFactory(NetworkFactory):

    @override
    def __init__(
            self,
            model_shape: list[float],
            activations: list[str],
            input_shape: int,
            output_shape: int = 1
    ) -> None:
        """
        Instanitates a Network factory that is used to
        construct a Neural Network.
        """
        super().__init__(model_shape,
                         activations,
                         input_shape,
                         output_shape)
        self._model = LstmModel()
