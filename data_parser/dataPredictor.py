class PredictedDataGenerator:
    def __init__(self, predicted_residuals: list[float], extrapolated_sme: list[float]):
        self._predicted_data_validator(predicted_residuals, extrapolated_sme)
        self._predicted_residuals = predicted_residuals
        self._extrapolated_sme = extrapolated_sme

    def get_closing_prices(self):
        closing_prices = self._extrapolated_sme + self._predicted_residuals
        return closing_prices
    
    def _predicted_data_validator(self, res: list[float], ext_sme: list[float]) -> None:
        res_len = len(res)
        ext_sme_len = len(ext_sme)
        if res_len != ext_sme_len:
            raise ValueError("The list of residuals needs to be the same size"
                             " as the extrapolated sme. You provided:\n"
                             f"len(predicted_residuals) = {res_len}\n"
                             f"len(extrapolated_sme) = {ext_sme_len}.")