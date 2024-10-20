import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
from pandas.core.series import Series



class PlotStocks:
    def __init__(
            self,
            stockData: list[tuple[Series,Series,Series,Series,Series]],
            sma: list[float] = None,
            extrapolated_sma: list[float] = None,
            residuals: list[float] = None,
            predicted_closing_prices: list[float] = None,
            sma_length: int = 3
            ):
        self.dates, self.open_, self.high, self.low, self.close = list(zip(*stockData))
        self.sma = sma
        self.sma_length = sma_length
        self.extrapolated_sma = extrapolated_sma
        self.residuals = residuals
        self.predicted_closing_prices = predicted_closing_prices
        self.masterPlot_on = False

    def masterPlot(self, simpleMovingAverage = True, predictedClosingPrices = False):
        self.masterPlot_on = True
        _, (ax, ax1) = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [3, 1]})
        
        # Create the Plots
        self._plot_candlestick(ax, simpleMovingAverage, predictedClosingPrices)
        self._plot_residuals(ax1)

        # Adjust the spacing between subplots (increase hspace)
        plt.subplots_adjust(hspace=0.5)

        plt.show()
        self.masterPlot_on = False

    def plot_candlestick(self, simpleMovingAverage = False, predictedClosingPrices = False):
        _, ax = plt.subplots()
        
        self._plot_candlestick(ax, simpleMovingAverage, predictedClosingPrices)

        plt.show()

    def _plot_candlestick(self, ax, simpleMovingAverage = False, predictedClosingPrices = False):
        # Number of days
        num_data = len(self.high)

        # Create an array of index values to represent days
        dates = self.dates
        dates_numeric = mdates.date2num(dates)

        # Candlestick plotting
        for i in range(num_data):
            # Plot the line between high and low (the wick)
            ax.plot([dates[i], dates[i]], [self.high[i], self.low[i]], color='black')

            # Determine the color based on the open and close prices
            color = 'green' if self.close[i] > self.open_[i] else 'red'
            print("open", "high", "low", "close")
            print(self.open_[i], self.high[i], self.low[i], self.close[i])
            print(color)
            # Plot the rectangle (the body) between open and close
            ax.add_patch(plt.Rectangle((dates_numeric[i] - 0.2, self.open_[i]), 0.4, abs(self.close[i] - self.open_[i]), color=color))

        if simpleMovingAverage:
            self._plot_sma(ax, dates)

        if predictedClosingPrices:
            self._plot_predicted_closing_prices(ax)

        # Formatting
        ax.set_title("Stock Data")
        ax.set_ylabel('Price')

        ax.legend()

    def _plot_sma(self, ax, dates):
        # Delete the first n elements
        x_val_SMA = dates[self.sma_length-1:]
        ax.plot(x_val_SMA, self.sma, color="red", label = "SMA")
        if self.extrapolated_sma is not None:
            # Getting the x values for the days that the extrapolation happend
            nr_days_extrapolated = len(self.extrapolated_sma)
            last_day = dates[-1]

            future_dates = pd.bdate_range(start=last_day, periods=nr_days_extrapolated).tolist()
            
            ax.plot(future_dates, self.extrapolated_sma, color = "yellow", label = f"Extrapolated SMA ({nr_days_extrapolated} days)")

    def plot_residuals(self):
        """
        This method plots the residuals.

        :param ax1: used only when called 
        """
        _, ax = plt.subplots()

        self._plot_residuals(ax)

        plt.show()

    def _plot_residuals(self, ax):
        # Number of data points
        num_data = len(self.residuals)
        
        # Create an array of index values to represent days
        dates = self.dates[-num_data:]#range(num_data)

        ax.plot(dates, self.residuals, color="black", label="Residuals")

        # Formatting
        ax.set_title("Plot of the residuals")
        ax.set_ylabel("Value")

        ax.legend()

    def plot_predicted_closing_prices(self):
        _, ax = plt.subplots()

        self._plot_predicted_closing_prices(ax)

        plt.show()

    def _plot_predicted_closing_prices(self, ax):
        nr_days_extrapolated = len(self.predicted_closing_prices)
        last_day = self.dates[-1]

        future_dates = pd.bdate_range(start=last_day, periods=nr_days_extrapolated).tolist()

        future_dates_numeric = mdates.date2num(future_dates)
        
        for predicted_closing_price in self.predicted_closing_prices:
            ax.hlines(
                predicted_closing_price,
                future_dates_numeric - 0.2,
                future_dates_numeric + 0.2,
                color = "blue",
                label = f"Predicted Closing Prices ({nr_days_extrapolated} days)"
                )
            