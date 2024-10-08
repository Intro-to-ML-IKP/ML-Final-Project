import matplotlib.pyplot as plt

class PlotStocks:
    def __init__(
            self,
            stockData: list[tuple[float,float,float,float]],
            sma: list[float] = None,
            extrapolated_sma: list[float] = None,
            residuals: list[float] = None,
            sma_length: int = 3
            ):
        self.high, self.low, self.open_, self.close = list(zip(*stockData))
        self.sma = sma
        self.sma_length = sma_length
        self.extrapolated_sma = extrapolated_sma
        self.residuals = residuals
        self.masterPlot_on = False

    def masterPlot(self, simpleMovingAverage = True):
        self.masterPlot_on = True
        _, (ax, ax1) = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [3, 1]})
        
        # Create the Plots
        self._plot_candlestick(ax, simpleMovingAverage)
        self._plot_residuals(ax1)

        # Adjust the spacing between subplots (increase hspace)
        plt.subplots_adjust(hspace=0.5)

        plt.show()
        self.masterPlot_on = False

    def plot_candlestick(self, simpleMovingAverage = False, ax = None):
        _, ax = plt.subplots()
        
        self._plot_candlestick(ax, simpleMovingAverage)

        plt.show()

    def _plot_candlestick(self, ax, simpleMovingAverage = False):
        # Number of days
        num_data = len(self.high)

        # Create an array of index values to represent days
        days = range(num_data)

        # Candlestick plotting
        for i in range(num_data):
            # Plot the line between high and low (the wick)
            ax.plot([days[i], days[i]], [self.low[i], self.high[i]], color='black')

            # Determine the color based on the open and close prices
            color = 'green' if self.close[i] >= self.open_[i] else 'red'

            # Plot the rectangle (the body) between open and close
            ax.add_patch(plt.Rectangle((days[i] - 0.2, self.open_[i]), 0.4, self.close[i] - self.open_[i], color=color))

        if simpleMovingAverage:
            self._plot_sma(ax, days)

        # Formatting
        ax.set_title("Stock Data")
        ax.set_ylabel('Price')

        ax.legend()

    def _plot_sma(self, ax, days):
        # Delete the first n elements
        x_val_SMA = days[self.sma_length-1:]
        ax.plot(x_val_SMA, self.sma, color="red", label = "SMA")
        if self.extrapolated_sma is not None:
            # Getting the x values for the days that the extrapolation happend
            nr_days_extrapolated = len(self.extrapolated_sma)
            last_day = days[-1]

            x_val_extrapolated_SMA = list(range(nr_days_extrapolated))
            x_val_extrapolated_SMA = [x + last_day for x in x_val_extrapolated_SMA]
            
            ax.plot(x_val_extrapolated_SMA, self.extrapolated_sma, color = "yellow", label = f"Extrapolated SMA ({nr_days_extrapolated} days)")


    def plot_residuals(self, ax1 = None):
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
        days = range(num_data)

        ax.plot(days, self.residuals, color="black", label="Residuals")

        # Formatting
        ax.set_title("Plot of the residuals")
        ax.set_ylabel("Value")

        ax.legend()