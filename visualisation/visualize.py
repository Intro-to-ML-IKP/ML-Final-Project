import matplotlib.pyplot as plt

def plot_candlestick(
        high,
        low,
        open_,
        close,
        sma: list[float] = None, sma_length: int = 3,
        extrapolated_sma: list[float] = None,  
        title='Candlestick Chart'):
    _, ax = plt.subplots()

    # Number of data points
    num_data = len(high)
    
    # Create an array of index values to represent days
    days = range(num_data)

    # Candlestick plotting
    for i in range(num_data):
        # Plot the line between high and low (the wick)
        ax.plot([days[i], days[i]], [low[i], high[i]], color='black')

        # Determine the color based on the open and close prices
        color = 'green' if close[i] >= open_[i] else 'red'

        # Plot the rectangle (the body) between open and close
        ax.add_patch(plt.Rectangle((days[i] - 0.2, open_[i]), 0.4, close[i] - open_[i], color=color))

    if sma is not None:
        # Delete the first n elements
        x_val_SMA = days[sma_length-1:]
        ax.plot(x_val_SMA, sma, color="red", label = "SMA")
        if extrapolated_sma is not None:
            # Getting the x values for the days that the extrapolation happend
            nr_days_extrapolated = len(extrapolated_sma)
            last_day = days[-1]

            x_val_extrapolated_SMA = list(range(nr_days_extrapolated))
            x_val_extrapolated_SMA = [x + last_day for x in x_val_extrapolated_SMA]
            
            ax.plot(x_val_extrapolated_SMA, extrapolated_sma, color = "yellow", label = f"Extrapolated SMA ({nr_days_extrapolated} days)")

    # Formatting
    ax.set_title(title)
    ax.set_ylabel('Price')

    ax.legend()
    # Show the plot
    plt.show()

def plot_residuals(residuals):
    _, ax = plt.subplots()

    # Number of data points
    num_data = len(residuals)
    
    # Create an array of index values to represent days
    days = range(num_data)

    ax.plot(days, residuals, color="black", label="Residuals")

    # Formatting
    ax.set_title("Plot of the residuals")
    ax.set_ylabel("Value")

    ax.legend()
    # Show the plot
    plt.show()