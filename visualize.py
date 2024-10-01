import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_candlestick(high, low, open_, close, title='Candlestick Chart'):
    fig, ax = plt.subplots()

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

    # Formatting
    ax.set_title(title)
    ax.set_ylabel('Price')

    # Show the plot
    plt.show()