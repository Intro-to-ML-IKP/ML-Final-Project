# Main program
from stock_getter import Stock
from matplotlib.pyplot import figure,show

def plotData(open, high, low, close):
    fig = figure()
    frame = fig.add_subplot()
    for i,params in enumerate(zip(open, high,low,close)):
        o,h,l,c = params
        if o >= c:
            frame.scatter(i,c, color="r")
        else: 
            frame.scatter(i,c, color="lime")

    show()

def main():
    stock = Stock("AAPL", "2024-01-01", "2024-09-01")
    open, high, low, close = stock.get_data()
    plotData(open, high, low, close)
    # with open("test.txt", "w") as f:
    #     for c in closes:
    #         f.write(str(c))
    #         f.write("\n")
    
    

if __name__ == "__main__":
    main()