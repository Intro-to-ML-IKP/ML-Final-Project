# Main program
from stock_getter import Stock
from matplotlib.pyplot import figure,show

def main():
    stock = Stock("AAPL", "2024-01-01", "2024-09-01")
    closes = stock.get_data()
    with open("test.txt", "w") as f:
        for c in closes:
            f.write(str(c))
            f.write("\n")
    
    xList = range(len(closes))

    fig = figure()
    frame = fig.add_subplot()
    frame.scatter(xList, closes)
    show()

if __name__ == "__main__":
    main()