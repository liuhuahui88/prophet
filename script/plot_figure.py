from prophet.utils.figure import *
from prophet.data.data_storage import *

if __name__ == '__main__':
    storage = StockDataStorage('../data/chinese_stock_codes.csv', '../data/history')

    symbol = '600000'

    history = storage.load_history(symbol)

    figure = Figure()

    figure.plot(history, symbol, '2022-11-01', '2023-01-01')
