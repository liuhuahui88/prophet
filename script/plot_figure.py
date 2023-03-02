from prophet.utils.figure import *
from prophet.data.data_storage import *

if __name__ == '__main__':
    storage = StockDataStorage('../data/chinese_stock_codes.csv', '../data/history')

    history = storage.load_history('600000')

    figure = Figure()

    figure.plot(history, start_date='2022-11-01', end_date='2023-01-01')
