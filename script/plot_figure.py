from prophet.utils.figure import *
from prophet.data.data_storage import *

if __name__ == '__main__':

    stock_db = StockDataStorage('../data/chinese_stock_codes.csv', '../data/history')

    df = stock_db.load_history('600000')

    fig = Figure()

    fig.plot(df, '2022-11-01', '2023-01-01')
