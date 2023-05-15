import time
import os

from prophet.data.data_spider import *
from prophet.data.data_storage import *


if __name__ == '__main__':
    storage = StockDataStorage('../data/chinese_stock_codes.csv', '../data/history_temp')

    spider = StockDataSpider()

    symbols = storage.get_symbols()
    end_date = '2023-05-15'
    interval_in_seconds = 3

    num_of_symbols = len(symbols)
    for i in range(num_of_symbols):
        symbol = symbols[i]
        name = storage.get_name(symbol)

        history = storage.load_history(symbol)
        if len(history) != 0 and history.Date.iloc[-1] >= end_date:
            continue

        print('{}/{} : {}'.format(i + 1, num_of_symbols, [symbol, name]))

        history = spider.crawl(symbol)

        if history.shape[0] == 0:
            err_msg = "no record"
            print(err_msg)
            os.system("say '{}'".format(err_msg))

        storage.save_history(symbol, history)

        time.sleep(interval_in_seconds)
