import time
import os

from prophet.data.data_spider import *
from prophet.data.data_storage import *


if __name__ == '__main__':
    storage = StockDataStorage('../data/chinese_stock_codes.csv', '../data/history_temp')

    spider = StockDataSpider()

    interval_in_seconds = 3

    symbols = storage.get_symbols()
    num_of_symbols = len(symbols)
    for i in range(num_of_symbols):
        symbol = symbols[i]
        name = storage.get_name(symbol)

        print('{}/{} : {}'.format(i + 1, num_of_symbols, [symbol, name]))

        df = spider.crawl(symbol)

        if df.shape[0] == 0:
            err_msg = "no record"
            print(err_msg)
            os.system("say '{}'".format(err_msg))

        storage.save_history(symbol, df)

        time.sleep(interval_in_seconds)
