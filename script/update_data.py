import time
import os

from prophet.data_spider import *
from prophet.data_storage import *


if __name__ == '__main__':

    name_file_path = '../data/chinese_stock_codes.csv'
    history_dir_path = '../data/history_temp'

    interval_in_seconds = 3

    spider = StockDataSpider()
    storage = StockDataStorage(name_file_path, history_dir_path)

    codes = storage.get_codes()
    num_of_codes = len(codes)
    for i in range(num_of_codes):
        code = codes[i]
        name = storage.get_name(code)

        print('processing {}/{} : {} {}'.format(i + 1, num_of_codes, code, name))

        df = spider.crawl(code)

        if df.shape[0] == 0:
            err_msg = "records not exist"
            print(err_msg)
            os.system("say '{}'".format(err_msg))

        storage.save_history(code, df)

        time.sleep(interval_in_seconds)
