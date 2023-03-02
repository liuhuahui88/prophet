from prophet.data.data_storage import *
from prophet.data.data_validator import *


if __name__ == '__main__':
    storage = StockDataStorage('../data/chinese_stock_codes.csv', '../data/history')

    validator = StockDataValidator('2010-01-01')

    for symbol in storage.get_symbols():
        name = storage.get_name(symbol)
        history = storage.load_history(symbol)
        summary = validator.validate(symbol, history)
        for description in summary:
            invalid_results = summary[description]
            for r in invalid_results:
                print([symbol, name], description, r)
