from prophet.data.data_storage import *
from prophet.data.data_validator import *


if __name__ == '__main__':
    validator = StockDataValidator('2010-01-01')

    name_file_path = '../data/chinese_stock_codes.csv'
    history_dir_path = '../data/history'

    storage = StockDataStorage(name_file_path, history_dir_path)

    for code in storage.get_codes():
        name = storage.get_name(code)
        history = storage.load_history(code)
        summary = validator.validate(code, history)
        for description in summary:
            invalid_results = summary[description]
            for r in invalid_results:
                print([code, name], description, r)
