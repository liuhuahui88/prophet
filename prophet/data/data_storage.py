import pandas as pd


class StockDataStorage:

    def __init__(self, name_file_path, history_dir_path):
        name_df = pd.read_csv(name_file_path, dtype=str)
        self.__name_dict = {name_df.loc[i]['symbol']: name_df.loc[i]['name'] for i in name_df.index}

        self.__history_dir_path = history_dir_path

    def get_symbols(self, condition=lambda x: True):
        return [k for k in self.__name_dict.keys() if condition(k)]

    def get_name(self, symbol):
        return self.__name_dict.get(symbol, 'UNKNOWN')

    def load_history(self, symbol, start_date=None, end_date=None):
        history = pd.read_csv(self.__format_path(symbol))
        if start_date is not None:
            history = history[history.Date >= start_date]
        if end_date is not None:
            history = history[history.Date < end_date]
        history = history.reset_index(drop=True)
        return history

    def load_histories(self, symbols, start_date=None, end_date=None):
        return [self.load_history(symbol, start_date, end_date) for symbol in symbols]

    def save_history(self, symbol, history_df):
        history_df.to_csv(self.__format_path(symbol), index=False)

    def __format_path(self, symbol):
        return '{}/{}.csv'.format(self.__history_dir_path, symbol)
