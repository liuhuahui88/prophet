import pandas as pd


class StockDataStorage:

    def __init__(self, name_file_path, history_dir_path):
        name_df = pd.read_csv(name_file_path, dtype=str)
        self.__name_dict = {name_df.loc[i]['symbol']: name_df.loc[i]['name'] for i in name_df.index}

        self.__history_dir_path = history_dir_path

    def get_symbols(self):
        return list(self.__name_dict.keys())

    def get_name(self, symbol):
        return self.__name_dict.get(symbol, 'UNKNOWN')

    def load_history(self, symbol):
        return pd.read_csv(self.__format_path(symbol))

    def save_history(self, symbol, history_df):
        history_df.to_csv(self.__format_path(symbol), index=False)

    def __format_path(self, symbol):
        return '{}/{}.csv'.format(self.__history_dir_path, symbol)
