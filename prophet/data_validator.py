import pandas as pd


class StockDataValidator:

    def __init__(self, start_date=None):
        self.start_date = start_date

        self.check_list = [
            self.__def_check(self.__check_na, 'Open'),
            self.__def_check(self.__check_na, 'Low'),
            self.__def_check(self.__check_na, 'High'),
            self.__def_check(self.__check_na, 'Close'),
            self.__def_check(self.__check_na, 'Volume'),

            self.__def_check(self.__check_neg, 'Open'),
            self.__def_check(self.__check_neg, 'Low'),
            self.__def_check(self.__check_neg, 'High'),
            self.__def_check(self.__check_neg, 'Close'),
            self.__def_check(self.__check_neg, 'Volume'),

            self.__def_check(self.__check_ge, 'Low', 'High'),

            self.__def_check(self.__check_oob, 'Close', 'Low', 'High'),
            self.__def_check(self.__check_oob, 'Open', 'Low', 'High'),

            self.__def_check(self.__check_jump, 'Close'),
        ]

    def validate(self, code, history: pd.DataFrame):
        if self.start_date is not None:
            history = history[history['Date'] >= self.start_date]

        summary = {}
        for check in self.check_list:
            description, invalid_records = check(code, history)
            summary[description] = invalid_records
        return summary

    @staticmethod
    def __def_check(f, *args):
        def g(code, history):
            code, marks = f(code, history, *args)
            return code, history[marks]['Date']
        return g

    @staticmethod
    def __check_na(code, history: pd.DataFrame, column):
        return '{} = NA'.format(column), history[column].isna()

    @staticmethod
    def __check_neg(code, history: pd.DataFrame, column):
        return '{} <= 0'.format(column), history[column] <= 0

    @staticmethod
    def __check_ge(code, history: pd.DataFrame, column1, column2):
        return '{} > {}'.format(column1, column2), history[column1] > history[column2]

    @staticmethod
    def __check_oob(code, history: pd.DataFrame, column1, column2, column3):
        return '{} OOB [{}, {}]'.format(column1, column2, column3),\
            ~history[column1].between(history[column2], history[column3])

    @staticmethod
    def __check_jump(code, history: pd.DataFrame, column):
        threshold = 0.22 if code[0] == '3' else 0.11
        return '{} Jump'.format(column), (history[column] / history[column].shift(1) - 1).abs() > threshold

