import datetime

import numpy as np
import pandas as pd

from prophet.agent.baseline_agent import BaselineAgent
from prophet.agent.oracle_agent import OracleAgent
from prophet.agent.smart_agent import SmartAgent
from prophet.bt.back_tester import BackTester
from prophet.bt.broker import Broker
from prophet.data.data_extractor import DataExtractor
from prophet.data.data_storage import StockDataStorage
from prophet.predictor.abstract_predictor import Predictor
from prophet.utils.constant import Const


class PlayGround:

    def __init__(self, name_file_path, history_file_path, commission_rate):
        self.storage = StockDataStorage(name_file_path, history_file_path)

        self.commission_rate = commission_rate
        self.log_friction = -(np.log(1 - commission_rate) + np.log(1 / (1 + commission_rate)))

        self.extractor = DataExtractor(commission_rate)

    def train(self, symbols, start_date, train_end_date, test_end_date, predictor: Predictor,
              debug_train=False, debug_test=False):
        histories = self.storage.load_histories(symbols, start_date, test_end_date)

        syms = self.__broadcast_symbols(symbols, histories)
        dates = self.extractor.extract_and_concat(histories, ['date'])['date']

        features = self.extractor.extract_and_concat(histories, predictor.get_feature_names())
        labels = self.extractor.extract_and_concat(histories, predictor.get_label_names())

        train_syms, train_dates, train_sample_set = self.__split_sample_set(
            syms, dates, features, labels, dates['Date'].apply(lambda d: d < train_end_date))
        test_syms, test_dates, test_sample_set = self.__split_sample_set(
            syms, dates, features, labels, dates['Date'].apply(lambda d: d >= train_end_date))

        predictor.fit(train_sample_set, test_sample_set)

        if debug_train:
            train_results = predictor.predict(train_sample_set)
            self.__save_samples(train_syms, train_dates, train_sample_set, train_results, 'train')

        if debug_test:
            test_results = predictor.predict(test_sample_set)
            self.__save_samples(test_syms, test_dates, test_sample_set, test_results, 'test')

    def test(self, symbols, start_date, end_date,
             predictors, delta_free_list=None, threshold=0, top_k=1,
             with_baseline=False, with_oracle=False, verbose=False):
        bt = BackTester(self.storage, Broker(self.commission_rate))

        histories = self.__load_histories(symbols, start_date, end_date)

        for name, predictor in predictors.items():
            caches = self.__build_caches(symbols, histories, predictor)
            delta = 0 if delta_free_list is not None and name in delta_free_list else self.log_friction
            bt.register('SMT_' + name, SmartAgent(caches, delta, threshold, top_k))

        if with_baseline:
            bt.register('BASE', BaselineAgent())
        for symbol in symbols:
            if with_oracle:
                bt.register('ORA_' + symbol, OracleAgent(symbol, self.storage, self.extractor))

        result = bt.back_test(symbols, start_date, end_date, verbose=verbose)

        return result

    def __broadcast_symbols(self, symbols, histories):
        symbols_df_list = []
        for i in range(len(histories)):
            if len(histories[i]) != 0:
                s = [symbols[i]] * len(histories[i])
                symbols_df = pd.DataFrame({'Symbol': s})
                symbols_df_list.append(symbols_df)
        return pd.concat(symbols_df_list).reset_index(drop=True)

    def __split_sample_set(self, syms, dates, features, labels, condition):
        syms = syms[condition]
        dates = dates[condition]
        features = {k: v[condition] for k, v in features.items()}
        labels = {k: v[condition] for k, v in labels.items()}
        size = len(dates)
        return syms, dates, Predictor.SampleSet(features, labels, size)

    def __save_samples(self, syms, dates, sample_set, results, prefix):
        values = [syms, dates] + \
                 list(sample_set.features.values()) + \
                 list(sample_set.labels.values()) + \
                 list(results.values())
        samples = pd.concat([value.reset_index(drop=True) for value in values], axis=1)
        samples.to_csv('csvs/{}_samples.csv'.format(prefix), index=False)

    def __load_histories(self, symbols, start_date, end_date):
        start_datetime = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        offset = datetime.timedelta(days=Const.WINDOW_SIZE)
        approx_start_datetime = start_datetime - offset
        approx_start_date = approx_start_datetime.strftime('%Y-%m-%d')
        histories = self.storage.load_histories(symbols, approx_start_date, end_date)
        return histories

    def __build_caches(self, symbols, histories, predictor):
        size = sum([len(history) for history in histories])
        features = self.extractor.extract_and_concat(histories, predictor.get_feature_names())
        sample_set = Predictor.SampleSet(features, None, size)

        results = predictor.predict(sample_set)
        scores = list(results.values())[0].iloc[:, 0]

        caches = {}

        start = 0
        for i, symbol in enumerate(symbols):
            history = histories[i]
            end = start + len(history)
            caches[symbol] = dict(zip(history.Date, scores[start: end]))
            start = end

        return caches
