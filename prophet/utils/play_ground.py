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

    def train(self, symbols, start_date, train_end_date, test_end_date, predictor: Predictor, axis,
              debug_train=False, debug_test=False, debug_features=False):
        histories = self.__load_histories(symbols, start_date, test_end_date, axis == 1)
        sample_set = self.__create_sample_set(symbols, histories, predictor, axis)
        train_sample_set = self.__split_sample_set(sample_set, sample_set.ids['date'] < train_end_date)
        test_sample_set = self.__split_sample_set(sample_set, sample_set.ids['date'] >= train_end_date)

        predictor.fit(train_sample_set, test_sample_set)

        if debug_train:
            train_results = predictor.predict(train_sample_set)
            self.__save_debug_info(train_sample_set, train_results, debug_features, 'train')

        if debug_test:
            test_results = predictor.predict(test_sample_set)
            self.__save_debug_info(test_sample_set, test_results, debug_features, 'test')

    def test(self, symbols, start_date, end_date, predictors, axis,
             delta_free_list=None, global_threshold=0, local_threshold=0, top_k=1, weighted=False,
             with_baseline=False, with_oracle=False, verbose_bt=False, verbose_agent=False):
        bt = BackTester(self.storage, Broker(self.commission_rate))
        histories = self.__load_histories(symbols, start_date, end_date, axis == 1, Const.WINDOW_SIZE)

        for name, predictor in predictors.items():
            caches = self.__build_caches(symbols, histories, predictor, axis)
            delta = 0 if delta_free_list is not None and name in delta_free_list else self.log_friction
            agent = SmartAgent(caches, delta, global_threshold, local_threshold, top_k, weighted, verbose_agent)
            bt.register('SMT_' + name, agent)

        if with_baseline:
            bt.register('BASE', BaselineAgent())
        for symbol in symbols:
            if with_oracle:
                bt.register('ORA_' + symbol, OracleAgent(symbol, self.storage, self.extractor))

        result = bt.back_test(symbols, start_date, end_date, verbose=verbose_bt)

        return result

    def __create_sample_set(self, symbols, histories, predictor, axis):
        dates = self.extractor.extract_and_concat(histories, ['date'], axis)['date'].iloc[:, 0]
        features = self.extractor.extract_and_concat(histories, predictor.get_feature_names(), axis)
        labels = self.extractor.extract_and_concat(histories, predictor.get_label_names(), axis)
        size = sum([len(h) for h in histories]) if axis == 0 else len(histories[0])
        ids = dict(date=dates) if axis == 1 else dict(date=dates, symbol=self.__broadcast_symbols(symbols, histories))
        return Predictor.SampleSet(ids, features, labels, size)

    def __broadcast_symbols(self, symbols, histories):
        symbols_df_list = []
        for i in range(len(histories)):
            if len(histories[i]) != 0:
                s = [symbols[i]] * len(histories[i])
                symbols_df = pd.DataFrame({'Symbol': s})
                symbols_df_list.append(symbols_df)
        return pd.concat(symbols_df_list).reset_index(drop=True)

    def __split_sample_set(self, sample_set, condition):
        ids = {k: v[condition] for k, v in sample_set.ids.items()}
        features = {k: v[condition] for k, v in sample_set.features.items()}
        labels = {k: v[condition] for k, v in sample_set.labels.items()}
        return Predictor.SampleSet(ids, features, labels, condition.sum())

    def __save_debug_info(self, sample_set, results, debug_features, prefix):
        values = list(sample_set.ids.values()) + \
                 list(sample_set.features.values() if debug_features else []) + \
                 list(sample_set.labels.values()) + \
                 list(results.values())
        samples = pd.concat([value.reset_index(drop=True) for value in values], axis=1)
        samples.to_csv('csvs/{}_samples.csv'.format(prefix), index=False)

    def __load_histories(self, symbols, start_date, end_date, expands, offset=0):
        start_datetime = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        offset = datetime.timedelta(days=offset)
        approx_start_datetime = start_datetime - offset
        approx_start_date = approx_start_datetime.strftime('%Y-%m-%d')
        histories = self.storage.load_histories(symbols, approx_start_date, end_date, expands)
        return histories

    def __build_caches(self, symbols, histories, predictor, axis):
        size = sum([len(history) for history in histories])
        features = self.extractor.extract_and_concat(histories, predictor.get_feature_names(), axis=axis)
        sample_set = Predictor.SampleSet(None, features, None, size)

        results = predictor.predict(sample_set)
        scores = list(results.values())[0]

        caches = {}

        start = 0
        for i, symbol in enumerate(symbols):
            history = histories[i]
            end = start + len(history)
            s = scores.iloc[start: end, 0] if axis == 0 else scores.iloc[:, i]
            caches[symbol] = dict(zip(history.Date, s)) if len(history) != 0 else dict()
            start = end

        return caches
