import datetime

import numpy as np

from prophet.agent.baseline_agent import BaselineAgent
from prophet.agent.oracle_agent import OracleAgent
from prophet.agent.smart_agent import SmartAgent
from prophet.bt.back_tester import BackTester
from prophet.bt.broker import Broker
from prophet.data.data_extractor import DataExtractor
from prophet.data.data_predictor import DataPredictor
from prophet.data.data_storage import StockDataStorage
from prophet.utils.constant import Const


class PlayGround:

    def __init__(self, name_file_path, history_file_path, commission_rate):
        self.storage = StockDataStorage(name_file_path, history_file_path)

        self.commission_rate = commission_rate
        self.log_friction = -(np.log(1 - commission_rate) + np.log(1 / (1 + commission_rate)))

        self.extractor = DataExtractor(commission_rate)

    def train_predictor(self, symbols, start_date, train_end_date, test_end_date,
                        model, batch_size, epochs, monitor, patience, verbose=False, debug=False):
        histories = self.storage.load_histories(symbols, start_date, test_end_date)

        predictor = DataPredictor(model)
        predictor.learn(symbols, histories, train_end_date, self.extractor,
                        batch_size, epochs, monitor, patience, verbose, debug)

        return predictor

    def test_smart_agent(self, symbols, start_date, end_date,
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

    def __load_histories(self, symbols, start_date, end_date):
        start_datetime = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        offset = datetime.timedelta(days=Const.WINDOW_SIZE)
        approx_start_datetime = start_datetime - offset
        approx_start_date = approx_start_datetime.strftime('%Y-%m-%d')
        histories = self.storage.load_histories(symbols, approx_start_date, end_date)
        return histories

    def __build_caches(self, symbols, histories, predictor):
        results = predictor.predict(histories, self.extractor)
        scores = list(results.values())[0].iloc[:, 0]

        caches = {}

        start = 0
        for i, symbol in enumerate(symbols):
            history = histories[i]
            end = start + len(history)
            caches[symbol] = dict(zip(history.Date, scores[start: end]))
            start = end

        return caches
