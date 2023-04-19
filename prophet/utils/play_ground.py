import numpy as np

from prophet.agent.baseline_agent import BaselineAgent
from prophet.agent.ensemble_agent import EnsembleAgent
from prophet.agent.oracle_agent import OracleAgent
from prophet.agent.smart_agent import SmartAgent
from prophet.bt.back_tester import BackTester
from prophet.bt.broker import Broker
from prophet.data.data_extractor import DataExtractor
from prophet.data.data_predictor import DataPredictor
from prophet.data.data_storage import StockDataStorage


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

    def test_smart_agent(self, symbol, start_date, end_date,
                         predictors, delta_free_list=None,
                         with_baseline=False, with_oracle=False):
        bt = BackTester(self.storage, Broker(self.commission_rate))

        for name, predictor in predictors.items():
            delta = 0 if delta_free_list is not None and name in delta_free_list else self.log_friction
            bt.register('SMT_' + name, SmartAgent(symbol, self.storage, self.extractor, predictor, delta))

        if with_baseline:
            bt.register('BASE', BaselineAgent())
        if with_oracle:
            bt.register('ORA', OracleAgent(symbol, self.storage, self.extractor))

        result = bt.back_test([symbol], start_date, end_date)

        return result

    def test_ensemble_agent(self, symbols, start_date, end_date,
                            predictors, delta_free_list=None,
                            with_baseline=False, with_oracle=False):
        bt = BackTester(self.storage, Broker(self.commission_rate))

        for name, predictor in predictors.items():
            delta = 0 if delta_free_list is not None and name in delta_free_list else self.log_friction
            agents = [SmartAgent(s, self.storage, self.extractor, predictor, delta) for s in symbols]
            bt.register('ENS_' + name, EnsembleAgent(agents))
        if with_baseline:
            bt.register('BASE', BaselineAgent())
        for symbol in symbols:
            if with_oracle:
                bt.register('ORA_' + symbol, OracleAgent(symbol, self.storage, self.extractor))

        result = bt.back_test(symbols, start_date, end_date)

        return result
