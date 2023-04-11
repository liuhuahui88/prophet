import numpy as np

from prophet.agent.naive_agent import NaiveAgent
from prophet.agent.smart_agent import SmartAgent
from prophet.bt.back_tester import *
from prophet.data.data_extractor import DataExtractor
from prophet.data.data_predictor import DataPredictor

if __name__ == '__main__':
    storage = StockDataStorage('../data/chinese_stock_codes.csv', '../data/history')

    commission_rate = 0.0
    log_friction = -(np.log(1 - commission_rate) + np.log(1 / (1 + commission_rate)))

    exp_data_predictor = DataPredictor()
    exp_data_predictor.set_data_extractor(DataExtractor(commission_rate))
    exp_data_predictor.load_model('models/experimental')

    sota_data_predictor = DataPredictor()
    sota_data_predictor.set_data_extractor(DataExtractor(commission_rate))
    sota_data_predictor.load_model('models/sota_reg')

    for symbol in [s for s in storage.get_symbols() if s[0] == '3' and s <= '300010']:
        bt = BackTester(storage, Broker(commission_rate))

        bt.register('SMART_EXP', SmartAgent(symbol, exp_data_predictor, 'next_log_gain', 0, log_friction))
        bt.register('SMART_SOTA', SmartAgent(symbol, sota_data_predictor, 'next_log_gain', 0, log_friction))
        bt.register('BUY&HOLD', NaiveAgent(symbol))

        result = bt.back_test([symbol], '2022-01-01', '2023-01-01', verbose=True)
        result.print()
        result.plot('SMART_EXP')
