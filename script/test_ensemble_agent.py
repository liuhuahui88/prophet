import numpy as np

from prophet.agent.ensemble_agent import EnsembleAgent
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

    symbols = [s for s in storage.get_symbols() if s[0] == '3' and s <= '300010']

    bt = BackTester(storage, Broker(commission_rate))

    exp_agents = [SmartAgent(s, exp_data_predictor, 'next_log_gain', 0, log_friction) for s in symbols]
    bt.register('ENSEMBLE_EXP', EnsembleAgent(exp_agents))

    sota_agents = [SmartAgent(s, sota_data_predictor, 'next_log_gain', 0, log_friction) for s in symbols]
    bt.register('ENSEMBLE_SOTA', EnsembleAgent(sota_agents))

    for symbol in symbols:
        bt.register('BUY&HOLD_' + symbol, NaiveAgent(symbol))

    result = bt.back_test(symbols, '2022-01-01', '2023-01-01')
    result.print()
    result.plot()
