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
    extractor = DataExtractor(commission_rate)

    exp_reg_predictor = DataPredictor.load('models/exp_reg')
    sota_reg_predictor = DataPredictor.load('models/sota_reg')
    exp_cls_predictor = DataPredictor.load('models/exp_cls')
    sota_cls_predictor = DataPredictor.load('models/sota_cls')

    symbols = storage.get_symbols(lambda s: s[0] == '3' and s <= '300010')

    bt = BackTester(storage, Broker(commission_rate))

    exp_agents = [SmartAgent(s, storage, extractor, exp_reg_predictor, log_friction) for s in symbols]
    bt.register('ENSEMBLE_EXP_REG', EnsembleAgent(exp_agents))

    sota_agents = [SmartAgent(s, storage, extractor, sota_reg_predictor, log_friction) for s in symbols]
    bt.register('ENSEMBLE_SOTA_REG', EnsembleAgent(sota_agents))

    exp_agents = [SmartAgent(s, storage, extractor, exp_cls_predictor, log_friction) for s in symbols]
    bt.register('ENSEMBLE_EXP_CLS', EnsembleAgent(exp_agents))

    sota_agents = [SmartAgent(s, storage, extractor, sota_cls_predictor, log_friction) for s in symbols]
    bt.register('ENSEMBLE_SOTA_CLS', EnsembleAgent(sota_agents))

    for symbol in symbols:
        bt.register('BUY&HOLD_' + symbol, NaiveAgent(symbol))

    result = bt.back_test(symbols, '2022-01-01', '2023-01-01')
    result.print()
    result.plot()
