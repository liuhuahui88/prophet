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
    extractor = DataExtractor(commission_rate)

    exp_reg_predictor = DataPredictor.load('models/exp_reg')
    sota_reg_predictor = DataPredictor.load('models/sota_reg')
    exp_cls_predictor = DataPredictor.load('models/exp_cls')
    sota_cls_predictor = DataPredictor.load('models/sota_cls')

    for symbol in storage.get_symbols(lambda s: s[0] == '3' and s <= '300010'):
        bt = BackTester(storage, Broker(commission_rate))

        bt.register('SMART_EXP_REG', SmartAgent(symbol, storage, extractor, exp_reg_predictor, log_friction))
        bt.register('SMART_SOTA_REG', SmartAgent(symbol, storage, extractor, sota_reg_predictor, log_friction))
        bt.register('SMART_EXP_CLS', SmartAgent(symbol, storage, extractor, exp_cls_predictor, log_friction))
        bt.register('SMART_SOTA_CLS', SmartAgent(symbol, storage, extractor, sota_cls_predictor, log_friction))

        bt.register('BUY&HOLD', NaiveAgent(symbol))

        result = bt.back_test([symbol], '2022-01-01', '2023-01-01')
        result.print()
        result.plot()
