from prophet.agent.moving_average_agent import MovingAverageAgent
from prophet.agent.naive_agent import NaiveAgent
from prophet.agent.oracle_agent import OracleAgent
from prophet.bt.back_tester import *

if __name__ == '__main__':
    storage = StockDataStorage('../data/chinese_stock_codes.csv', '../data/history')

    commission_rate = 0.01

    bt = BackTester(storage, Broker(commission_rate))

    symbol = '600000'

    bt.register('B&H', NaiveAgent(symbol))
    bt.register('B&S', NaiveAgent(symbol, False, True))
    bt.register('S&B', NaiveAgent(symbol, True, True))
    bt.register('MAA', MovingAverageAgent(symbol, 5, 10))
    bt.register('ORA', OracleAgent(symbol, storage, commission_rate))

    result = bt.back_test([symbol], '2014-01-01', '2016-01-01')
    result.print()
    result.plot()
