from prophet.agent.buy_and_hold_agent import BuyAndHoldAgent
from prophet.agent.buy_and_sell_agent import BuyAndSellAgent
from prophet.agent.moving_average_agent import MovingAverageAgent
from prophet.agent.oracle_agent import OracleAgent
from prophet.agent.perfect_agent import PerfectAgent
from prophet.bt.back_tester import *

if __name__ == '__main__':
    storage = StockDataStorage('../data/chinese_stock_codes.csv', '../data/history')

    commission_rate = 0.01
    discount = (1 - commission_rate) / (1 + commission_rate)

    bt = BackTester(storage, Broker(commission_rate))

    symbol = '600000'

    bt.register('B&H', BuyAndHoldAgent(symbol))
    bt.register('B&S', BuyAndSellAgent(symbol, False))
    bt.register('S&B', BuyAndSellAgent(symbol, True))
    bt.register('MAA', MovingAverageAgent(symbol, 5, 10))
    bt.register('ORA', OracleAgent(symbol, storage))
    bt.register('PER', PerfectAgent(symbol, storage))

    result = bt.back_test(symbol, '2014-01-01', '2016-01-01')
    result.print()
    result.plot()
