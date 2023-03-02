from prophet.agent.buy_and_hold_agent import BuyAndHoldAgent
from prophet.agent.buy_and_sell_agent import BuyAndSellAgent
from prophet.agent.moving_average_agent import MovingAverageAgent
from prophet.agent.oracle_agent import OracleAgent
from prophet.bt.back_tester import *

if __name__ == '__main__':
    storage = StockDataStorage('../data/chinese_stock_codes.csv', '../data/history')

    bt = BackTester(storage, Broker(0.01))

    symbol = '600000'

    bt.register('B&H', BuyAndHoldAgent(symbol))
    bt.register('B&S', BuyAndSellAgent(symbol, False))
    bt.register('S&B', BuyAndSellAgent(symbol, True))
    bt.register('MAA', MovingAverageAgent(symbol, 5, 10))
    bt.register('ORA', OracleAgent(symbol, storage, 0.99 / 1.01))

    result = bt.back_test(symbol, '2014-01-01', '2016-01-01')
    result.print()
    result.plot()
