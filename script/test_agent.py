from prophet.agent.buy_and_hold_agent import BuyAndHoldAgent
from prophet.agent.switch_agent import SwitchAgent
from prophet.agent.moving_average_agent import MovingAverageAgent
from prophet.agent.perfect_indicator_agent import PerfectIndicatorAgent
from prophet.agent.perfect_action_agent import PerfectActionAgent
from prophet.bt.back_tester import *

if __name__ == '__main__':
    storage = StockDataStorage('../data/chinese_stock_codes.csv', '../data/history')

    commission_rate = 0.01

    bt = BackTester(storage, Broker(commission_rate))

    symbol = '600000'

    bt.register('B&H', BuyAndHoldAgent(symbol))
    bt.register('B&S', SwitchAgent(symbol, False))
    bt.register('S&B', SwitchAgent(symbol, True))
    bt.register('MAA', MovingAverageAgent(symbol, 5, 10))
    bt.register('PIA', PerfectIndicatorAgent(symbol, storage, commission_rate))
    bt.register('PAA', PerfectActionAgent(symbol, storage, commission_rate))

    result = bt.back_test([symbol], '2014-01-01', '2016-01-01')
    result.print()
    result.plot()
