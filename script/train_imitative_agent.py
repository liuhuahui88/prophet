from prophet.agent.buy_and_hold_agent import BuyAndHoldAgent
from prophet.agent.imitative_agent import ImitativeAgent
from prophet.agent.perfect_indicator_agent import PerfectIndicatorAgent
from prophet.bt.back_tester import *

if __name__ == '__main__':
    storage = StockDataStorage('../data/chinese_stock_codes.csv', '../data/history')

    commission_rate = 0.01

    bt = BackTester(storage, Broker(commission_rate))

    symbol = '600000'

    start_date = '2010-01-01'
    train_end_date = '2011-01-01'
    test_end_date = '2012-01-01'

    bt.register('PIA', PerfectIndicatorAgent(symbol, storage, commission_rate))

    result = bt.back_test(symbol, start_date, train_end_date)

    imitative_agent = ImitativeAgent(symbol, commission_rate)
    imitative_agent.observe(result.history, result.cases[0].actions)
    bt.register('IMI', imitative_agent)

    bt.register('B&H', BuyAndHoldAgent(symbol))

    result = bt.back_test(symbol, train_end_date, test_end_date)
    result.print()
    result.plot('IMI')
