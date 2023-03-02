from prophet.agent.buy_and_hold_agent import BuyAndHoldAgent
from prophet.agent.buy_and_sell_agent import BuyAndSellAgent
from prophet.agent.moving_average_agent import MovingAverageAgent
from prophet.agent.oracle_agent import OracleAgent
from prophet.utils.figure import Figure
from prophet.bt.back_tester import *


if __name__ == '__main__':
    stock_db = StockDataStorage('../data/chinese_stock_codes.csv', '../data/history')
    broker = Broker(0.01)
    bt = BackTester(stock_db, broker)

    symbol = '600000'
    name = stock_db.get_name(symbol)

    bt.register('B&H', BuyAndHoldAgent(symbol))
    bt.register('B&S', BuyAndSellAgent(symbol, False))
    bt.register('S&B', BuyAndSellAgent(symbol, True))
    bt.register('MAA', MovingAverageAgent(symbol, 5, 10))
    bt.register('ORA', OracleAgent(symbol, stock_db, 0.99 / 1.01))

    df, cases = bt.back_test(symbol, '2014-01-01', '2016-01-01')

    value_names = []
    for case in cases:
        print('{} : {} : {}'.format([symbol, name], case.name, case.evaluator))

        value_name = 'V_' + case.name
        value_names.append(value_name)
        df[value_name] = case.evaluator.values[1:]

    figure = Figure(value_names=value_names)
    figure.plot(df, str([symbol, name]))
