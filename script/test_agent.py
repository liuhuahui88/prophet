from prophet.agent import *
from prophet.bt.back_tester import *
from prophet.utils.figure import *


if __name__ == '__main__':
    stock_db = StockDataStorage('../data/chinese_stock_codes.csv', '../data/history')
    broker = Broker()
    bt = BackTester(stock_db, broker)

    symbol = '600000'
    name = stock_db.get_name(symbol)

    bt.register('B&H', BuyAndHoldAgent(symbol))
    bt.register('B&S', BuyAndSellAgent(symbol, False))
    bt.register('S&B', BuyAndSellAgent(symbol, True))
    bt.register('MAA', MovingAverageAgent(symbol, 5, 10))
    bt.register('ORA', OracleAgent(symbol, stock_db, 1))

    df, cases = bt.back_test(symbol, '2014-01-01', '2014-02-01')

    value_names = []
    for case in cases:
        print('{} : {} : {}'.format([symbol, name], case.name, case.evaluator))

        value_name = 'V_' + case.name
        value_names.append(value_name)
        df[value_name] = case.evaluator.values[1:]

    figure = Figure(value_names=value_names)
    figure.plot(df, str([symbol, name]))
