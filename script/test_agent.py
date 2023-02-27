from prophet.bt.back_tester import *
from prophet.agent.buy_and_hold_agent import *
from prophet.agent.buy_and_sell_agent import *
from prophet.utils.figure import *


if __name__ == '__main__':
    stock_db = StockDataStorage('../data/chinese_stock_codes.csv', '../data/history')
    broker = Broker()
    bt = BackTester(stock_db, broker)

    symbol = '605599'
    name = stock_db.get_name(symbol)

    bt.register('B&H', BuyAndHoldAgent(symbol))
    bt.register('B&S', BuyAndSellAgent(symbol))

    cases = bt.back_test(symbol)

    df = stock_db.load_history(symbol)

    value_names = []
    for case in cases:
        print('{} : {} : {}'.format([symbol, name], case.name, case.evaluator))

        value_name = 'V_' + case.name
        value_names.append(value_name)
        df[value_name] = case.evaluator.values[1:]

    figure = Figure(value_names=value_names)
    figure.plot(df, str([symbol, name]))
