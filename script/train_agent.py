from prophet.agent.oracle_agent import OracleAgent
from prophet.agent.imitative_agent import ImitativeAgent
from prophet.agent.buy_and_hold_agent import BuyAndHoldAgent
from prophet.utils.figure import Figure
from prophet.bt.back_tester import *


if __name__ == '__main__':
    stock_db = StockDataStorage('../data/chinese_stock_codes.csv', '../data/history')
    broker = Broker(0.01)
    bt = BackTester(stock_db, broker)

    symbol = '600000'
    name = stock_db.get_name(symbol)

    start_date = '2010-01-01'
    train_end_date = '2011-01-01'
    test_end_date = '2012-01-01'

    bt.register('ORA', OracleAgent(symbol, stock_db, 0.99 / 1.01))

    df, cases = bt.back_test(symbol, start_date, train_end_date)

    imitative_agent = ImitativeAgent(symbol, 30)
    imitative_agent.observe(df, cases[0].actions)

    bt.register('IMI', imitative_agent)

    bt.register('B&H', BuyAndHoldAgent(symbol))

    df, cases = bt.back_test(symbol, train_end_date, test_end_date)

    value_names = []
    for case in cases:
        print('{} : {} : {}'.format([symbol, name], case.name, case.evaluator))

        value_name = 'V_' + case.name
        value_names.append(value_name)
        df[value_name] = case.evaluator.values[1:]

    figure = Figure(value_names=value_names)
    figure.plot(df, str([symbol, name]))

