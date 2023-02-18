from prophet.agent.buy_and_hold_agent import *
from prophet.bt.back_tester import *

if __name__ == '__main__':
    stock_db = StockDataStorage('../data/chinese_stock_codes.csv', '../data/history')
    broker = Broker()
    bt = BackTester(stock_db, broker)

    code = '605599'
    name = stock_db.get_name(code)

    bt.register('Base', BuyAndHoldAgent(code))

    cases = bt.back_test(code)

    for case in cases:
        print('{} : {} : {}'.format([code, name], case.name, case.evaluator))
