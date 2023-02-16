from prophet.back_tester import *
from prophet.agent.buy_and_hold_agent import *


if __name__ == '__main__':
    broker = Broker()
    stock_db = StockDataStorage('../data/chinese_stock_codes.csv', '../data/history')

    bt = BackTester(broker, stock_db)

    code = '605599'
    name = stock_db.get_name(code)

    evaluator = bt.back_test(BuyAndHoldAgent(code), Account(1000000), code)

    print('{} : {}'.format([code, name], evaluator))
