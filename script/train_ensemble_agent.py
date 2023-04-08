from prophet.agent.ensemble_agent import EnsembleAgent
from prophet.agent.naive_agent import NaiveAgent
from prophet.agent.perfect_action_agent import PerfectActionAgent
from prophet.agent.smart_agent import SmartAgent
from prophet.bt.back_tester import *

if __name__ == '__main__':
    storage = StockDataStorage('../data/chinese_stock_codes.csv', '../data/history')

    commission_rate = 0.0

    bt = BackTester(storage, Broker(commission_rate))

    symbols = [s for s in storage.get_symbols() if s[0] == '3' and s <= '300003']

    start_date = '2010-01-01'
    train_end_date = '2022-01-01'
    test_end_date = '2023-01-01'

    agents = []
    for symbol in symbols:
        result = bt.back_test([symbol], start_date, train_end_date)
        smart_agent = SmartAgent(symbol, commission_rate)
        smart_agent.observe(result.histories[0])
        agents.append(smart_agent)
    bt.register('ENS', EnsembleAgent(agents))

    for symbol in symbols:
        bt.register('B&H-' + symbol, NaiveAgent(symbol))
        bt.register('PAA-' + symbol, PerfectActionAgent(symbol, storage, commission_rate))

    result = bt.back_test(symbols, train_end_date, test_end_date)
    result.print()
    result.plot()
