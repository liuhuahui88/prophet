import math

import pandas as pd

from prophet.data.data_storage import StockDataStorage
from prophet.agent.abstract_agent import Agent
from prophet.utils.account import Account
from prophet.utils.evaluator import Evaluator
from prophet.bt.broker import Broker
from prophet.bt.liquidity import Liquidity
from prophet.utils.figure import Figure
from prophet.utils.constant import Const


class BackTester:

    def __init__(self, storage: StockDataStorage, broker: Broker, init_cash=1000000):
        self.storage = storage
        self.broker = broker
        self.init_cash = init_cash
        self.agents = {}

    def register(self, name: str, agent: Agent):
        self.agents[name] = agent

    def back_test(self, symbols, start_date=None, end_date=None):
        names = [self.storage.get_name(symbol) for symbol in symbols]
        histories = [self.__load_history(symbol, start_date, end_date) for symbol in symbols]

        cases = [BackTester.TestCase(name, self.agents[name], self.broker, self.init_cash) for name in self.agents]

        for i in range(len(histories[0])):
            date = self.__create_date(histories, i)
            prices = self.__create_prices(symbols, histories, i)
            volumes = self.__create_volumes(symbols, histories, i)
            liquidities = self.__create_liquidities(symbols, histories, i)
            for case in cases:
                case.handle(date, prices, volumes, liquidities)

        return BackTester.TestResult(symbols, names, histories, cases)

    class TestResult:

        def __init__(self, symbols, names, histories, cases):
            self.symbols = symbols
            self.names = names
            self.histories = histories
            self.cases = cases

        def print(self):
            for case in self.cases:
                print('{} : {} : {}'.format([self.symbols, self.names], case.name, case.evaluator))

        def plot(self, action_agent_name=None):
            temp_history = self.histories[0].copy()

            value_names = []
            for case in self.cases:
                value_name = 'V_' + case.name
                value_names.append(value_name)
                temp_history[value_name] = case.evaluator.values[1:]

            if action_agent_name is None:
                action_name = None
            else:
                case = next(x for x in self.cases if x.name == action_agent_name)
                action_name = 'A_' + case.name
                temp_history[action_name] = case.actions

            figure = Figure(action_name=action_name, value_names=value_names)
            figure.plot(temp_history, str([self.symbols, self.names]))

    class TestCase:

        def __init__(self, name, agent, broker, init_cash):
            self.name = name
            self.agent = agent
            self.broker = broker
            self.account = Account(init_cash)
            self.actions = []
            self.evaluator = Evaluator(init_cash)

        def handle(self, date, prices, volumes, liquidities):
            ctx = self.create_agent_context(date, prices, volumes, liquidities)
            self.agent.handle(ctx)

            action = ctx.get_action()
            self.actions.append(action)

            value = self.calculate_account_value(prices)
            self.evaluator.feed(value)

        def create_agent_context(self, date: str, prices: dict, volumes: dict, liquidities: dict):
            return BackTester.AgentContext(self.broker, self.account, date, prices, volumes, liquidities)

        def calculate_account_value(self, prices: dict):
            cash_value = self.account.get_cash()

            capital_value = 0
            for symbol in prices.keys():
                capital_value += prices.get(symbol) * self.account.get_volume(symbol)

            value = cash_value + capital_value
            return value

    class AgentContext(Agent.Context):

        def __init__(self, broker: Broker, account: Account, date: str, prices: dict, volumes: dict, liquidities: dict):
            self.__broker = broker
            self.__account = account
            self.__date = date
            self.__prices = prices
            self.__volumes = volumes
            self.__liquidities = liquidities
            self.__action = None

        def get_account(self):
            return self.__account

        def get_prices(self):
            return self.__prices

        def get_volumes(self):
            return self.__volumes

        def get_date(self):
            return self.__date

        def get_action(self):
            return self.__action

        def bid(self, symbol, cash=float('inf'), price=float('inf')):
            self.__action = Const.BID

            cash = min(cash, self.__account.get_cash())
            cash = cash - self.__broker.calculate_commission(cash)

            volume, cash = self.__liquidities[symbol].bid(cash, price)
            if volume != 0:
                self.__broker.trade(self.__account, symbol, volume, -cash)

        def ask(self, symbol, volume=float('inf'), price=0):
            self.__action = Const.ASK

            volume = min(volume, self.__account.get_volume(symbol))

            volume, cash = self.__liquidities[symbol].ask(volume, price)
            if volume != 0:
                self.__broker.trade(self.__account, symbol, -volume, cash)

    def __load_history(self, symbol, start_date, end_date):
        history = self.storage.load_history(symbol)
        if start_date is not None:
            history = history[history.Date >= start_date]
        if end_date is not None:
            history = history[history.Date < end_date]
        history = history.reset_index(drop=True)
        return history

    @staticmethod
    def __create_date(histories, idx):
        return histories[0].iloc[idx].Date

    @staticmethod
    def __create_prices(symbols, histories, idx):
        res = {}
        for i in range(len(symbols)):
            res[symbols[i]] = histories[i].iloc[idx].Close
        return res

    @staticmethod
    def __create_volumes(symbols, histories, idx):
        res = {}
        for i in range(len(symbols)):
            res[symbols[i]] = histories[i].iloc[idx].Volume * 1.0
        return res

    @staticmethod
    def __create_liquidities(symbols, histories, idx):
        res = {}
        for i in range(len(symbols)):
            diff = math.log(histories[i].iloc[idx].Close / histories[i].iloc[idx - 1].Close) if idx != 0 else 0
            threshold = math.log(1.19 if symbols[i][0] == '3' else 1.09) if idx != 0 else 0
            has_ask = diff < threshold
            has_bid = diff > -threshold
            res[symbols[i]] = Liquidity(histories[i].iloc[idx].Close, 0, has_ask, has_bid)
        return res
