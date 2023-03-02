import math

import pandas as pd

from prophet.data.data_storage import StockDataStorage
from prophet.agent.abstract_agent import Agent
from prophet.utils.account import Account
from prophet.utils.evaluator import Evaluator
from prophet.bt.broker import Broker
from prophet.bt.liquidity import Liquidity
from prophet.utils.figure import Figure


class BackTester:

    def __init__(self, storage: StockDataStorage, broker: Broker, init_cash=1000000):
        self.storage = storage
        self.broker = broker
        self.init_cash = init_cash
        self.agents = {}

    def register(self, name: str, agent: Agent):
        self.agents[name] = agent

    def back_test(self, symbol: str, start_date=None, end_date=None):
        name = self.storage.get_name(symbol)
        history = self.__load_history(symbol, start_date, end_date)

        cases = [BackTester.TestCase(name, self.agents[name], self.broker, self.init_cash) for name in self.agents]

        for i in range(len(history)):
            date = self.__create_date(history, i)
            prices = self.__create_prices(symbol, history, i)
            liquidities = self.__create_liquidities(symbol, history, i)
            for case in cases:
                case.handle(date, prices, liquidities)

        return BackTester.TestResult(symbol, name, history, cases)

    class TestResult:

        def __init__(self, symbol, name, history, cases):
            self.symbol = symbol
            self.name = name
            self.history = history
            self.cases = cases

        def print(self):
            for case in self.cases:
                print('{} : {} : {}'.format([self.symbol, self.name], case.name, case.evaluator))

        def plot(self):
            temp_history = self.history.copy()

            value_names = []
            for case in self.cases:
                value_name = 'V_' + case.name
                value_names.append(value_name)
                temp_history[value_name] = case.evaluator.values[1:]

            figure = Figure(value_names=value_names)
            figure.plot(temp_history, str([self.symbol, self.name]))

    class TestCase:

        def __init__(self, name, agent, broker, init_cash):
            self.name = name
            self.agent = agent
            self.broker = broker
            self.account = Account(init_cash)
            self.actions = []
            self.evaluator = Evaluator(init_cash)

        def handle(self, date, prices, liquidities):
            ctx = self.create_agent_context(date, prices, liquidities)
            self.agent.handle(ctx)

            action = ctx.get_action()
            self.actions.append(action)

            value = self.calculate_account_value(prices)
            self.evaluator.feed(value)

        def create_agent_context(self, date: str, prices: dict, liquidities: dict):
            return BackTester.AgentContext(self.broker, self.account, date, prices, liquidities)

        def calculate_account_value(self, prices: dict):
            cash_value = self.account.get_cash()

            capital_value = 0
            for symbol in prices.keys():
                capital_value += prices.get(symbol) * self.account.get_volume(symbol)

            value = cash_value + capital_value
            return value

    class AgentContext(Agent.Context):

        BID = 1
        ASK = 0

        def __init__(self, broker: Broker, account: Account, date: str, prices: dict, liquidities: dict):
            self.__broker = broker
            self.__account = account
            self.__date = date
            self.__prices = prices
            self.__liquidities = liquidities
            self.__action = None

        def get_account(self):
            return self.__account

        def get_prices(self):
            return self.__prices

        def get_date(self):
            return self.__date

        def get_action(self):
            return self.__action

        def bid(self, symbol, cash=float('inf'), price=float('inf')):
            assert self.__action is None
            self.__action = self.BID

            cash = min(cash, self.__account.get_cash())
            cash = cash - self.__broker.calculate_commission(cash)

            volume, cash = self.__liquidities[symbol].bid(cash, price)
            if volume != 0:
                self.__broker.trade(self.__account, symbol, volume, -cash)

        def ask(self, symbol, volume=float('inf'), price=0):
            assert self.__action is None
            self.__action = self.ASK

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
    def __create_date(history: pd.DataFrame, idx):
        return history.iloc[idx].Date

    @staticmethod
    def __create_prices(symbol, history: pd.DataFrame, idx):
        return {symbol: history.iloc[idx].Close}

    @staticmethod
    def __create_liquidities(symbol, history: pd.DataFrame, idx):
        diff = math.log(history.iloc[idx].Close / history.iloc[idx - 1].Close) if idx != 0 else 0
        threshold = math.log(1.19 if symbol[0] == '3' else 1.09) if idx != 0 else 0
        has_ask = diff < threshold
        has_bid = diff > -threshold
        return {symbol: Liquidity(history.iloc[idx].Close, 0, has_ask, has_bid)}
