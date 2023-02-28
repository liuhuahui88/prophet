import math

from prophet.data.data_storage import *
from prophet.agent.abstract_agent import *
from prophet.utils.evaluator import *
from prophet.bt.broker import *
from prophet.bt.liquidity import *


class BackTester:

    def __init__(self, stock_db: StockDataStorage, broker: Broker, init_cash=1000000):
        self.stock_db = stock_db
        self.broker = broker
        self.init_cash = init_cash
        self.agents = {}

    def register(self, name: str, agent: Agent):
        self.agents[name] = agent

    def back_test(self, symbol: str):
        df = self.stock_db.load_history(symbol)
        cases = [BackTester.TestCase(name, self.agents[name], self.broker, self.init_cash) for name in self.agents]

        for i in range(len(df)):
            date = self.__create_date(df, i)
            prices = self.__create_prices(symbol, df, i)
            liquidities = self.__create_liquidities(symbol, df, i)
            for case in cases:
                case.handle(date, prices, liquidities)

        return cases

    class TestCase:

        def __init__(self, name, agent, broker, init_cash):
            self.name = name
            self.agent = agent
            self.broker = broker
            self.account = Account(init_cash)
            self.evaluator = Evaluator(init_cash)

        def handle(self, date, prices, liquidities):
            ctx = self.create_agent_context(date, prices, liquidities)
            self.agent.handle(ctx)

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

        def __init__(self, broker: Broker, account: Account, date: str, prices: dict, liquidities: dict):
            self.__broker = broker
            self.__account = account
            self.__date = date
            self.__prices = prices
            self.__liquidities = liquidities

        def get_account(self):
            return self.__account

        def get_prices(self):
            return self.__prices

        def get_date(self):
            return self.__date

        def bid(self, symbol, cash=float('inf'), price=float('inf')):
            cash = min(cash, self.__account.get_cash())
            cash = cash - self.__broker.calculate_commission(cash)

            volume, cash = self.__liquidities[symbol].bid(cash, price)
            if volume != 0:
                self.__broker.trade(self.__account, symbol, volume, -cash)

        def ask(self, symbol, volume=float('inf'), price=0):
            volume = min(volume, self.__account.get_volume(symbol))

            volume, cash = self.__liquidities[symbol].ask(volume, price)
            if volume != 0:
                self.__broker.trade(self.__account, symbol, -volume, cash)

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
