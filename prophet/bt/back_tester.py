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

    def back_test(self, code: str):
        df = self.stock_db.load_history(code)
        cases = [BackTester.TestCase(name, self.agents[name], self.broker, self.init_cash) for name in self.agents]

        for i in range(len(df)):
            prices = self.__create_prices(code, df, i)
            liquidities = self.__create_liquidities(code, df, i)
            for case in cases:
                case.handle(prices, liquidities)

        return cases

    class TestCase:

        def __init__(self, name, agent, broker, init_cash):
            self.name = name
            self.agent = agent
            self.broker = broker
            self.account = Account(init_cash)
            self.evaluator = Evaluator(init_cash)

        def handle(self, prices, liquidities):
            ctx = self.create_agent_context(prices, liquidities)
            self.agent.handle(ctx)

            value = self.calculate_account_value(prices)
            self.evaluator.feed(value)

        def create_agent_context(self, prices: dict, liquidities: dict):
            return BackTester.AgentContext(self.account, prices, self.broker, liquidities)

        def calculate_account_value(self, prices: dict):
            cash_value = self.account.get_cash()

            capital_value = 0
            for capital_id in prices.keys():
                capital_value += prices.get(capital_id) * self.account.get_capital(capital_id)

            value = cash_value + capital_value
            return value

    class AgentContext(Agent.Context):

        def __init__(self, account: Account, prices: dict, broker: Broker, liquidities: dict):
            self.__account = account
            self.__prices = prices
            self.__broker = broker
            self.__liquidities = liquidities

        def get_account(self):
            return self.__account

        def get_prices(self):
            return self.__prices

        def trade(self, capital_id, volume):
            price = self.__liquidities[capital_id].get_price(volume)
            self.__broker.trade(self.__account, capital_id, volume, price)

    @staticmethod
    def __create_prices(code, history: pd.DataFrame, idx):
        return {code: history.loc[idx].Close}

    @staticmethod
    def __create_liquidities(code, history: pd.DataFrame, idx):
        return {code: Liquidity(history.loc[idx].Close)}
