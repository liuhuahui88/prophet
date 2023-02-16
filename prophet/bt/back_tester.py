from prophet.data.data_storage import *
from prophet.agent.abstract_agent import *
from prophet.utils.evaluator import *
from prophet.bt.broker import *
from prophet.bt.liquidity import *


class BackTester:

    def __init__(self, broker: Broker, stock_db: StockDataStorage):
        self.broker = broker
        self.stock_db = stock_db

    def back_test(self, agent: Agent, account: Account, code: str):
        history_df = self.stock_db.load_history(code)

        evaluator = Evaluator()
        prices = self.__create_prices(code, history_df, 0)
        evaluator.feed(self.__calculate_value(account, prices))

        for i in range(len(history_df)):
            prices = self.__create_prices(code, history_df, i)
            liquidities = self.__create_liquidities(code, history_df, i)
            agent.handle(BackTester.AgentContext(account, prices, self.broker, liquidities))
            evaluator.feed(self.__calculate_value(account, prices))

        return evaluator

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

    @staticmethod
    def __create_agent_context(account: Account, prices: dict, broker: Broker, liquidities: dict):
        return BackTester.AgentContext(account, prices, broker, liquidities)

    @staticmethod
    def __calculate_value(account: Account, prices: dict):
        cash_value = account.get_cash()

        capital_value = 0
        for capital_id in prices.keys():
            capital_value += prices.get(capital_id) * account.get_capital(capital_id)

        value = cash_value + capital_value
        return value
