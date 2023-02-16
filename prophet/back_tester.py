import numpy as np

from prophet.data_storage import *
from prophet.exchange import *
from prophet.agent.abstract_agent import *


class Evaluator:

    def __init__(self):
        self.values = []

    def feed(self, value):
        self.values.append(value)

    def get_gain_cum(self):
        return self.__calculate_gain(self.values[0], self.values[-1])

    def get_gain_avg(self):
        return np.exp(np.log(self.get_gain_cum()) / (len(self.values) - 1))

    def get_gain_std(self):
        log_gains = []
        for i in range(len(self.values) - 1):
            gain = self.__calculate_gain(self.values[i], self.values[i + 1])
            log_gains.append(np.log(gain))
        return np.exp(np.std(log_gains))

    def get_sharp_ratio(self):
        return 1 + (self.get_gain_avg() - 1) / self.get_gain_std()

    def get_worst_drawdown(self):
        worst_drawdown = 1
        max_value = min_value = self.values[0]
        for value in self.values:
            if value > max_value:
                max_value = min_value = value
            elif value < min_value:
                min_value = value
                drawdown = self.__calculate_gain(max_value, min_value)
                worst_drawdown = min(drawdown, worst_drawdown)
        return worst_drawdown

    def __str__(self):
        return 'gain_cum={:.4f}, gain_avg={:.4f}, gain_std={:.4f}, sharp_ratio={:.4f}, worst_drawdown={:.4f}'.format(
            self.get_gain_cum(),
            self.get_gain_avg(),
            self.get_gain_std(),
            self.get_sharp_ratio(),
            self.get_worst_drawdown())

    @staticmethod
    def __calculate_gain(value1, value2):
        return value2 / value1


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