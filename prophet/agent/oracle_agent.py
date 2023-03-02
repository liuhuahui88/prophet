from prophet.agent.abstract_agent import Agent
from prophet.data.data_storage import StockDataStorage


class OracleAgent(Agent):

    CASH_UPPER_BOUND = 100000000

    def __init__(self, symbol, storage: StockDataStorage, discount):
        self.symbol = symbol
        self.history = storage.load_history(symbol)
        self.indexes = {self.history.iloc[i].Date: i for i in range(len(self.history))}
        self.discount = discount

    def handle(self, ctx: Agent.Context):
        if ctx.get_account().get_cash() > OracleAgent.CASH_UPPER_BOUND:
            return

        if self.f(ctx.get_date(), self.discount, 1):
            ctx.ask(self.symbol)
        elif not self.f(ctx.get_date(), 1, 1 / self.discount):
            ctx.bid(self.symbol)

    def f(self, date, min_gain, max_gain):
        index = self.indexes[date]
        max_index = len(self.history)

        closes = self.history['Close']
        base = closes.iloc[index]
        for i in range(index, max_index):
            gain = closes.iloc[i] / base
            if gain < min_gain:
                return True
            if gain > max_gain:
                return False
        return True
