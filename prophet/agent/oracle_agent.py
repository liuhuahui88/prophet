from prophet.agent.abstract_agent import *
from prophet.data.data_storage import *


class OracleAgent(Agent):

    CASH_UPPER_BOUND = 100000000

    def __init__(self, symbol, ds: StockDataStorage, window_size):
        self.symbol = symbol

        self.df = ds.load_history(symbol)
        self.df['NextClose'] = self.df['Close'].rolling(window_size).median().shift(-window_size)

        self.indexes = {self.df.iloc[i].Date: i for i in range(len(self.df))}

    def handle(self, ctx: Agent.Context):
        if ctx.get_account().get_cash() > OracleAgent.CASH_UPPER_BOUND:
            return

        index = self.indexes[ctx.get_date()]

        close = self.df['Close'].iloc[index]
        next_close = self.df['NextClose'].iloc[index]

        if close > next_close:
            ctx.ask(self.symbol)
        else:
            ctx.bid(self.symbol)
