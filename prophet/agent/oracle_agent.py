from prophet.agent.abstract_agent import *
from prophet.data.data_storage import *


class OracleAgent(Agent):

    CASH_MULTIPLIER = 1000

    def __init__(self, capital_id, ds: StockDataStorage, window_size):
        self.capital_id = capital_id

        self.df = ds.load_history(capital_id)
        self.df['NextClose'] = self.df['Close'].rolling(window_size).median().shift(-window_size)

        self.index = 0
        self.target_cash = None

    def handle(self, ctx: Agent.Context):
        if self.index == 0:
            self.target_cash = ctx.get_account().get_cash() * OracleAgent.CASH_MULTIPLIER

        if ctx.get_account().get_cash() < self.target_cash:
            close = self.df['Close'].iloc[self.index]
            next_close = self.df['NextClose'].iloc[self.index]

            if close > next_close:
                volume = ctx.get_account().get_capital(self.capital_id)
                ctx.trade(self.capital_id, -volume)
            else:
                cash = ctx.get_account().get_cash()
                price = ctx.get_prices()[self.capital_id]
                volume = int(cash / price)
                ctx.trade(self.capital_id, volume)

        self.index += 1
