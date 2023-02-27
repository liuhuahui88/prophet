from prophet.agent.abstract_agent import *


class AgentContextForTest(Agent.Context):

    def __init__(self, account: Account, prices: dict):
        self.account = account
        self.prices = prices

        self.symbol = None
        self.cash = None
        self.volume = None
        self.price = None

    def get_account(self):
        return self.account

    def get_prices(self):
        return self.prices

    def bid(self, symbol, cash=float('inf'), price=float('inf')):
        self.symbol = symbol
        self.cash = cash
        self.price = price

    def ask(self, symbol, volume=float('inf'), price=0):
        self.symbol = symbol
        self.volume = volume
        self.price = price
