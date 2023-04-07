from prophet.agent.abstract_agent import *


class AgentContextForTest(Agent.Context):

    def __init__(self, account: Account, prices: dict, volumes: dict, date: str):
        self.account = account
        self.prices = prices
        self.volumes = volumes
        self.date = date

        self.symbol = None
        self.cash = None
        self.volume = None
        self.price = None

    def get_account(self):
        return self.account

    def get_prices(self):
        return self.prices

    def get_volumes(self):
        return self.volumes

    def get_date(self):
        return self.date

    def bid(self, symbol, cash=float('inf'), price=float('inf')):
        self.symbol = symbol
        self.cash = cash
        self.price = price

    def ask(self, symbol, volume=float('inf'), price=0):
        self.symbol = symbol
        self.volume = volume
        self.price = price
