from prophet.agent.abstract_agent import *


class AgentContextForTest(Agent.Context):

    def __init__(self, account: Account, prices: dict):
        self.account = account
        self.prices = prices

        self.capital_id = None
        self.cash = None
        self.volume = None
        self.price = None

    def get_account(self):
        return self.account

    def get_prices(self):
        return self.prices

    def bid(self, capital_id, cash=float('inf'), price=float('inf')):
        self.capital_id = capital_id
        self.cash = cash
        self.price = price

    def ask(self, capital_id, volume=float('inf'), price=0):
        self.capital_id = capital_id
        self.volume = volume
        self.price = price
