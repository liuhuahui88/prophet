from prophet.agent.abstract_agent import *


class AgentContext(Agent.Context):

    def __init__(self, account: Account, prices: dict):
        self.account = account
        self.prices = prices

        self.capital_id = None
        self.volume = None

    def get_account(self):
        return self.account

    def get_prices(self):
        return self.prices

    def trade(self, capital_id, volume):
        self.capital_id = capital_id
        self.volume = volume
