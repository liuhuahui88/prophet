from unittest import TestCase

from prophet.agent.buy_and_hold_agent import *


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


class TestBuyAndHoldAgent(TestCase):

    def test_handle(self):
        capital_id = '600000'

        agent = BuyAndHoldAgent(capital_id)
        account = Account(1000)

        ctx2 = AgentContext(account, {capital_id: 300})
        agent.handle(ctx2)
        self.assertEqual(ctx2.capital_id, capital_id)
        self.assertEqual(ctx2.volume, 3)

        account.set_cash(100)
        account.set_capital(capital_id, 3)

        ctx2 = AgentContext(account, {capital_id: 30})
        agent.handle(ctx2)
        self.assertEqual(ctx2.capital_id, capital_id)
        self.assertEqual(ctx2.volume, 3)
