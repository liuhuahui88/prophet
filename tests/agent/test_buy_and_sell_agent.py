from unittest import TestCase

from tests.agent.agent_context import *

from prophet.agent.buy_and_sell_agent import *


class TestBuyAndSellAgent(TestCase):

    def test_handle(self):
        capital_id = '600000'

        agent = BuyAndSellAgent(capital_id)
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
        self.assertEqual(ctx2.volume, -3)
