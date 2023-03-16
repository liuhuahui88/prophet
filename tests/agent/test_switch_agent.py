from unittest import TestCase

from tests.agent.agent_context import *

from prophet.agent.switch_agent import *


class TestBuyAndSellAgent(TestCase):

    def test_handle(self):
        symbol = '600000'

        agent = SwitchAgent(symbol)
        account = Account(1000)

        ctx1 = AgentContextForTest(account, {symbol: 300}, '2020-01-01')
        agent.handle(ctx1)
        self.assertEqual(ctx1.symbol, symbol)
        self.assertEqual(ctx1.volume, None)
        self.assertEqual(ctx1.cash, float('inf'))
        self.assertEqual(ctx1.price, float('inf'))

        account.set_cash(100)
        account.set_volume(symbol, 3)

        ctx2 = AgentContextForTest(account, {symbol: 30}, '2020-01-02')
        agent.handle(ctx2)
        self.assertEqual(ctx2.symbol, symbol)
        self.assertEqual(ctx2.volume, float('inf'))
        self.assertEqual(ctx2.cash, None)
        self.assertEqual(ctx2.price, 0)
