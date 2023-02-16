from unittest import TestCase

from prophet.bt.broker import *


class TestBroker(TestCase):

    def test_trade(self):
        broker = Broker(0.1)

        account = Account(1000)
        capital_id = '600000'

        broker.trade(account, capital_id, 20, 20)
        self.assertEqual(account.get_cash(), 560)

        broker.trade(account, capital_id, -20, 20)
        self.assertEqual(account.get_cash(), 920)
