from unittest import TestCase

from prophet.bt.broker import *


class TestBroker(TestCase):

    def test(self):
        broker = Broker(0.1)

        account = Account(1000)
        symbol = '600000'

        commission = broker.calculate_commission(1000)
        self.assertEqual(commission, 100)

        self.assertRaises(ValueError, broker.calculate_commission, -1000)

        broker.trade(account, symbol, 20, -400)
        self.assertEqual(account.get_cash(), 560)

        broker.trade(account, symbol, -20, 400)
        self.assertEqual(account.get_cash(), 920)

        self.assertRaises(ValueError, broker.trade, account, symbol, 20, 400)
        self.assertRaises(ValueError, broker.trade, account, symbol, -20, -400)
