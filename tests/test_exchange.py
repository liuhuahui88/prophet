from unittest import TestCase

from prophet.exchange import Account
from prophet.exchange import Broker


class TestAccount(TestCase):

    def setUp(self) -> None:
        self.capital_id = '600000'

    def test_cash(self):
        account = Account()
        self.assertEqual(account.get_cash(), 0)

        account.set_cash(100)
        self.assertEqual(account.get_cash(), 100)

        account.add_cash(200)
        self.assertEqual(account.get_cash(), 300)

    def test_capital(self):
        account = Account()
        self.assertEqual(account.get_capitals(), {})
        self.assertEqual(account.get_capital(self.capital_id), 0)

        account.set_capital(self.capital_id, 100)
        self.assertEqual(account.get_capital(self.capital_id), 100)

        account.add_capital(self.capital_id, 200)
        self.assertEqual(account.get_capital(self.capital_id), 300)


class TestBroker(TestCase):

    def test_trade(self):
        broker = Broker(0.1, 5)

        account = Account(1000)
        capital_id = '600000'
        prices = {capital_id: 15}
        volume = 20

        broker.trade(account, prices, capital_id, volume)
        self.assertEqual(account.get_cash(), 560)

        broker.trade(account, prices, capital_id, -volume)
        self.assertEqual(account.get_cash(), 740)
