from unittest import TestCase

from prophet.utils.account import Account


class TestAccount(TestCase):

    def setUp(self) -> None:
        self.symbol = '600000'

    def test_cash(self):
        account = Account()
        self.assertEqual(account.get_cash(), 0)

        account.set_cash(100)
        self.assertEqual(account.get_cash(), 100)

        account.add_cash(200)
        self.assertEqual(account.get_cash(), 300)

    def test_volume(self):
        account = Account()
        self.assertEqual(account.get_volumes(), {})
        self.assertEqual(account.get_volume(self.symbol), 0)

        account.set_volume(self.symbol, 0)
        self.assertEqual(account.get_volume(self.symbol), 0)

        account.set_volume(self.symbol, 100)
        self.assertEqual(account.get_volume(self.symbol), 100)

        account.add_volume(self.symbol, 200)
        self.assertEqual(account.get_volume(self.symbol), 300)
