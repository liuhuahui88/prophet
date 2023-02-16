from unittest import TestCase

from prophet.utils.account import Account


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
