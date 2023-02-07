from unittest import TestCase

from prophet.exchange import Account
from prophet.exchange import Broker
from prophet.exchange import Agent
from prophet.exchange import Context
from prophet.exchange import Exchange


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


class TestAgent(TestCase):

    def test_handle(self):
        capital_id = '600000'

        agent = Agent(capital_id)
        broker = Broker()
        account = Account(1000)

        agent.handle(Context(broker, account, {capital_id: 300}))
        self.assertEqual(account.get_cash(), 100)
        self.assertEqual(account.get_capital(capital_id), 3)

        agent.handle(Context(broker, account, {capital_id: 30}))
        self.assertEqual(account.get_cash(), 10)
        self.assertEqual(account.get_capital(capital_id), 6)


class TestExchange(TestCase):

    def test(self):
        exchange = Exchange()

        capital_id = '600000'

        exchange.register(Agent(capital_id), Broker(), Account(1000))

        exchange.broadcast({capital_id: 100})
        self.assertEqual(exchange.account.get_cash(), 0)
        self.assertEqual(exchange.account.get_capital(capital_id), 10)
