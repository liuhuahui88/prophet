from unittest import TestCase

from prophet.exchange import Account
from prophet.exchange import Liquidity
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


class TestLiquidity(TestCase):

    def test_get_price(self):
        liquidity = Liquidity(100, slippage=1)
        self.assertEqual(liquidity.get_price(0), 100)
        self.assertEqual(liquidity.get_price(1), 101)
        self.assertEqual(liquidity.get_price(-1), 99)

        no_ask_liquidity = Liquidity(100, has_ask=False)
        self.assertEqual(no_ask_liquidity.get_price(0), 100)
        self.assertEqual(no_ask_liquidity.get_price(1), float('inf'))
        self.assertEqual(no_ask_liquidity.get_price(-1), 100)

        no_bid_liquidity = Liquidity(100, has_bid=False)
        self.assertEqual(no_bid_liquidity.get_price(0), 100)
        self.assertEqual(no_bid_liquidity.get_price(1), 100)
        self.assertEqual(no_bid_liquidity.get_price(-1), 0)


class TestBroker(TestCase):

    def test_trade(self):
        broker = Broker(0.1)

        account = Account(1000)
        capital_id = '600000'

        broker.trade(account, capital_id, 20, 20)
        self.assertEqual(account.get_cash(), 560)

        broker.trade(account, capital_id, -20, 20)
        self.assertEqual(account.get_cash(), 920)


class TestAgent(TestCase):

    def test_handle(self):
        capital_id = '600000'

        agent = Agent(capital_id)
        broker = Broker()
        account = Account(1000)

        agent.handle(Context(broker, account, {capital_id: Liquidity(300)}))
        self.assertEqual(account.get_cash(), 100)
        self.assertEqual(account.get_capital(capital_id), 3)

        agent.handle(Context(broker, account, {capital_id: Liquidity(30)}))
        self.assertEqual(account.get_cash(), 10)
        self.assertEqual(account.get_capital(capital_id), 6)


class TestExchange(TestCase):

    def test(self):
        exchange = Exchange()

        capital_id = '600000'

        exchange.register(Agent(capital_id), Broker(), Account(1000))

        exchange.broadcast({capital_id: Liquidity(100)})
        self.assertEqual(exchange.account.get_cash(), 0)
        self.assertEqual(exchange.account.get_capital(capital_id), 10)
