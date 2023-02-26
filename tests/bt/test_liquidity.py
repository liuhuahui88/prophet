from unittest import TestCase

from prophet.bt.liquidity import *


class TestLiquidity(TestCase):

    def test(self):
        liquidity = Liquidity(100, slippage=10)
        self.assertEqual(liquidity.bid(0, 110), (0, 0))
        self.assertEqual(liquidity.bid(1000, 100), (0, 0))
        self.assertEqual(liquidity.bid(1000, 110), (9, 990))
        self.assertEqual(liquidity.bid(1000, 120), (9, 990))
        self.assertEqual(liquidity.ask(0, 90), (0, 0))
        self.assertEqual(liquidity.ask(10, 100), (0, 0))
        self.assertEqual(liquidity.ask(10, 90), (10, 900))
        self.assertEqual(liquidity.ask(10, 80), (10, 900))

        no_ask_liquidity = Liquidity(100, has_ask=False)
        self.assertEqual(no_ask_liquidity.bid(1000, 100), (0, 0))
        self.assertEqual(no_ask_liquidity.ask(10, 100), (10, 1000))

        no_bid_liquidity = Liquidity(100, has_bid=False)
        self.assertEqual(no_bid_liquidity.bid(1000, 100), (10, 1000))
        self.assertEqual(no_bid_liquidity.ask(10, 100), (0, 0))
