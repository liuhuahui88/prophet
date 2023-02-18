from unittest import TestCase

from prophet.bt.liquidity import *


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