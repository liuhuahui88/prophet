class Liquidity:

    def __init__(self, price=0, slippage=0, has_ask=True, has_bid=True):
        self.__price = price
        self.__has_ask = has_ask
        self.__has_bid = has_bid
        self.__slippage = slippage

    def get_price(self, volume=0):
        if volume > 0:
            return self.__price + self.__slippage if self.__has_ask else float('inf')
        elif volume < 0:
            return self.__price - self.__slippage if self.__has_bid else 0
        else:
            return self.__price
