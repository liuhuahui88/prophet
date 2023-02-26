class Liquidity:

    def __init__(self, price=0, slippage=0, has_ask=True, has_bid=True):
        self.__price = price
        self.__has_ask = has_ask
        self.__has_bid = has_bid
        self.__slippage = slippage

    def bid(self, cash, price):
        ask_price = self.__price + self.__slippage
        if self.__has_ask and ask_price <= price:
            volume = int(cash / ask_price)
            return volume, volume * ask_price
        else:
            return 0, 0

    def ask(self, volume, price):
        bid_price = self.__price - self.__slippage
        if self.__has_bid and bid_price >= price:
            return volume, volume * bid_price
        else:
            return 0, 0
