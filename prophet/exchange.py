from prophet.utils.account import Account


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


class Broker:

    def __init__(self, commission_rate=0):
        self.__commission_rate = commission_rate

    def trade(self, account: Account, capital_id, volume, price):
        cost = volume * price
        commission = abs(cost) * self.__commission_rate

        account.add_cash(-(cost + commission))
        account.add_capital(capital_id, volume)
