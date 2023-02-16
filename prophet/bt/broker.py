from prophet.utils.account import *


class Broker:

    def __init__(self, commission_rate=0):
        self.__commission_rate = commission_rate

    def trade(self, account: Account, capital_id, volume, price):
        cost = volume * price
        commission = abs(cost) * self.__commission_rate

        account.add_cash(-(cost + commission))
        account.add_capital(capital_id, volume)
