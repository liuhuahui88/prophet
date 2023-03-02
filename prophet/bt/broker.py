from prophet.utils.account import Account


class Broker:

    def __init__(self, commission_rate=0):
        self.__commission_rate = commission_rate

    def trade(self, account: Account, symbol, delta_volume, delta_cash):
        if delta_cash * delta_volume >= 0:
            raise ValueError('delta_volume and delta_cash are invalid: {}, {}'.format(delta_volume, delta_cash))

        commission = self.calculate_commission(abs(delta_cash))
        account.add_cash(delta_cash - commission)
        account.add_volume(symbol, delta_volume)

    def calculate_commission(self, cash):
        if cash <= 0:
            raise ValueError('negative cash is invalid: {}'.format(cash))

        return cash * self.__commission_rate
