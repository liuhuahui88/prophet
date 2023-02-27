class Account:

    def __init__(self, cash=0, volumes=None):
        if volumes is None:
            volumes = dict()
        self.__cash = cash
        self.__volumes = volumes

    def add_cash(self, delta_cash):
        cash = self.get_cash()
        new_cash = cash + delta_cash
        self.set_cash(new_cash)

    def set_cash(self, cash):
        self.__validate_amount('cash', cash)
        self.__cash = cash

    def get_cash(self):
        return self.__cash

    def add_volume(self, symbol, delta_volume):
        volume = self.get_volume(symbol)
        new_volume = volume + delta_volume
        self.set_volume(symbol, new_volume)

    def set_volume(self, symbol, volume):
        self.__validate_amount('volume', volume)
        if volume == 0 and symbol in self.__volumes:
            del self.__volumes[symbol]
        else:
            self.__volumes[symbol] = volume

    def get_volume(self, symbol):
        return self.__volumes.get(symbol, 0)

    def get_volumes(self):
        return self.__volumes

    def __str__(self):
        return str('cash={}, volumes={}'.format(self.__cash, self.__volumes))

    @staticmethod
    def __validate_amount(name, amount):
        if amount < 0:
            raise ValueError('negative amount of {0} is invalid: {1}'.format(name, amount))
