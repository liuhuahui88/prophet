class Account:

    def __init__(self, cash=0, capitals=None):
        if capitals is None:
            capitals = dict()
        self.__cash = cash
        self.__capitals = capitals

    def add_cash(self, delta_amount):
        current_amount = self.get_cash()
        new_amount = current_amount + delta_amount
        self.set_cash(new_amount)

    def set_cash(self, amount):
        self.__validate_amount('cash', amount)
        self.__cash = amount

    def get_cash(self):
        return self.__cash

    def add_capital(self, capital_id, delta_amount):
        current_amount = self.get_capital(capital_id)
        new_amount = current_amount + delta_amount
        self.set_capital(capital_id, new_amount)

    def set_capital(self, capital_id, amount):
        self.__validate_amount('capital', amount)
        if amount == 0 and capital_id in self.__capitals:
            del self.__capitals[capital_id]
        else:
            self.__capitals[capital_id] = amount

    def get_capital(self, capital_id):
        return self.__capitals.get(capital_id, 0)

    def get_capitals(self):
        return self.__capitals

    def __str__(self):
        return str('cash={}, capitals={}'.format(self.__cash, self.__capitals))

    @staticmethod
    def __validate_amount(name, amount):
        if amount < 0:
            raise ValueError('negative amount of {0} is invalid: {1}'.format(name, amount))
