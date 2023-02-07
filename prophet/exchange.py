from __future__ import annotations


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
        if amount == 0:
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


class Broker:

    def __init__(self, commission_rate=0, slippage=0):
        self.__commission_rate = commission_rate
        self.__slippage = slippage

    def trade(self, account, prices, capital_id, volume):
        price = prices.get(capital_id, 0)
        slipped_price = price + self.__slippage * (1 if volume >= 0 else -1)

        capital_value = slipped_price * volume
        commission = abs(capital_value) * self.__commission_rate

        account.add_cash(-(capital_value + commission))
        account.add_capital(capital_id, volume)


class Context:

    def __init__(self, broker: Broker, account: Account, prices: dict):
        self.broker = broker
        self.account = account
        self.prices = prices

    def trade(self, capital_id, volume):
        self.broker.trade(self.account, self.prices, capital_id, volume)


class Agent:

    def __init__(self, capital_id):
        self.capital_id = capital_id

    def handle(self, ctx: Context):
        cash = ctx.account.get_cash()
        price = ctx.prices.get(self.capital_id, 0)
        if price != 0:
            volume = int(cash / price)
            ctx.trade(self.capital_id, volume)


class Exchange:

    agent = None
    broker = None
    account = None

    def register(self, agent: Agent, broker: Broker, account: Account):
        self.agent = agent
        self.broker = broker
        self.account = account

    def broadcast(self, prices):
        self.agent.handle(Context(self.broker, self.account, prices))

