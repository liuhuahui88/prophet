from abc import ABC, abstractmethod

from prophet.utils.account import *


class Agent(ABC):

    class Context(ABC):

        @abstractmethod
        def get_account(self) -> Account:
            pass

        @abstractmethod
        def get_prices(self):
            pass

        @abstractmethod
        def get_date(self):
            pass

        @abstractmethod
        def bid(self, symbol, cash=float('inf'), price=float('inf')):
            pass

        @abstractmethod
        def ask(self, symbol, volume=float('inf'), price=0):
            pass

    @abstractmethod
    def handle(self, ctx: Context):
        pass
