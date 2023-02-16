from abc import ABC, abstractmethod

from prophet.exchange import Account


class Agent(ABC):

    class Context(ABC):

        @abstractmethod
        def get_account(self) -> Account:
            pass

        @abstractmethod
        def get_prices(self):
            pass

        @abstractmethod
        def trade(self, capital_id, volume):
            pass

    @abstractmethod
    def handle(self, ctx: Context):
        pass
