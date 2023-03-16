from prophet.agent.abstract_agent import Agent
from prophet.data.data_extractor import DataExtractor
from prophet.data.data_storage import StockDataStorage
from prophet.utils.constant import Const


class OracleAgent(Agent):

    CASH_UPPER_BOUND = 100000000

    def __init__(self, symbol, storage: StockDataStorage):
        self.symbol = symbol
        self.history = storage.load_history(symbol)
        self.indexes = {self.history.iloc[i].Date: i for i in range(len(self.history))}

        data_extractor = DataExtractor(['indicator_of_bid', 'indicator_of_ask'])
        data = data_extractor.extract(self.history)
        self.indicator_of_bid = data['indicator_of_bid']
        self.indicator_of_ask = data['indicator_of_ask']

    def handle(self, ctx: Agent.Context):
        if ctx.get_account().get_cash() > OracleAgent.CASH_UPPER_BOUND:
            return

        idx = self.indexes[ctx.get_date()]

        if self.indicator_of_ask['Indicator'].iloc[idx] == Const.DOWN:
            ctx.ask(self.symbol)
        elif self.indicator_of_bid['Indicator'].iloc[idx] == Const.UP:
            ctx.bid(self.symbol)
