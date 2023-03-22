from prophet.agent.abstract_agent import Agent
from prophet.data.data_extractor import DataExtractor
from prophet.data.data_storage import StockDataStorage
from prophet.utils.constant import Const


class PerfectIndicatorAgent(Agent):

    def __init__(self, symbol, storage: StockDataStorage, commission_rate):
        self.symbol = symbol
        self.history = storage.load_history(symbol)
        self.indexes = {self.history.iloc[i].Date: i for i in range(len(self.history))}

        data_extractor = DataExtractor(commission_rate)
        data = data_extractor.extract(self.history, ['perfect_indicator_of_bid', 'perfect_indicator_of_ask'])
        self.perfect_indicator_of_bid = data['perfect_indicator_of_bid']
        self.perfect_indicator_of_ask = data['perfect_indicator_of_ask']

    def handle(self, ctx: Agent.Context):
        idx = self.indexes[ctx.get_date()]

        if self.perfect_indicator_of_ask['Indicator'].iloc[idx] == Const.DOWN:
            ctx.ask(self.symbol)
        elif self.perfect_indicator_of_bid['Indicator'].iloc[idx] == Const.UP:
            ctx.bid(self.symbol)
