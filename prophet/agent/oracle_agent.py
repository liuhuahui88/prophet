from prophet.agent.abstract_agent import Agent
from prophet.data.data_extractor import DataExtractor
from prophet.data.data_storage import StockDataStorage
from prophet.utils.constant import Const


class OracleAgent(Agent):

    def __init__(self, symbol, storage: StockDataStorage, commission_rate):
        self.symbol = symbol
        self.history = storage.load_history(symbol)
        self.indexes = {self.history.iloc[i].Date: i for i in range(len(self.history))}

        data_extractor = DataExtractor(commission_rate)
        data = data_extractor.extract(self.history, ['oracle'])
        self.oracle = data['oracle']

    def handle(self, ctx: Agent.Context):
        index = self.indexes[ctx.get_date()]

        volume = ctx.get_account().get_volume(self.symbol)
        position = 'Empty' if volume == 0 else 'Full'
        action = self.oracle[position + 'Action'].iloc[index]

        if action == Const.ASK:
            ctx.ask(self.symbol)
        else:
            ctx.bid(self.symbol)
