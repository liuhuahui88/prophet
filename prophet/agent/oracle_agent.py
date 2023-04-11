from prophet.agent.abstract_agent import Agent
from prophet.data.data_extractor import DataExtractor
from prophet.data.data_storage import StockDataStorage
from prophet.utils.constant import Const


class OracleAgent(Agent):

    def __init__(self, symbol, storage: StockDataStorage, data_extractor: DataExtractor):
        self.symbol = symbol

        history = storage.load_history(symbol)
        self.indexes = {history.iloc[i].Date: i for i in range(len(history))}

        data = data_extractor.extract(history, ['oracle'])
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
