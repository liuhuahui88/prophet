from prophet.agent.abstract_agent import Agent
from prophet.data.data_extractor import DataExtractor
from prophet.data.data_storage import StockDataStorage
from prophet.utils.constant import Const


class PerfectActionAgent(Agent):

    def __init__(self, symbol, storage: StockDataStorage):
        self.symbol = symbol
        self.history = storage.load_history(symbol)
        self.indexes = {self.history.iloc[i].Date: i for i in range(len(self.history))}

        data_extractor = DataExtractor(['perfect_action_when_empty', 'perfect_action_when_full'])
        data = data_extractor.extract(self.history)
        self.perfect_action_when_empty = data['perfect_action_when_empty']
        self.perfect_action_when_full = data['perfect_action_when_full']

    def handle(self, ctx: Agent.Context):
        index = self.indexes[ctx.get_date()]

        volume = ctx.get_account().get_volume(self.symbol)
        df = self.perfect_action_when_empty if volume == 0 else self.perfect_action_when_full

        action = df['Action'].iloc[index]

        if action == Const.ASK:
            ctx.ask(self.symbol)
        else:
            ctx.bid(self.symbol)
