from prophet.agent.abstract_agent import Agent
from prophet.data.data_extractor import DataExtractor
from prophet.data.data_predictor import DataPredictor
from prophet.data.data_storage import StockDataStorage
from prophet.utils.constant import Const


class SmartAgent(Agent):

    def __init__(self, symbol, storage: StockDataStorage, data_extractor: DataExtractor, data_predictor: DataPredictor, delta):
        self.symbol = symbol

        history = storage.load_history(symbol)
        results = data_predictor.predict(history, data_extractor)
        scores = list(results.values())[0].iloc[:, 0]
        self.cache = {history.iloc[i].Date: scores[i] for i in range(len(history))}

        self.delta = delta

    def handle(self, ctx: Agent.Context):
        score = self.predict(ctx)

        if score is None:
            return

        action = Const.BID if score > 0 else Const.ASK

        if action == Const.ASK:
            ctx.ask(self.symbol)
        else:
            ctx.bid(self.symbol)

    def predict(self, ctx: Agent.Context):
        if ctx.get_date() not in self.cache:
            return None

        score = self.cache[ctx.get_date()]

        if ctx.get_account().get_volume(self.symbol) != 0:
            score += self.delta

        return score
