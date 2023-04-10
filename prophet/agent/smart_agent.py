from prophet.agent.abstract_agent import Agent
from prophet.data.data_collector import DataCollector
from prophet.data.data_predictor import DataPredictor
from prophet.utils.constant import Const


class SmartAgent(Agent):

    def __init__(self, symbol, data_predictor: DataPredictor, name, column, delta):
        self.symbol = symbol
        self.data_collector = DataCollector(self.symbol)
        self.data_predictor = data_predictor
        self.name = name
        self.column = column
        self.delta = delta

    def handle(self, ctx: Agent.Context):
        score = self.predict(ctx)

        action = Const.BID if score > 0 else Const.ASK

        if action == Const.ASK:
            ctx.ask(self.symbol)
        else:
            ctx.bid(self.symbol)

    def predict(self, ctx: Agent.Context):
        self.data_collector.feed(ctx)

        # accelerate the prediction by processing the latest history only
        history = self.data_collector.get().tail(Const.WINDOW_SIZE)

        results = self.data_predictor.predict(history)

        # select the score for the last sample in prediction result
        score = results[self.name].iloc[-1, self.column]

        if ctx.get_account().get_volume(self.symbol) != 0:
            score += self.delta

        return score
