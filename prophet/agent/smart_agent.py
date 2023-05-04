from prophet.agent.abstract_agent import Agent
from prophet.utils.constant import Const


class SmartAgent(Agent):

    def __init__(self, symbol, cache, delta):
        self.symbol = symbol
        self.cache = cache
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
