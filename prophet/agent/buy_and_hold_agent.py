from prophet.agent.abstract_agent import Agent


class BuyAndHoldAgent(Agent):

    def __init__(self, symbol):
        self.symbol = symbol

    def handle(self, ctx: Agent.Context):
        ctx.bid(self.symbol)

