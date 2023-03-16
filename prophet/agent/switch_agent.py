from prophet.agent.abstract_agent import Agent


class SwitchAgent(Agent):

    def __init__(self, symbol, is_selling=False):
        self.symbol = symbol
        self.is_selling = is_selling

    def handle(self, ctx: Agent.Context):
        if self.is_selling:
            ctx.ask(self.symbol)
        else:
            ctx.bid(self.symbol)

        self.is_selling = not self.is_selling
