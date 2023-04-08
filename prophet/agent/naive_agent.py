from prophet.agent.abstract_agent import Agent


class NaiveAgent(Agent):

    def __init__(self, symbol, is_selling=False, is_switching=False):
        self.symbol = symbol
        self.is_selling = is_selling
        self.is_switching = is_switching

    def handle(self, ctx: Agent.Context):
        if self.is_selling:
            ctx.ask(self.symbol)
        else:
            ctx.bid(self.symbol)

        if self.is_switching:
            self.is_selling = not self.is_selling
