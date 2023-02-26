from prophet.agent.abstract_agent import *


class BuyAndSellAgent(Agent):

    def __init__(self, capital_id, is_selling=False):
        self.capital_id = capital_id
        self.is_selling = is_selling

    def handle(self, ctx: Agent.Context):
        if self.is_selling:
            ctx.ask(self.capital_id)
        else:
            ctx.bid(self.capital_id)

        self.is_selling = not self.is_selling
