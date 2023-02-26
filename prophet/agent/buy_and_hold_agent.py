from prophet.agent.abstract_agent import *


class BuyAndHoldAgent(Agent):

    def __init__(self, capital_id):
        self.capital_id = capital_id

    def handle(self, ctx: Agent.Context):
        ctx.bid(self.capital_id)

