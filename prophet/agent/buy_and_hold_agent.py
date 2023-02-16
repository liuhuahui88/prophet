from prophet.agent.abstract_agent import *


class BuyAndHoldAgent(Agent):

    def __init__(self, capital_id):
        self.capital_id = capital_id

    def handle(self, ctx: Agent.Context):
        cash = ctx.get_account().get_cash()
        price = ctx.get_prices()[self.capital_id]
        volume = int(cash / price)
        ctx.trade(self.capital_id, volume)

