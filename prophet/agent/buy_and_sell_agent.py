from prophet.agent.abstract_agent import *


class BuyAndSellAgent(Agent):

    def __init__(self, capital_id):
        self.capital_id = capital_id
        self.is_selling = False

    def handle(self, ctx: Agent.Context):
        if self.is_selling:
            volume = ctx.get_account().get_capital(self.capital_id)
            ctx.trade(self.capital_id, -volume)
        else:
            cash = ctx.get_account().get_cash()
            price = ctx.get_prices()[self.capital_id]
            volume = int(cash / price)
            ctx.trade(self.capital_id, volume)

        self.is_selling = not self.is_selling
