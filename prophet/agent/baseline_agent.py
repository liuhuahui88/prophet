from prophet.agent.abstract_agent import Agent


class BaselineAgent(Agent):

    def handle(self, ctx: Agent.Context):
        symbols = ctx.get_prices().keys()
        num_symbols = len(symbols)

        if num_symbols == 0:
            return

        cash = ctx.get_account().get_cash()
        for symbol in symbols:
            ctx.bid(symbol, cash / num_symbols)
