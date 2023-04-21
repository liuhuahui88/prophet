from prophet.agent.abstract_agent import Agent


class EnsembleAgent(Agent):

    def __init__(self, agents, top_k):
        self.agents = agents
        self.top_k = top_k

    def handle(self, ctx: Agent.Context):
        symbols, num_symbols = self.select_symbols(ctx)

        for symbol in ctx.get_prices().keys():
            ctx.ask(symbol)

        if num_symbols == 0:
            return

        cash = ctx.get_account().get_cash()
        for symbol in symbols:
            ctx.bid(symbol, int(cash / num_symbols))

    def select_symbols(self, ctx: Agent.Context):
        records = {agent.symbol: agent.predict(ctx) for agent in self.agents}
        records = {k: v for k, v in records.items() if v is not None}

        symbols = list(records.keys())
        scores = list(records.values())

        indexes = sorted(range(len(symbols)), key=lambda i: scores[i], reverse=True)

        num_best_symbols = min(self.top_k, len(symbols))
        best_symbols = [symbols[indexes[i]] for i in range(num_best_symbols)]

        return best_symbols, num_best_symbols
