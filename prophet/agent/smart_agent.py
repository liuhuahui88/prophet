from prophet.agent.abstract_agent import Agent


class SmartAgent(Agent):

    def __init__(self, caches, delta, threshold, top_k):
        self.caches = caches
        self.delta = delta
        self.threshold = threshold
        self.top_k = top_k

    def handle(self, ctx: Agent.Context):
        symbols, scores, n = self.select_symbols(ctx)

        for symbol in ctx.get_prices().keys():
            ctx.ask(symbol)

        cash = ctx.get_account().get_cash()
        for symbol in symbols:
            ctx.bid(symbol, int(cash / n))

    def select_symbols(self, ctx: Agent.Context):
        date = ctx.get_date()
        account = ctx.get_account()

        records = {}
        for symbol, cache in self.caches.items():
            if date not in cache:
                continue

            score = cache[date]
            if account.get_volume(symbol) > 0:
                score += self.delta

            if score <= self.threshold:
                continue

            records[symbol] = score

        symbols = list(records.keys())
        scores = list(records.values())

        indexes = sorted(range(len(symbols)), key=lambda i: scores[i], reverse=True)

        n = min(self.top_k, len(symbols))
        best_symbols = [symbols[i] for i in indexes[:n]]
        best_scores = [scores[i] for i in indexes[:n]]

        return best_symbols, best_scores, n
